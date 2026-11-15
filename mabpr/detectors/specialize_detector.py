"""SPECIALIZE — route an activity to the resources that are actually good at it."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from mabpr.detectors.extra_resources_detector import (
    _operating_window, normalize_resource_id, resource_capacity_hours)

MIN_OBS_PER_RESOURCE = 20
MIN_QUALIFIED_RESOURCES = 4
MIN_SPECIALISTS = 2
MIN_SERVICE_S = 36.0
MIN_SPEEDUP = 1.50
MIN_SEPARATION_Z = 3.0

MIN_MODEL_SPEEDUP = 1.20
TARGET_UTILISATION = 0.80

@dataclass
class SpecializeEvidence:
    """Per-resource performance at one activity, and what restricting it costs."""
    occurrences: int
    qualified_resources: int
    duration_mean_all: float
    duration_mean_specialists: float
    speedup: float
    separation_z: float
    spread_ratio: float
    specialists: list
    specialist_count: int
    arrival_rate_per_h: float
    utilisation_after: float
    projected_service_saving_h: float

    model_speedup: float = 0.0
    model_specialist_s: float = 0.0
    model_other_s: float = 0.0

@dataclass
class SpecializeCandidate:
    candidate_id: str
    heuristic: str
    target_activity: str
    evidence: SpecializeEvidence
    heuristic_context: str
    additional_context: str
    process_fragment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = asdict(self.evidence)
        return d

def choose_specialists(per_resource: dict, occurrences: int, capacity_h: dict,
                       window_h: float,
                       *, target_utilisation: float = TARGET_UTILISATION) -> dict:
    """The smallest fastest subset that can still carry the activity's whole load."""

    ranked = sorted(per_resource.items(), key=lambda kv: kv[1]["mean"])
    if not ranked:
        return {"specialists": [], "k": 0, "mean_s": 0.0, "utilisation_after": 0.0,
                "infeasible": True}
    mean_all_s = statistics.fmean([v["mean"] for v in per_resource.values()])
    demand_h = occurrences * (mean_all_s / 3600.0)

    def rho_for(chosen):
        mean_s = statistics.fmean([c[1]["mean"] for c in chosen])
        avail_h = sum(capacity_h.get(str(c[0]), 0.0) for c in chosen)
        if avail_h <= 0:
            avail_h = window_h * len(chosen)
        return mean_s, demand_h / avail_h

    for k in range(max(MIN_SPECIALISTS, 1), len(ranked) + 1):
        mean_s, rho = rho_for(ranked[:k])
        if rho <= target_utilisation:
            return {"specialists": [c[0] for c in ranked[:k]], "k": k,
                    "mean_s": mean_s, "utilisation_after": rho}

    mean_s, rho = rho_for(ranked)
    return {"specialists": [], "k": 0, "mean_s": mean_s,
            "utilisation_after": rho, "infeasible": True}

def _per_activity(df: pd.DataFrame) -> dict:
    """{activity: (per-resource stats, occurrences, arrival rate)} from the real log."""
    act = "concept:name" if "concept:name" in df.columns else "activity"
    res = "org:resource" if "org:resource" in df.columns else "resource"
    end = "time:timestamp" if "time:timestamp" in df.columns else "end_time"
    start = "start_timestamp" if "start_timestamp" in df.columns else "start_time"

    case = ("case:concept:name" if "case:concept:name" in df.columns
            else "case_id" if "case_id" in df.columns else None)
    d = pd.DataFrame({
        "act": df[act], "res": df[res] if res in df.columns else "NOT_SET",
        "start": pd.to_datetime(df[start], format="mixed", utc=True),
        "end": pd.to_datetime(df[end], format="mixed", utc=True),
    })
    if case:
        d["case_id"] = df[case].values
    d["dur_s"] = (d["end"] - d["start"]).dt.total_seconds().clip(lower=0)

    d = d.rename(columns={"start": "start_time", "end": "end_time",
                          "dur_s": "_dur_s"})
    d["serv_h"] = d["_dur_s"] / 3600.0
    d, window_h, window_note = _operating_window(d)
    d = d.rename(columns={"start_time": "start", "end_time": "end",
                          "_dur_s": "dur_s"})

    out = {}
    for a, g in d.groupby("act"):
        stats = {}
        for r, gr in g.groupby("res"):
            if len(gr) < MIN_OBS_PER_RESOURCE:
                continue
            stats[str(r)] = {
                "mean": float(gr["dur_s"].mean()),
                "sd": float(gr["dur_s"].std(ddof=1)) if len(gr) > 1 else 0.0,
                "n": int(len(gr)),
            }
        out[a] = (stats, len(g), len(g) / window_h, float(g["dur_s"].mean()))
    return out, window_h, window_note

def model_speedup(params: dict, activity: str, specialists: set,
                  bpmn_path: str | None) -> dict:
    """What the SIMULATED model thinks the specialists' speed advantage is."""

    if not params or not bpmn_path or not Path(bpmn_path).exists():
        return {}
    import pm4py

    bpmn = pm4py.read_bpmn(str(bpmn_path))
    node = next((n for n in bpmn.get_nodes() if n.get_name() == activity), None)
    if node is None:
        return {}
    tid = str(node.get_id())
    entry = next((e for e in params.get("task_resource_distribution") or []
                  if str(e.get("task_id")) == tid), None)
    if not entry:
        return {}

    means = {}
    for r in entry.get("resources") or []:
        p = r.get("distribution_params") or []
        if p:
            means[normalize_resource_id(r.get("resource_id"))] = float(
                p[0].get("value", 0.0))
    spec = [v for k, v in means.items() if k in specialists and v > 0]
    rest = [v for k, v in means.items() if k not in specialists and v > 0]
    if not spec or not rest:
        return {}
    return {"model_specialist_s": round(statistics.fmean(spec), 2),
            "model_other_s": round(statistics.fmean(rest), 2),
            "model_speedup": round(statistics.fmean(rest)
                                   / max(1e-9, statistics.fmean(spec)), 3)}

def detect_specialize(store, df: pd.DataFrame, *, context: dict | None = None,
                      report: list | None = None) -> list[SpecializeCandidate]:
    """Activities whose fast resources are measurably faster AND sufficient."""
    params = (context or {}).get("params") or {}
    per_activity, window_h, window_note = _per_activity(df)
    capacity_h = resource_capacity_hours(params, window_h)
    kept: list[SpecializeCandidate] = []
    for a, (per_res, n, arrival, overall_mean) in sorted(per_activity.items()):
        reasons = []
        if overall_mean < MIN_SERVICE_S:
            reasons.append(f"mean service {overall_mean:.1f} s — a status event has no "
                           f"expertise to concentrate")
        if len(per_res) < MIN_QUALIFIED_RESOURCES:
            reasons.append(
                f"only {len(per_res)} resources have >= {MIN_OBS_PER_RESOURCE} "
                f"observations, so there is no pool to select within (a 'fast' "
                f"resource measured on a handful of events is a small sample, "
                f"not a skill)")
        if reasons:
            if report is not None:
                report.append({"activity": a, "rejected_because": reasons})
            continue

        pick = choose_specialists(per_res, n, capacity_h, window_h)
        means = [v["mean"] for v in per_res.values()]
        mean_all = statistics.fmean(means)
        if pick.get("infeasible") or not pick["specialists"]:
            reasons.append(
                f"no subset can carry the load: even all {len(per_res)} resources "
                f"run at {pick['utilisation_after']:.0%} of the "
                f"{TARGET_UTILISATION:.0%} target, so this is a capacity problem "
                f"(Extra Resources), not a specialisation one")
        else:
            speedup = mean_all / max(1e-9, pick["mean_s"])

            spec_sds = [per_res[r]["sd"] for r in pick["specialists"]]
            spec_ns = [per_res[r]["n"] for r in pick["specialists"]]
            se_spec = math.sqrt(sum((s * s) / max(1, k) for s, k in
                                    zip(spec_sds, spec_ns))) / max(1, len(spec_ns))
            se_all = (statistics.pstdev(means) / math.sqrt(len(means))
                      if len(means) > 1 else 0.0)
            se = math.sqrt(se_spec ** 2 + se_all ** 2)
            z = ((mean_all - pick["mean_s"]) / se) if se > 0 else 0.0
            if speedup < MIN_SPEEDUP:
                reasons.append(f"specialists are only {speedup:.2f}x faster "
                               f"(<{MIN_SPEEDUP}x) — not worth losing the pool for")
            if abs(z) < MIN_SEPARATION_Z:
                reasons.append(f"the speed gap is z={z:.2f}, inside the noise "
                               f"(|z|>{MIN_SEPARATION_Z} required) — these resources "
                               f"are not measurably different, they are differently "
                               f"sampled")

        ms = {}
        if not reasons:
            ms = model_speedup(params, a, {str(s) for s in pick["specialists"]},
                               (context or {}).get("baseline_bpmn"))
        if reasons:
            if report is not None:
                report.append({"activity": a, "rejected_because": reasons,
                               "qualified_resources": len(per_res),
                               **({"model": ms} if ms else {})})
            continue

        ev = SpecializeEvidence(
            occurrences=int(n), qualified_resources=len(per_res),
            duration_mean_all=round(mean_all, 2),
            duration_mean_specialists=round(pick["mean_s"], 2),
            speedup=round(speedup, 3), separation_z=round(z, 2),
            spread_ratio=round(max(means) / max(1e-9, min(means)), 2),
            specialists=list(pick["specialists"]), specialist_count=pick["k"],
            arrival_rate_per_h=round(arrival, 5),
            utilisation_after=round(pick["utilisation_after"], 4),
            projected_service_saving_h=round((mean_all - pick["mean_s"]) / 3600.0, 5),
            model_speedup=ms.get("model_speedup", 0.0),
            model_specialist_s=ms.get("model_specialist_s", 0.0),
            model_other_s=ms.get("model_other_s", 0.0),
        )
        kept.append(SpecializeCandidate(
            candidate_id="", heuristic="Specialize", target_activity=a, evidence=ev,
            heuristic_context=(
                "Target problem types: inefficient execution due to generalist "
                "resources performing diverse tasks without developing expertise. "
                "Applicable when the same activity is performed by many different "
                "resources with varying performance levels."),
            additional_context=(
                f"{n} executions spread over {len(per_res)} resources with enough "
                f"observations to rank ({max(means):.0f} s slowest against "
                f"{min(means):.0f} s fastest, {ev.spread_ratio:.1f}x). Restricting the "
                f"activity to its {pick['k']} fastest brings the mean from "
                f"{mean_all:.0f} s to {pick['mean_s']:.0f} s, a {speedup:.2f}x speedup "
                f"separated from the rest at z={z:.1f}. That subset would run at "
                f"{pick['utilisation_after']:.0%} utilisation against a "
                f"{TARGET_UTILISATION:.0%} target, so it can still carry the arrival "
                f"rate of {arrival:.2f}/h. ⚠️ This trades servers for speed: the "
                f"projected {ev.projected_service_saving_h:.4f} h per occurrence is a "
                f"SERVICE-time saving only, and whether the smaller pool queues is "
                f"for the simulation to decide, not the detector."),
        ))
    for i, c in enumerate(kept, start=1):
        c.candidate_id = f"SPC-{i:03d}"
    return kept

def print_candidates(cands: list[SpecializeCandidate]) -> None:
    print(f"\nSpecialize: {len(cands)} candidate(s)")
    for c in cands:
        e = c.evidence
        print(f"  {c.candidate_id}  {c.target_activity}")
        print(f"      {e.qualified_resources} -> {e.specialist_count} resources | "
              f"{e.duration_mean_all:.0f}s -> {e.duration_mean_specialists:.0f}s "
              f"({e.speedup:.2f}x, z={e.separation_z:.1f}) | "
              f"utilisation after {e.utilisation_after:.0%}")

def save_candidates_json(cands: list[SpecializeCandidate], path: str,
                         report: list | None = None) -> None:
    Path(path).write_text(json.dumps(
        {"heuristic": "Specialize",
         "candidates": [c.to_dict() for c in cands],
         "rejected": report or []},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
