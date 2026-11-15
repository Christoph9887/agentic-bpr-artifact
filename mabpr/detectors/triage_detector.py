"""TRIAGE — give a case type its own version of an activity, sized to its effort."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from mabpr.detectors.case_types_detector import CASE_ID_COL, candidate_splits

MIN_SERVICE_S = 36.0
MIN_OCCURRENCES = 100
MIN_PER_GROUP = 30
MIN_EFFORT_RATIO = 1.5
MIN_SEPARATION_Z = 3.0

@dataclass
class TriageEvidence:
    activity: str
    attribute: str
    rule_label: str
    comparison: str
    value: str
    occurrences: int
    n_in: int
    n_out: int
    duration_mean_in_s: float
    duration_mean_out_s: float
    effort_ratio: float
    separation_z: float

@dataclass
class TriageCandidate:
    candidate_id: str
    heuristic: str
    target_activity: str
    evidence: TriageEvidence
    heuristic_context: str
    additional_context: str
    process_fragment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = asdict(self.evidence)
        return d

def detect_triage(store, df: pd.DataFrame, *, context: dict | None = None,
                  report: list | None = None) -> list:
    """Activities whose EFFORT differs measurably between case types."""
    params = (context or {}).get("params") or {}
    splits = candidate_splits(df, params)
    if not splits:
        if report is not None:
            report.append({"activity": "*", "rejected_because": [
                "no case attribute is both in the log and declared in "
                "params.case_attributes, so there is no type to triage on"]})
        return []

    act = "concept:name" if "concept:name" in df.columns else "activity"
    end = "time:timestamp" if "time:timestamp" in df.columns else "end_time"
    start = "start_timestamp" if "start_timestamp" in df.columns else "start_time"
    d = pd.DataFrame({
        "case": df[CASE_ID_COL], "act": df[act],
        "s": pd.to_datetime(df[start], format="mixed", utc=True),
        "e": pd.to_datetime(df[end], format="mixed", utc=True)})
    d["dur"] = (d["e"] - d["s"]).dt.total_seconds().clip(lower=0)

    best = {}
    for a, g in d.groupby("act"):
        if len(g) < MIN_OCCURRENCES or float(g["dur"].mean()) < MIN_SERVICE_S:
            if report is not None:
                report.append({"activity": a, "occurrences": int(len(g)),
                               "mean_s": round(float(g["dur"].mean()), 1),
                               "rejected_because": [
                                   f"{len(g)} occurrences at {g['dur'].mean():.1f} s "
                                   f"mean — a status event or too rare to have an "
                                   f"effort profile per case type"]})
            continue
        for sp in splits:
            gi = g[g["case"].isin(sp["in"])]["dur"]
            go = g[g["case"].isin(sp["out"])]["dur"]
            if len(gi) < MIN_PER_GROUP or len(go) < MIN_PER_GROUP:
                continue
            se = math.sqrt(gi.var(ddof=1) / len(gi) + go.var(ddof=1) / len(go))
            if not se or se <= 0:
                continue
            mi, mo = float(gi.mean()), float(go.mean())
            ratio = max(mi, mo) / max(1e-9, min(mi, mo))
            z = (mo - mi) / se
            rec = {"activity": a, "split": sp["label"], "n_in": int(len(gi)),
                   "n_out": int(len(go)), "mean_in_s": round(mi, 1),
                   "mean_out_s": round(mo, 1), "ratio": round(ratio, 2),
                   "z": round(z, 2)}
            reasons = []
            if ratio < MIN_EFFORT_RATIO:
                reasons.append(f"the two types differ by only {ratio:.2f}x "
                               f"(<{MIN_EFFORT_RATIO}x) — one activity serves both")
            if abs(z) < MIN_SEPARATION_Z:
                reasons.append(
                    f"the effort gap is z={z:.2f}, inside the noise "
                    f"(|z|>{MIN_SEPARATION_Z} required). Service durations here are "
                    f"heavy-tailed, so a {ratio:.2f}x ratio between two noisy means "
                    f"is not a measurable difference — the same reason Specialize "
                    f"rejects a 19.52x speedup at z=2.08")
            if reasons:
                if report is not None:
                    report.append({**rec, "rejected_because": reasons})
                continue
            prev = best.get(a)
            if prev is None or abs(z) > abs(prev[1]["z"]):
                best[a] = (sp, rec)

    kept = []
    for a, (sp, rec) in sorted(best.items(), key=lambda kv: -abs(kv[1][1]["z"])):
        slow_in = rec["mean_in_s"] > rec["mean_out_s"]
        ev = TriageEvidence(
            activity=a, attribute=sp["attribute"], rule_label=sp["label"],
            comparison=sp["comparison"], value=sp["value"],
            occurrences=rec["n_in"] + rec["n_out"], n_in=rec["n_in"],
            n_out=rec["n_out"], duration_mean_in_s=rec["mean_in_s"],
            duration_mean_out_s=rec["mean_out_s"], effort_ratio=rec["ratio"],
            separation_z=abs(rec["z"]))
        kept.append(TriageCandidate(
            candidate_id="", heuristic="Triage", target_activity=a, evidence=ev,
            heuristic_context=(
                "Target problem types: inefficient processing caused by treating all "
                "cases equally despite differences in complexity or required effort. "
                "Applicable when cases with different processing requirements are "
                "handled by the same activity without differentiation."),
            additional_context=(
                f"At {a!r}, cases where {sp['label']} take "
                f"{rec['mean_in_s']:.0f} s against {rec['mean_out_s']:.0f} s for the "
                f"rest ({rec['ratio']:.2f}x, z={abs(rec['z']):.1f}, "
                f"{rec['n_in']} vs {rec['n_out']} executions) — the "
                f"{'first' if slow_in else 'second'} group needs materially more "
                f"effort. Splitting the activity per type lets each version carry its "
                f"own fitted duration instead of one pooled distribution. ⚠️ This is "
                f"the ONE heuristic here that cannot be a parameter edit: a Prosimos "
                f"task holds exactly one service distribution, so the versions must "
                f"be real nodes behind a conditioned split."),
            process_fragment={"attribute": sp["attribute"],
                              "comparison": sp["comparison"], "value": sp["value"],
                              "target_activity": a}))
    for i, c in enumerate(kept, start=1):
        c.candidate_id = f"TRI-{i:03d}"
    return kept

def print_candidates(cands: list) -> None:
    print(f"\nTriage: {len(cands)} candidate(s)")
    for c in cands:
        e = c.evidence
        print(f"  {c.candidate_id}  {e.activity} | {e.rule_label}")
        print(f"      {e.duration_mean_in_s:.0f}s vs {e.duration_mean_out_s:.0f}s "
              f"({e.effort_ratio:.2f}x, z={e.separation_z:.1f})")

def save_candidates_json(cands: list, path: str, report: list | None = None) -> None:
    Path(path).write_text(json.dumps(
        {"heuristic": "Triage", "candidates": [c.to_dict() for c in cands],
         "rejected": report or []},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
