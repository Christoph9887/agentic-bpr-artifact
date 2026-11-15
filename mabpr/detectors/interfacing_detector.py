"""INTERFACING — stop information coming back wrong and forcing a redo."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from mabpr.detectors.contact_reduction_detector import prepare
from mabpr.detectors.rework_structure import MIN_CASES, rework_gateways

MIN_TRIGGER_SHARE = 0.10
MIN_REDONE_ACTIVITIES = 1.0
MIN_IDLE_AFTER_H = 0.5
CASE_ID = "case:concept:name"

@dataclass
class InterfacingEvidence:
    trigger_activity: str
    occurrences: int
    cases: int
    cases_triggering_rework: int
    trigger_share: float
    redone_activities_mean: float
    redone_activities_top: list
    idle_after_mean_h: float
    loop_probability: float
    loop_probability_after: float
    waiting_time_before: float
    gateway_id: str

@dataclass
class InterfacingCandidate:
    candidate_id: str
    heuristic: str
    target_activity: str
    evidence: InterfacingEvidence
    heuristic_context: str
    additional_context: str
    process_fragment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = asdict(self.evidence)
        return d

def rework_triggered(d: pd.DataFrame, activity: str) -> dict:
    """After `activity`, how much work the case had ALREADY done gets done again."""

    hit_cases, redone_counts, redone_names = 0, [], {}
    for _, g in d[d["case"].isin(d.loc[d["act"] == activity, "case"].unique())].groupby("case"):
        acts = list(g["act"])
        if activity not in acts:
            continue
        idx = [i for i, a in enumerate(acts) if a == activity]
        redone = set()
        for i in idx:
            before = set(acts[:i]) - {activity}
            after = set(acts[i + 1:])
            redone |= (before & after)
        if redone:
            hit_cases += 1
            for r in redone:
                redone_names[r] = redone_names.get(r, 0) + 1
        redone_counts.append(len(redone))
    n = len(redone_counts)
    return {
        "cases": n,
        "cases_triggering_rework": hit_cases,
        "trigger_share": round(hit_cases / max(1, n), 4),
        "redone_activities_mean": round(
            sum(redone_counts) / max(1, n), 3),
        "redone_activities_top": [k for k, _ in sorted(
            redone_names.items(), key=lambda kv: -kv[1])[:5]],
    }

def detect_interfacing(store, df: pd.DataFrame, *, context: dict | None = None,
                       report: list | None = None) -> list:
    """Events after which the case redoes work it had already completed."""
    params = (context or {}).get("params") or {}
    gateways = rework_gateways((context or {}).get("baseline_bpmn"), params)
    d = prepare(df)

    kept = []
    for a in sorted(d["act"].unique()):
        sub = d[d["act"] == a]
        n_cases = int(sub["case"].nunique())
        if n_cases < MIN_CASES:
            if report is not None:
                report.append({"activity": a, "cases": n_cases, "rejected_because": [
                    f"only {n_cases} cases reach it — too few to call an interface "
                    f"pattern"]})
            continue
        prof = rework_triggered(d, a)
        idle = float(sub["gap_h"].mean() or 0.0)
        rec = {"activity": a, **prof, "idle_after_mean_h": round(idle, 4)}
        reasons = []
        if prof["trigger_share"] < MIN_TRIGGER_SHARE:
            reasons.append(
                f"only {prof['trigger_share']:.1%} of its cases redo any earlier "
                f"activity afterwards (<{MIN_TRIGGER_SHARE:.0%}) — nothing is coming "
                f"back wrong, so there is no interface defect to standardise")
        if prof["redone_activities_mean"] < MIN_REDONE_ACTIVITIES:
            reasons.append(
                f"on average only {prof['redone_activities_mean']:.2f} earlier "
                f"activities are redone after it; that is repetition of the step "
                f"itself (Contact Reduction), not correction of prior work")
        if idle < MIN_IDLE_AFTER_H:
            reasons.append(f"the case idles {idle:.2f} h after it "
                           f"(<{MIN_IDLE_AFTER_H} h) — no external exchange is being "
                           f"waited on")
        if a not in gateways:
            reasons.append("the mined model has no exclusive two-path gateway whose "
                           "loop returns to this activity alone, so the correction "
                           "rate is not a parameter this redesign can lower")
        if reasons:
            if report is not None:
                report.append({**rec, "rejected_because": reasons})
            continue

        gid, loop_path, exit_path, loop_p = gateways[a]

        after = round(max(0.0, loop_p * (1.0 - prof["trigger_share"])), 4)
        ev = InterfacingEvidence(
            trigger_activity=a, occurrences=int(len(sub)), cases=prof["cases"],
            cases_triggering_rework=prof["cases_triggering_rework"],
            trigger_share=prof["trigger_share"],
            redone_activities_mean=prof["redone_activities_mean"],
            redone_activities_top=prof["redone_activities_top"],
            idle_after_mean_h=round(idle, 4),
            loop_probability=round(loop_p, 4), loop_probability_after=after,
            waiting_time_before=round(idle * prof["trigger_share"], 4),
            gateway_id=gid)
        kept.append(InterfacingCandidate(
            candidate_id="", heuristic="Interfacing", target_activity=a, evidence=ev,
            heuristic_context=(
                "Target problem types: errors and delays caused by inconsistent "
                "information exchange with customers or external partners. "
                "Applicable when information is repeatedly corrected, clarified or "
                "re-entered due to inconsistent formats or interfaces."),
            additional_context=(
                f"{a!r} occurs in {prof['cases']} cases, and in "
                f"{prof['trigger_share']:.0%} of them the case afterwards REDOES work "
                f"it had already completed — on average "
                f"{prof['redone_activities_mean']:.2f} distinct earlier activities, "
                f"most often {prof['redone_activities_top'][:3]}. That is the log "
                f"signature of information arriving wrong, and it is what separates "
                f"this from Contact Reduction (where the step merely repeats itself). "
                f"The case then idles {idle:.2f} h. Standardising the exchange lowers "
                f"the correction rate at gateway {gid[:12]} from {loop_p:.0%} to "
                f"{after:.0%}. ⚠️ The idle wait is WALL-CLOCK and the mined model has "
                f"no construct for it, so {ev.waiting_time_before:.3f} h/case is an "
                f"ESTIMATE from the log reported beside the measurement, never added "
                f"to it."),
            process_fragment={"gateway_id": gid, "loop_path_id": loop_path,
                              "exit_path_id": exit_path,
                              "loop_probability_after": after,
                              "target_activity": a}))
    for i, c in enumerate(kept, start=1):
        c.candidate_id = f"IF-{i:03d}"
    return kept

def print_candidates(cands: list) -> None:
    print(f"\nInterfacing: {len(cands)} candidate(s)")
    for c in cands:
        e = c.evidence
        print(f"  {c.candidate_id}  {e.trigger_activity}")
        print(f"      {e.cases} cases, {e.trigger_share:.0%} trigger rework "
              f"({e.redone_activities_mean:.2f} activities redone, e.g. "
              f"{e.redone_activities_top[:2]}) | idle {e.idle_after_mean_h:.2f} h | "
              f"loop {e.loop_probability:.0%} -> {e.loop_probability_after:.0%}")

def save_candidates_json(cands: list, path: str, report: list | None = None) -> None:
    Path(path).write_text(json.dumps(
        {"heuristic": "Interfacing",
         "candidates": [c.to_dict() for c in cands], "rejected": report or []},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
