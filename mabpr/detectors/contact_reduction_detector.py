"""CONTACT REDUCTION — stop asking the same outsider the same thing twice."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from mabpr.detectors.rework_structure import (MIN_CASES, MIN_REPEAT_SHARE,
                                              repeat_profile, rework_gateways)

MIN_IDLE_AFTER_H = 1.0
MIN_IDLE_TO_SERVICE = 5.0
CASE_ID = "case:concept:name"

@dataclass
class ContactEvidence:
    contact_activity: str
    occurrences: int
    cases: int
    cases_repeating: int
    repeat_share: float
    mean_per_case: float
    max_per_case: int
    extra_occurrences: int
    service_h_mean: float
    idle_after_mean_h: float
    idle_to_service_ratio: float
    loop_probability: float
    loop_probability_after: float
    waiting_time_before: float
    gateway_id: str

@dataclass
class ContactCandidate:
    candidate_id: str
    heuristic: str
    target_activity: str
    evidence: ContactEvidence
    heuristic_context: str
    additional_context: str
    process_fragment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = asdict(self.evidence)
        return d

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """case / act / service / gap-after, sorted. The one frame both detectors use."""
    act = "concept:name" if "concept:name" in df.columns else "activity"
    end = "time:timestamp" if "time:timestamp" in df.columns else "end_time"
    start = "start_timestamp" if "start_timestamp" in df.columns else "start_time"
    d = pd.DataFrame({
        "case": df[CASE_ID] if CASE_ID in df.columns else df["case_id"],
        "act": df[act],
        "start": pd.to_datetime(df[start], format="mixed", utc=True),
        "end": pd.to_datetime(df[end], format="mixed", utc=True)})
    d["serv_h"] = (d["end"] - d["start"]).dt.total_seconds().clip(lower=0) / 3600
    d = d.sort_values(["case", "start"]).reset_index(drop=True)
    d["next_start"] = d.groupby("case")["start"].shift(-1)

    d["gap_h"] = ((d["next_start"] - d["end"]).dt.total_seconds() / 3600).clip(lower=0)
    return d

def detect_contact_reduction(store, df: pd.DataFrame, *, context: dict | None = None,
                             report: list | None = None) -> list:
    """Activities the process waits on an outsider for, and asks again."""
    params = (context or {}).get("params") or {}
    gateways = rework_gateways((context or {}).get("baseline_bpmn"), params)
    d = prepare(df)

    kept = []
    for a in sorted(d["act"].unique()):
        prof = repeat_profile(d, a)
        if not prof:
            continue
        serv = float(d.loc[d["act"] == a, "serv_h"].mean())
        idle = prof["idle_after_mean_h"]
        ratio = idle / serv if serv > 0 else (float("inf") if idle > 0 else 0.0)
        rec = {"activity": a, **prof, "service_h_mean": round(serv, 4),
               "idle_to_service_ratio": (round(ratio, 2) if ratio != float("inf")
                                         else "inf")}
        reasons = []
        if prof["cases"] < MIN_CASES:
            reasons.append(f"only {prof['cases']} cases perform it — too few to call "
                           f"a communication pattern")
        if idle < MIN_IDLE_AFTER_H:
            reasons.append(f"the case idles only {idle:.2f} h after it "
                           f"(<{MIN_IDLE_AFTER_H} h), so nothing outside the process "
                           f"is being waited on")
        if ratio < MIN_IDLE_TO_SERVICE:
            reasons.append(f"the idle wait is only {ratio:.1f}x its own service time "
                           f"(<{MIN_IDLE_TO_SERVICE}x) — that is a queue for a "
                           f"resource, not a wait for an outsider")
        if prof["repeat_share"] < MIN_REPEAT_SHARE:
            reasons.append(f"only {prof['repeat_share']:.1%} of cases do it more than "
                           f"once (<{MIN_REPEAT_SHARE:.0%}), so there is no repeated "
                           f"exchange to consolidate")
        if a not in gateways:
            reasons.append("the mined model has no exclusive two-path gateway whose "
                           "loop returns to this activity alone, so there is no "
                           "probability to reduce and the consolidation cannot be "
                           "expressed as a parameter edit")
        if reasons:
            if report is not None:
                report.append({**rec, "rejected_because": reasons})
            continue

        gid, loop_path, exit_path, loop_p = gateways[a]

        after = round(max(0.0, loop_p * (1.0 - prof["repeat_share"])), 4)
        ev = ContactEvidence(
            contact_activity=a, occurrences=prof["occurrences"], cases=prof["cases"],
            cases_repeating=prof["cases_repeating"], repeat_share=prof["repeat_share"],
            mean_per_case=prof["mean_per_case"], max_per_case=prof["max_per_case"],
            extra_occurrences=prof["extra_occurrences"], service_h_mean=round(serv, 4),
            idle_after_mean_h=idle,
            idle_to_service_ratio=(round(ratio, 2) if ratio != float("inf") else 9999.0),
            loop_probability=round(loop_p, 4), loop_probability_after=after,

            waiting_time_before=round(prof["removable_h_per_repeating_case"], 4),
            gateway_id=gid)
        kept.append(ContactCandidate(
            candidate_id="", heuristic="Contact Reduction", target_activity=a,
            evidence=ev,
            heuristic_context=(
                "Target problem types: excessive communication with customers or "
                "external parties, delays caused by repeated information exchanges. "
                "Applicable when multiple interactions occur to collect or verify "
                "information that could be consolidated."),
            additional_context=(
                f"{a!r} runs {prof['occurrences']} times across {prof['cases']} cases, "
                f"and {prof['repeat_share']:.0%} of those cases do it more than once "
                f"(mean {prof['mean_per_case']:.1f}, max {prof['max_per_case']}). After "
                f"each one the case sits IDLE for {idle:.2f} h on average against "
                f"{serv:.3f} h of actual work — the process is waiting on somebody "
                f"outside it, not on a resource. Consolidating the exchange lowers the "
                f"repeat probability at gateway {gid[:12]} from {loop_p:.0%} to "
                f"{after:.0%}. ⚠️ The idle wait is WALL-CLOCK (overnight, weekend, the "
                f"customer's own delay) and the mined model has no construct for it, "
                f"so {ev.waiting_time_before:.3f} h/case is an ESTIMATE from the log, "
                f"not a simulated result — it is reported beside the measurement and "
                f"never added to it."),
            process_fragment={"gateway_id": gid, "loop_path_id": loop_path,
                              "exit_path_id": exit_path,
                              "loop_probability_after": after,
                              "target_activity": a}))
    for i, c in enumerate(kept, start=1):
        c.candidate_id = f"CR-{i:03d}"
    return kept

def print_candidates(cands: list) -> None:
    print(f"\nContact Reduction: {len(cands)} candidate(s)")
    for c in cands:
        e = c.evidence
        print(f"  {c.candidate_id}  {e.contact_activity}")
        print(f"      {e.cases} cases, {e.repeat_share:.0%} repeat "
              f"(mean {e.mean_per_case:.1f}, max {e.max_per_case}) | idle after "
              f"{e.idle_after_mean_h:.2f} h vs {e.service_h_mean:.3f} h service "
              f"({e.idle_to_service_ratio:.0f}x) | loop {e.loop_probability:.0%} -> "
              f"{e.loop_probability_after:.0%}")

def save_candidates_json(cands: list, path: str, report: list | None = None) -> None:
    Path(path).write_text(json.dumps(
        {"heuristic": "Contact Reduction",
         "candidates": [c.to_dict() for c in cands], "rejected": report or []},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
