"""NUMERICAL INVOLVEMENT — stop passing one case around so many hands."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

CASE_ID = "case:concept:name"
MIN_CASES = 100

MIN_FRAGMENTATION = 0.60
MIN_SHARE_FRAGMENTED = 0.15
MIN_PARTICIPANTS = 3

@dataclass
class InvolvementEvidence:
    target_activities: list
    cases: int
    cases_fragmented: int
    fragmented_share: float
    handover_rate_mean: float
    unique_resources_mean: float
    unique_resources_max: int
    events_per_case_mean: float
    pool_before: int
    pool_after: int
    overlaps_case_assignment: list
    overlap_share: float

@dataclass
class InvolvementCandidate:
    candidate_id: str
    heuristic: str
    target_activities: list
    evidence: InvolvementEvidence
    heuristic_context: str
    additional_context: str
    process_fragment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = asdict(self.evidence)
        return d

def _ca_claims(context: dict | None) -> dict:
    """{candidate_id: {activities}} for the Case Assignment candidates, if given."""
    out = {}
    for c in ((context or {}).get("detected_candidates") or []):
        if (c.get("heuristic") or "") != "Case Assignment":
            continue
        acts = set(c.get("target_activities") or [])
        ta = c.get("target_activity")
        if isinstance(ta, str) and ta:
            acts.add(ta)
        if acts:
            out[str(c.get("candidate_id"))] = acts
    return out

def detect_numerical_involvement(store, df: pd.DataFrame, *,
                                 context: dict | None = None,
                                 report: list | None = None) -> list:
    """Runs of consecutive steps that one case is passed around for."""
    act = "concept:name" if "concept:name" in df.columns else "activity"
    res = "org:resource" if "org:resource" in df.columns else "resource"
    start = "start_timestamp" if "start_timestamp" in df.columns else "start_time"
    d = pd.DataFrame({"case": df[CASE_ID], "act": df[act], "res": df[res],
                      "s": pd.to_datetime(df[start], format="mixed", utc=True)})
    d = d.sort_values(["case", "s"]).reset_index(drop=True)

    per_case = d.groupby("case").agg(events=("act", "size"),
                                     uniq=("res", "nunique"))
    handovers = d.groupby("case")["res"].apply(lambda s: int((s != s.shift()).sum()) - 1)
    per_case["handovers"] = handovers
    per_case["frag"] = per_case["handovers"] / (per_case["events"] - 1).clip(lower=1)
    fragmented = set(per_case.index[(per_case["frag"] >= MIN_FRAGMENTATION)
                                    & (per_case["uniq"] >= MIN_PARTICIPANTS)])

    ca = _ca_claims(context)
    kept = []
    for a, g in d.groupby("act"):
        cases = set(g["case"])
        if len(cases) < MIN_CASES:
            continue
        frag_here = cases & fragmented
        share = len(frag_here) / max(1, len(cases))
        sub = per_case.loc[sorted(cases)]
        rec = {"activity": a, "cases": len(cases), "cases_fragmented": len(frag_here),
               "fragmented_share": round(share, 4),
               "handover_rate_mean": round(float(sub["frag"].mean()), 4),
               "unique_resources_mean": round(float(sub["uniq"].mean()), 2)}
        if share < MIN_SHARE_FRAGMENTED:
            if report is not None:
                report.append({**rec, "rejected_because": [
                    f"only {share:.1%} of its cases are passed around "
                    f"(<{MIN_SHARE_FRAGMENTED:.0%} at a handover rate of "
                    f"{MIN_FRAGMENTATION:.0%}) — the coordination overhead this "
                    f"heuristic targets is not present here"]})
            continue

        nb = pd.concat([
            d.groupby("case")["act"].shift(1)[d["act"] == a],
            d.groupby("case")["act"].shift(-1)[d["act"] == a]]).dropna()
        run = [a] + [x for x, _ in nb.value_counts().head(2).items() if x != a]
        pool = set(d.loc[d["act"].isin(run), "res"].dropna().astype(str))
        overlap = sorted(cid for cid, acts in ca.items() if acts & set(run))
        ov_share = 0.0
        if overlap:
            claimed = set().union(*(ca[c] for c in overlap))
            ov_share = len(claimed & set(run)) / max(1, len(run))

        ev = InvolvementEvidence(
            target_activities=run, cases=len(cases),
            cases_fragmented=len(frag_here), fragmented_share=round(share, 4),
            handover_rate_mean=round(float(sub["frag"].mean()), 4),
            unique_resources_mean=round(float(sub["uniq"].mean()), 2),
            unique_resources_max=int(sub["uniq"].max()),
            events_per_case_mean=round(float(sub["events"].mean()), 2),
            pool_before=len(pool), pool_after=len(pool),
            overlaps_case_assignment=overlap, overlap_share=round(ov_share, 4))
        kept.append(InvolvementCandidate(
            candidate_id="", heuristic="Numerical Involvement",
            target_activities=run, evidence=ev,
            heuristic_context=(
                "Target problem types: excessive coordination overhead, delays "
                "caused by involving too many departments or participants in "
                "handling a case. Applicable when a single case is handled by many "
                "different resources or organizational units."),
            additional_context=(
                f"{share:.0%} of the {len(cases)} cases that reach {a!r} are passed "
                f"between a new person at {MIN_FRAGMENTATION:.0%} or more of their "
                f"steps, involving {ev.unique_resources_mean:.1f} people on average "
                f"and up to {ev.unique_resources_max}. ⚠️ The rate is "
                f"LENGTH-CONTROLLED (handovers / transitions), because raw participant "
                f"counts measure how long a case is: unique resources per case tracks "
                f"events per case at 0.29-0.36 on these logs. The redesign widens the "
                f"allocation across {run} so one person can carry the whole stretch — "
                f"{len(pool)} resources currently serve it."
                + (f" ⚠️ OVERLAPS Case Assignment {overlap} on "
                   f"{ov_share:.0%} of these activities: both end in the same "
                   f"parameter (who may perform what), selected on handover RATE here "
                   f"and on handover WAITING there. Applying both is applying one."
                   if overlap else
                   " No Case Assignment candidate claims these activities, so this is "
                   "a stretch CA's waiting-based selection does not reach.")),
            process_fragment={"target_activities": run}))
    for i, c in enumerate(kept, start=1):
        c.candidate_id = f"NI-{i:03d}"
    return kept

def print_candidates(cands: list) -> None:
    print(f"\nNumerical Involvement: {len(cands)} candidate(s)")
    for c in cands:
        e = c.evidence
        print(f"  {c.candidate_id}  {e.target_activities}")
        print(f"      {e.fragmented_share:.0%} of {e.cases} cases fragmented "
              f"(handover rate {e.handover_rate_mean:.2f}, "
              f"{e.unique_resources_mean:.1f} people, max {e.unique_resources_max}) "
              f"| pool {e.pool_before} | CA overlap {e.overlaps_case_assignment or '-'} "
              f"({e.overlap_share:.0%})")

def save_candidates_json(cands: list, path: str, report: list | None = None) -> None:
    Path(path).write_text(json.dumps(
        {"heuristic": "Numerical Involvement",
         "candidates": [c.to_dict() for c in cands], "rejected": report or []},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
