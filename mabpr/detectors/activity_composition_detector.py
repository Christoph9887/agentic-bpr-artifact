"""ACTIVITY COMPOSITION — merge fragmented consecutive work into one task."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

MIN_PAIR_FREQUENCY = 30
MIN_FOLLOW_PROB = 0.70
MIN_POOL_OVERLAP = 0.05

MIN_RECOVERABLE_H = 0.05
MIN_RECOVERABLE_SHARE = 0.20

MIN_HANDOVER_RATE = 0.50

@dataclass
class ActivityCompositionEvidence:
    """What the log says about one consecutive pair."""
    pair_frequency: int
    follow_prob_forward: float
    follow_prob_backward: float
    duration_a_mean: float
    duration_b_mean: float
    waiting_between_mean: float
    handover_rate: float
    resource_pool_overlap: float
    recoverable_h_per_occurrence: float
    recoverable_share: float
    cases_affected: int

@dataclass
class ActivityCompositionCandidate:
    candidate_id: str
    heuristic: str
    target_activities: list
    composite_name: str
    evidence: ActivityCompositionEvidence
    heuristic_context: str
    additional_context: str
    process_fragment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = asdict(self.evidence)
        return d

def _pair_stats(df: pd.DataFrame) -> tuple:
    """Directly-follows statistics with resources and gaps, from the real log."""
    act = "concept:name" if "concept:name" in df.columns else "activity"
    res = "org:resource" if "org:resource" in df.columns else "resource"
    case = "case:concept:name" if "case:concept:name" in df.columns else "case_id"
    end = "time:timestamp" if "time:timestamp" in df.columns else "end_time"
    start = "start_timestamp" if "start_timestamp" in df.columns else "start_time"

    d = pd.DataFrame({
        "case": df[case], "act": df[act],
        "res": df[res] if res in df.columns else "NOT_SET",
        "start": pd.to_datetime(df[start], format="mixed", utc=True),
        "end": pd.to_datetime(df[end], format="mixed", utc=True),
    }).sort_values(["case", "start"])
    d["dur_h"] = (d["end"] - d["start"]).dt.total_seconds() / 3600.0

    nxt = d.groupby("case").shift(-1)
    pairs = pd.DataFrame({
        "case": d["case"], "a": d["act"], "b": nxt["act"],
        "res_a": d["res"], "res_b": nxt["res"],
        "gap_h": (nxt["start"] - d["end"]).dt.total_seconds() / 3600.0,
    }).dropna(subset=["b"])

    dur = d.groupby("act")["dur_h"].mean().to_dict()
    pools = d.groupby("act")["res"].apply(lambda s: set(s.dropna())).to_dict()
    out_count = pairs.groupby("a").size().to_dict()
    in_count = pairs.groupby("b").size().to_dict()

    stats = {}
    for (a, b), g in pairs.groupby(["a", "b"]):
        pa, pb = pools.get(a, set()), pools.get(b, set())
        union = len(pa | pb)
        gap = float(g["gap_h"].clip(lower=0).mean())
        da, db = float(dur.get(a, 0.0)), float(dur.get(b, 0.0))
        end_to_end = gap + da + db
        stats[(a, b)] = {

            "share": (gap / end_to_end) if end_to_end > 0 else 0.0,
            "n": len(g),
            "fwd": len(g) / max(1, out_count.get(a, 1)),
            "bwd": len(g) / max(1, in_count.get(b, 1)),
            "gap": gap,
            "handover": float((g["res_a"] != g["res_b"]).mean()),
            "overlap": (len(pa & pb) / union) if union else 0.0,
            "cases": int(g["case"].nunique()),
            "dur_a": da, "dur_b": db,
        }
    return stats, dur

def composable_successor(bpmn_path: str | None, report: list | None = None) -> dict:
    """{activity: the ONE task reachable from it through gateways only}."""

    if not bpmn_path or not Path(bpmn_path).exists():
        return {}
    import pm4py

    bpmn = pm4py.read_bpmn(str(bpmn_path))
    by_id = {str(n.get_id()): n for n in bpmn.get_nodes()}
    out = defaultdict(list)
    inc = defaultdict(list)
    for f in bpmn.get_flows():
        out[str(f.get_source().get_id())].append(str(f.get_target().get_id()))
        inc[str(f.get_target().get_id())].append(str(f.get_source().get_id()))

    def is_task(n) -> bool:
        k = type(n).__name__
        return "Gateway" not in k and "Event" not in k and bool(n.get_name())

    def reachable_tasks(start_id: str, edges) -> set:
        """Named tasks reached from *start_id* through gateways and events only."""
        seen, stack, tasks = {start_id}, list(edges.get(start_id, [])), set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            n = by_id.get(nid)
            if n is None:
                continue
            if is_task(n):
                tasks.add(n.get_name())
            else:
                stack.extend(edges.get(nid, []))
        return tasks

    by_name = {n.get_name(): str(n.get_id())
               for n in bpmn.get_nodes() if is_task(n)}
    result = {}
    for node in bpmn.get_nodes():
        if not is_task(node):
            continue
        a_id, a_name = str(node.get_id()), node.get_name()
        forward = reachable_tasks(a_id, out)
        if len(forward) != 1:
            continue
        b_name = next(iter(forward))
        b_id = by_name.get(b_name)

        if b_id is None:
            continue
        backward = reachable_tasks(b_id, inc)
        if backward - {a_name}:
            if report is not None:
                report.append({
                    "pair": [a_name, b_name],
                    "rejected_because": [
                        f"{b_name!r} is also reachable from "
                        f"{sorted(backward - {a_name})}, so merging it into "
                        f"{a_name!r} would delete that work from every case that "
                        f"arrives the other way"]})
            continue
        result[a_name] = b_name
    return result

def detect_activity_composition(store, df: pd.DataFrame, *,
                                context: dict | None = None,
                                report: list | None = None
                                ) -> list[ActivityCompositionCandidate]:
    """Consecutive small-activity pairs whose merge would recover real cost."""

    stats, dur = _pair_stats(df)
    if not stats:
        return []

    composable = composable_successor((context or {}).get("baseline_bpmn"),
                                      report=report)

    kept: list[ActivityCompositionCandidate] = []
    for (a, b), s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        if a == b:
            continue
        reasons = []
        if s["n"] < MIN_PAIR_FREQUENCY:
            reasons.append(f"only {s['n']} occurrences (<{MIN_PAIR_FREQUENCY})")
        if s["fwd"] < MIN_FOLLOW_PROB or s["bwd"] < MIN_FOLLOW_PROB:
            reasons.append(
                f"not a deterministic sequence: P(next={b!r}|{a!r})={s['fwd']:.2f}, "
                f"P(prev={a!r}|{b!r})={s['bwd']:.2f} — merging a pair that only "
                f"sometimes runs together would force work into cases that skip it")

        if s["overlap"] < MIN_POOL_OVERLAP:
            reasons.append(
                f"resource pools barely overlap (Jaccard {s['overlap']:.2f}) — no "
                f"single resource performs both, so a composite task executed by "
                f"one resource is not staffable")

        if composable:
            succ = composable.get(a)
            if succ != b:
                reasons.append(
                    f"not composable in the baseline BPMN: walking forward from "
                    f"{a!r} through gateways reaches "
                    + (f"{succ!r}, not {b!r}" if succ else
                       "more than one task, so the merge would force "
                       f"{b!r} onto cases that take another branch"))
        if reasons:
            if report is not None:
                report.append({"pair": [a, b], "rejected_because": reasons,
                               "stats": s})
            continue
        ev = ActivityCompositionEvidence(
            pair_frequency=int(s["n"]),
            follow_prob_forward=round(s["fwd"], 4),
            follow_prob_backward=round(s["bwd"], 4),
            duration_a_mean=round(s["dur_a"], 5),
            duration_b_mean=round(s["dur_b"], 5),
            waiting_between_mean=round(s["gap"], 4),
            handover_rate=round(s["handover"], 4),
            resource_pool_overlap=round(s["overlap"], 4),
            recoverable_h_per_occurrence=round(s["gap"], 4),
            recoverable_share=round(s["share"], 4),
            cases_affected=int(s["cases"]),
        )
        kept.append(ActivityCompositionCandidate(
            candidate_id="", heuristic="Activity Composition",
            target_activities=[a, b], composite_name=f"{a} + {b}", evidence=ev,
            heuristic_context=(
                "Target problem types: excessive task fragmentation, repeated setup "
                "effort when switching between small activities, high coordination "
                "overhead. Applicable when a sequence of small activities is "
                "frequently executed together and handled by the same or similar "
                "resources."),
            additional_context=(
                f"{a!r} is directly followed by {b!r} {s['n']} times across "
                f"{s['cases']} cases, and the sequence is near-deterministic in both "
                f"directions ({s['fwd']:.0%} forward, {s['bwd']:.0%} backward). "
                f"⚠️ THE SEAM A MERGE WOULD REMOVE IS {s['gap'] * 3600:.0f} s "
                f"({s['gap']:.4f} h), which is {s['share']:.0%} of the "
                f"{s['gap'] + s['dur_a'] + s['dur_b']:.4f} h the pair occupies end to "
                f"end — decide for yourself whether that is worth a structural edit; "
                f"the detector no longer filters on it. The two activities take "
                f"{s['dur_a']:.4f} h and {s['dur_b']:.4f} h, so when both are ~0 that "
                f"share reads 100% however small the seam is. "
                f"{s['handover']:.0%} of executions change resources. Their pools "
                f"overlap at "
                f"Jaccard {s['overlap']:.2f}, so a single resource can perform the "
                f"merged task. The composite is given the SUM of the two service "
                f"times — no setup saving is assumed, because nothing in the log "
                f"measures how much of a duration is setup."),
        ))
    for i, c in enumerate(kept, start=1):
        c.candidate_id = f"AC-{i:03d}"
    return kept

def print_candidates(cands: list[ActivityCompositionCandidate]) -> None:
    print(f"\nActivity Composition: {len(cands)} candidate(s)")
    for c in cands:
        e = c.evidence
        print(f"  {c.candidate_id}  {c.composite_name}")
        print(f"      n={e.pair_frequency} ({e.cases_affected} cases) | "
              f"fwd {e.follow_prob_forward:.2f} bwd {e.follow_prob_backward:.2f} | "
              f"gap {e.waiting_between_mean:.3f} h | handover {e.handover_rate:.0%} | "
              f"pool overlap {e.resource_pool_overlap:.2f}")

def save_candidates_json(cands: list[ActivityCompositionCandidate], path: str,
                         report: list | None = None) -> None:
    payload = {
        "heuristic": "Activity Composition",
        "candidates": [c.to_dict() for c in cands],

        "rejected": report or [],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")
