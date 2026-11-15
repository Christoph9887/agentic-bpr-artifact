"""knockout_detector.py"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from .feature_extraction import FeatureStore

HEURISTIC_CONTEXT: str = (
    "Target problem types: unnecessary effort spent on cases that will "
    "eventually be rejected, inefficient ordering of validation or "
    "eligibility checks. "
    "Applicable structural situations: processes containing multiple checks "
    "or validation activities where failure of a condition leads to "
    "termination of the case. "
    "How the data is used: Activities with high rejection probability and "
    "relatively low execution effort should appear earlier in the sequence, "
    "as this minimizes the expected processing cost for cases that will "
    "eventually be rejected. "
    "Transformation style: reorder validation activities so that checks "
    "with the highest rejection probability and lowest effort are executed "
    "earlier in the process. "
    "Feasibility: Maybe. Rejection probabilities and termination patterns "
    "can be derived from event logs, but the effort of checks and the "
    "validity of reordering them depend on domain-specific business rules."
)

@dataclass
class ActivityKnockoutStats:
    activity: str
    termination_rate: float
    activity_duration_mean: float
    knockout_efficiency_ratio: float
    activity_position_ratio: float
    cumulative_effort_before_mean: float

@dataclass
class KnockOutEvidence:
    per_activity: list[ActivityKnockoutStats]
    current_ordering: list[str]
    optimal_ordering: list[str]
    current_ordering_efficiency: float
    optimal_ordering_efficiency: float
    efficiency_gap: float
    efficiency_gap_ratio: float

@dataclass
class KnockOutCandidate:
    candidate_id: str
    heuristic: str
    target_activities: list[str]
    heuristic_context: str
    process_fragment: None
    evidence: KnockOutEvidence
    additional_context: str

TERMINATION_RATE_MIN = 0.05

DURATION_EPSILON = 0.01

MIN_CHECKS = 2

def _compute_knockout_stats(
    df: pd.DataFrame,
    store: FeatureStore,
) -> list[ActivityKnockoutStats]:
    """For each activity, compute:"""

    df_sorted = df.sort_values(
        ["case:concept:name", "time:timestamp"]
    ).copy()

    has_forward_successor: set[str] = set()
    for (a, b), strength in store.control_flow.dependency_strength.items():
        if strength > 0.5:
            has_forward_successor.add(a)

    terminal_activities: set[str] = set()
    for act in store.activities:
        if act not in has_forward_successor:
            terminal_activities.add(act)

    print(f"  Terminal activities (auto-detected): {sorted(terminal_activities)}")

    case_sequences: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for _, row in df_sorted[
        ["case:concept:name", "concept:name", "start_timestamp", "time:timestamp"]
    ].iterrows():
        case_id = str(row["case:concept:name"])
        act = row["concept:name"]
        dur_h = max(
            0.0,
            (row["time:timestamp"] - row["start_timestamp"]).total_seconds() / 3600.0
        )
        case_sequences[case_id].append((act, dur_h))

    act_direct_termination: dict[str, int] = defaultdict(int)
    act_total_count: dict[str, int] = defaultdict(int)
    act_has_nonterminal_succ: set[str] = set()
    act_positions: dict[str, list[float]] = defaultdict(list)
    act_cum_efforts: dict[str, list[float]] = defaultdict(list)

    for case_id, seq in case_sequences.items():
        n = len(seq)
        if n == 0:
            continue

        cumulative = 0.0
        for i, (act, dur_h) in enumerate(seq):
            act_total_count[act] += 1

            pos_ratio = i / (n - 1) if n > 1 else 0.5
            act_positions[act].append(pos_ratio)

            act_cum_efforts[act].append(cumulative)
            cumulative += dur_h

            if i + 1 < n:
                next_act = seq[i + 1][0]
                if next_act in terminal_activities:
                    act_direct_termination[act] += 1
                else:
                    act_has_nonterminal_succ.add(act)

    results: list[ActivityKnockoutStats] = []

    for act in store.activities:
        total = act_total_count.get(act, 0)
        if total == 0:
            continue

        direct_terminated = act_direct_termination.get(act, 0)
        term_rate = direct_terminated / total

        if term_rate < TERMINATION_RATE_MIN:
            continue

        if act in terminal_activities:
            continue

        if act not in act_has_nonterminal_succ:
            continue

        duration = store.time.activity_duration_mean.get(act, 0.0)

        if duration < DURATION_EPSILON:
            continue

        positions = act_positions.get(act, [])
        efforts = act_cum_efforts.get(act, [])

        pos_ratio = sum(positions) / len(positions) if positions else 0.5
        cum_effort = sum(efforts) / len(efforts) if efforts else 0.0

        ko_ratio = term_rate / duration

        results.append(ActivityKnockoutStats(
            activity=act,
            termination_rate=round(term_rate, 4),
            activity_duration_mean=round(duration, 4),
            knockout_efficiency_ratio=round(ko_ratio, 4),
            activity_position_ratio=round(pos_ratio, 4),
            cumulative_effort_before_mean=round(cum_effort, 3),
        ))

    return results

def _compute_expected_cost(
    ordering: list[ActivityKnockoutStats],
) -> float:
    """Compute the expected processing cost for a given ordering of knock-out"""

    survival = 1.0
    total_cost = 0.0

    for stats in ordering:
        total_cost += stats.activity_duration_mean * survival
        survival *= (1.0 - stats.termination_rate)

    return total_cost

def _topological_order_in_bpmn(
    checks: list[ActivityKnockoutStats],
    store: FeatureStore,
) -> list[ActivityKnockoutStats]:
    """Order knock-out checks by their first-visit BFS time from the BPMN"""

    pgf = store.process_graph
    G = pgf.graph

    if G is None or G.number_of_nodes() == 0:
        return sorted(checks, key=lambda s: s.activity_position_ratio)

    start_nodes = [
        n for n in G.nodes
        if G.in_degree(n) == 0
        and pgf.node_metadata.get(n, {}).get("type") == "event"
    ]

    if not start_nodes:
        start_nodes = [n for n in G.nodes if G.in_degree(n) == 0]

    if not start_nodes:
        return sorted(checks, key=lambda s: s.activity_position_ratio)

    visit_time: dict[str, int] = {}
    t = 0
    queue: list[str] = []
    for n in start_nodes:
        visit_time[n] = t
        t += 1
        queue.append(n)
    while queue:
        n = queue.pop(0)
        for succ in G.successors(n):
            if succ in visit_time:
                continue
            visit_time[succ] = t
            t += 1
            queue.append(succ)

    UNREACHABLE = 10**9

    def check_visit_time(stats: ActivityKnockoutStats) -> int:
        node_id = pgf.activity_to_node.get(stats.activity)
        if node_id is None or node_id not in visit_time:
            return UNREACHABLE
        return visit_time[node_id]

    return sorted(
        checks,
        key=lambda s: (check_visit_time(s), s.activity_position_ratio),
    )

def _build_additional_context(
    evidence: KnockOutEvidence,
    store: FeatureStore,
) -> str:
    """Produce a structured natural-language note listing supplementary signals"""

    parts: list[str] = []

    gap = evidence.efficiency_gap
    if gap > 0:
        pct = (
            (gap / evidence.current_ordering_efficiency * 100)
            if evidence.current_ordering_efficiency > 0 else 0.0
        )
        parts.append(
            f"Reordering the knock-out checks from their current sequence "
            f"to the optimal sequence would reduce the expected processing "
            f"cost per case by {gap:.3f} hours ({pct:.1f}% improvement)."
        )
    else:
        parts.append(
            "The current ordering of knock-out checks is already optimal "
            "or near-optimal based on the observed termination rates and "
            "activity durations. No reordering is recommended at this time."
        )

    current = evidence.current_ordering
    optimal = evidence.optimal_ordering
    if current != optimal:

        movers: list[str] = []
        for i, act in enumerate(optimal):
            current_pos = current.index(act) if act in current else i
            if current_pos - i >= 2:
                movers.append(
                    f"'{act}' (currently position {current_pos + 1} → "
                    f"optimal position {i + 1})"
                )
        if movers:
            parts.append(
                "Activities that would benefit most from being moved earlier: "
                + "; ".join(movers) + "."
            )

    if evidence.per_activity:
        best = max(evidence.per_activity, key=lambda s: s.knockout_efficiency_ratio)
        parts.append(
            f"The most efficient knock-out check is '{best.activity}' "
            f"with a termination rate of {best.termination_rate:.1%} "
            f"and duration of {best.activity_duration_mean:.3f} hours "
            f"(efficiency ratio: {best.knockout_efficiency_ratio:.2f})."
        )

    if len(evidence.per_activity) >= 2 and len(evidence.current_ordering) >= 2:
        stats_by_name = {s.activity: s for s in evidence.per_activity}
        earliest = stats_by_name.get(evidence.current_ordering[0])
        latest = stats_by_name.get(evidence.current_ordering[-1])
        if (
            earliest is not None
            and latest is not None
            and earliest.knockout_efficiency_ratio < latest.knockout_efficiency_ratio
        ):
            parts.append(
                f"The earliest check '{earliest.activity}' "
                f"(efficiency ratio {earliest.knockout_efficiency_ratio:.2f}) "
                f"is less efficient than the latest check "
                f"'{latest.activity}' ({latest.knockout_efficiency_ratio:.2f}). "
                "Moving more efficient checks earlier would reduce wasted "
                "effort on cases that will be rejected."
            )

    parts.append(
        "No process_fragment is provided for this candidate because the "
        "Knock-out heuristic targets the ordering of check activities, "
        "not control-flow topology. The sequential reachability of the "
        "check activities has been validated through the DFG."
    )

    parts.append(
        "NOTE: The termination_rate reported for each activity measures "
        "DIRECT knock-out behaviour — the fraction of cases where the "
        "immediately next event after this activity is a terminal activity "
        "(e.g. rejection or cancellation). This is stricter than 'eventual "
        "termination' and confirms the activity itself acts as a decision "
        "gate."
    )

    parts.append(
        "Whether the check ordering can be changed depends on information "
        "availability at each stage, business rules mandating specific "
        "sequences, and regulatory requirements constraining the order of "
        "validation activities — none of which can be determined from the "
        "event log alone."
    )

    return " ".join(parts)

def detect_knockout(
    store: FeatureStore,
    df: pd.DataFrame,
) -> list[KnockOutCandidate]:
    """Detect Knock-out candidates from the event log."""

    all_stats = _compute_knockout_stats(df, store)

    if len(all_stats) < MIN_CHECKS:
        print(f"  Found {len(all_stats)} knock-out checks "
              f"(need ≥ {MIN_CHECKS}). No candidate produced.")
        return []

    print(f"  Knock-out check activities found: {len(all_stats)}")

    check_names = {s.activity for s in all_stats}
    has_connection = False
    for (a, b) in store.control_flow.dfg_frequency:
        if a in check_names and b in check_names:
            has_connection = True
            break
    if not has_connection:

        for act_a in check_names:
            succs_of_a = {
                b for (a, b) in store.control_flow.dfg_frequency if a == act_a
            }
            for act_b in check_names:
                if act_b == act_a:
                    continue
                preds_of_b = {
                    a for (a, b) in store.control_flow.dfg_frequency
                    if b == act_b
                }
                if succs_of_a & preds_of_b:
                    has_connection = True
                    break
            if has_connection:
                break

    if not has_connection:
        print("  No sequential connection found between knock-out checks. "
              "Reordering is not applicable.")
        return []

    current_order = _topological_order_in_bpmn(all_stats, store)
    current_names = [s.activity for s in current_order]

    optimal_order = sorted(
        all_stats, key=lambda s: s.knockout_efficiency_ratio, reverse=True
    )
    optimal_names = [s.activity for s in optimal_order]

    current_cost = _compute_expected_cost(current_order)
    optimal_cost = _compute_expected_cost(optimal_order)
    gap = current_cost - optimal_cost
    gap_ratio = (gap / current_cost) if current_cost > 0 else 0.0

    print(f"  Current ordering cost : {current_cost:.4f} h")
    print(f"  Optimal ordering cost : {optimal_cost:.4f} h")
    print(f"  Efficiency gap        : {gap:.4f} h ({gap_ratio:.1%})")

    evidence = KnockOutEvidence(
        per_activity=current_order,
        current_ordering=current_names,
        optimal_ordering=optimal_names,
        current_ordering_efficiency=round(current_cost, 4),
        optimal_ordering_efficiency=round(optimal_cost, 4),
        efficiency_gap=round(gap, 4),
        efficiency_gap_ratio=round(gap_ratio, 4),
    )

    candidate = KnockOutCandidate(
        candidate_id="KO-001",
        heuristic="Knock-out",
        target_activities=current_names,
        heuristic_context=HEURISTIC_CONTEXT,
        process_fragment=None,
        evidence=evidence,
        additional_context=_build_additional_context(evidence, store),
    )

    return [candidate]

def print_candidates(candidates: list[KnockOutCandidate]) -> None:
    print(f"\n{'='*70}")
    print(f"KNOCK-OUT CANDIDATES  ({len(candidates)} found)")
    print(f"{'='*70}")

    for c in candidates:
        print(f"\n[{c.candidate_id}]")
        print(f"  {'─'*60}")

        e = c.evidence
        print(f"  current_ordering_efficiency  : {e.current_ordering_efficiency:.4f} h")
        print(f"  optimal_ordering_efficiency  : {e.optimal_ordering_efficiency:.4f} h")
        print(f"  efficiency_gap               : {e.efficiency_gap:.4f} h")
        print(f"  efficiency_gap_ratio          : {e.efficiency_gap_ratio:.1%}")

        print(f"\n  Current ordering: {' → '.join(e.current_ordering)}")
        print(f"  Optimal ordering: {' → '.join(e.optimal_ordering)}")

        print(f"\n  Per-activity statistics ({len(e.per_activity)} checks):")
        print(f"    {'Activity':<35} {'Term%':>6} {'Dur(h)':>8} "
              f"{'Eff':>8} {'Pos':>6} {'CumEff(h)':>10}")
        print(f"    {'─'*35} {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*10}")
        for s in e.per_activity:
            print(f"    {s.activity:<35} {s.termination_rate:>5.1%} "
                  f"{s.activity_duration_mean:>8.4f} "
                  f"{s.knockout_efficiency_ratio:>8.2f} "
                  f"{s.activity_position_ratio:>6.2f} "
                  f"{s.cumulative_effort_before_mean:>10.3f}")

        print(f"\n  Additional context:")
        words = c.additional_context.split()
        line: list[str] = []
        lines: list[str] = []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 72:
                lines.append("    " + " ".join(line[:-1]))
                line = [w]
        if line:
            lines.append("    " + " ".join(line))
        print("\n".join(lines))

    print(f"\n{'='*70}\n")

def _stats_to_dict(s: ActivityKnockoutStats) -> dict:
    return {
        "activity":                     s.activity,
        "termination_rate":             s.termination_rate,
        "activity_duration_mean":       s.activity_duration_mean,
        "knockout_efficiency_ratio":    s.knockout_efficiency_ratio,
        "activity_position_ratio":      s.activity_position_ratio,
        "cumulative_effort_before_mean": s.cumulative_effort_before_mean,
    }

def _candidate_to_dict(c: KnockOutCandidate) -> dict:
    """Convert one candidate to a plain dict for json.dump()."""
    return {
        "candidate_id": c.candidate_id,
        "heuristic": c.heuristic,
        "target_activities": c.target_activities,
        "heuristic_context": c.heuristic_context,
        "process_fragment": None,
        "evidence": {
            "per_activity": [
                _stats_to_dict(s) for s in c.evidence.per_activity
            ],
            "current_ordering":            c.evidence.current_ordering,
            "optimal_ordering":            c.evidence.optimal_ordering,
            "current_ordering_efficiency":  c.evidence.current_ordering_efficiency,
            "optimal_ordering_efficiency":  c.evidence.optimal_ordering_efficiency,
            "efficiency_gap":              c.evidence.efficiency_gap,
            "efficiency_gap_ratio":        c.evidence.efficiency_gap_ratio,
        },
        "additional_context": c.additional_context,
    }

def save_candidates_json(
    candidates: list[KnockOutCandidate],
    output_path: str,
) -> None:
    """Serialise the full candidate list to a JSON file."""

    import json

    payload = {
        "heuristic":        "Knock-out",
        "total_candidates": len(candidates),
        "candidates":       [_candidate_to_dict(c) for c in candidates],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(candidates)} candidate(s) → {output_path}")
