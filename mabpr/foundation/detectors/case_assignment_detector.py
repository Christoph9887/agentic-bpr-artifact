"""case_assignment_detector.py"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from .feature_extraction import FeatureStore

HEURISTIC_CONTEXT: str = (
    "Target problem types: excessive handovers, repeated re-familiarization "
    "with the same case, fragmented responsibility across activities. "
    "Applicable structural situations: processes where multiple activities "
    "of the same case are performed by different resources although they "
    "could potentially be handled by the same person. "
    "How the data is used: A high number of handovers within a case, "
    "frequent switching of resources between consecutive activities, and "
    "increased waiting times after handovers indicate that assigning more "
    "steps of the same case to the same resource may improve continuity "
    "and efficiency. "
    "Transformation style: redesign resource allocation so that one resource "
    "handles as many activities of the same case as possible, especially "
    "for consecutive or strongly related tasks. "
    "Feasibility: Maybe. Excessive handovers can be detected from the "
    "event log, but assigning multiple activities to the same resource "
    "depends on role definitions, skills, and authorization constraints."
)

@dataclass
class CaseAssignmentEvidence:
    handover_count_mean: float

    resource_continuity_rate: float

    waiting_time_after_handover_mean: float
    resource_return_rate: float

    cross_activity_resource_overlap: float

    cross_trained_resource_count: int

    resource_profile: dict

    handover_pairs: list

@dataclass
class CaseAssignmentCandidate:
    candidate_id: str
    heuristic: str
    target_activities: list[str]
    heuristic_context: str
    evidence: CaseAssignmentEvidence
    additional_context: str

HANDOVER_RATE_MIN = 0.5

PAIR_FREQUENCY_MIN = 30

GROUP_SIZE_MIN = 2

MAX_GROUP_SIZE = 5

def _percentile(values: list[float], p: int) -> float:
    """Return the p-th percentile of a list of floats (0 ≤ p ≤ 100)."""
    if not values:
        return 0.0
    sv  = sorted(values)
    idx = (p / 100) * (len(sv) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(sv) - 1)
    return sv[lo] * (1 - (idx - lo)) + sv[hi] * (idx - lo)

@dataclass
class _PairStats:
    """Internal: handover statistics for one consecutive activity pair."""
    total: int = 0
    handovers: int = 0
    wait_sum_h: float = 0.0

    @property
    def handover_rate(self) -> float:
        return self.handovers / self.total if self.total > 0 else 0.0

    @property
    def mean_wait_after_handover(self) -> float:
        return self.wait_sum_h / self.handovers if self.handovers > 0 else 0.0

def _compute_pair_handover_stats(
    df: pd.DataFrame,
) -> dict[tuple[str, str], _PairStats]:
    """For every consecutive activity pair (A, B) observed in the log, compute"""

    if "org:resource" not in df.columns:
        return {}

    df_sorted = df.sort_values(
        ["case:concept:name", "time:timestamp"]
    ).copy()

    df_sorted["_prev_act"] = df_sorted.groupby("case:concept:name")[
        "concept:name"
    ].shift(1)
    df_sorted["_prev_res"] = df_sorted.groupby("case:concept:name")[
        "org:resource"
    ].shift(1)
    df_sorted["_prev_complete"] = df_sorted.groupby("case:concept:name")[
        "time:timestamp"
    ].shift(1)

    mask = df_sorted["_prev_act"].notna()
    df_pairs = df_sorted[mask].copy()

    df_pairs["_is_handover"] = df_pairs["org:resource"] != df_pairs["_prev_res"]
    df_pairs["_wait_h"] = (
        df_pairs["start_timestamp"] - df_pairs["_prev_complete"]
    ).dt.total_seconds().clip(lower=0) / 3600.0

    stats: dict[tuple[str, str], _PairStats] = defaultdict(_PairStats)

    for _, row in df_pairs[
        ["_prev_act", "concept:name", "_is_handover", "_wait_h"]
    ].iterrows():
        pair = (row["_prev_act"], row["concept:name"])
        ps = stats[pair]
        ps.total += 1
        if row["_is_handover"]:
            ps.handovers += 1
            ps.wait_sum_h += row["_wait_h"]

    return dict(stats)

def _find_connected_components(
    edges: list[tuple[str, str]],
) -> list[set[str]]:
    """Return connected components from a list of undirected edges."""
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    visited: set[str] = set()
    components: list[set[str]] = []

    for node in adj:
        if node in visited:
            continue
        component: set[str] = set()
        frontier = [node]
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for nb in adj[current]:
                if nb not in visited:
                    frontier.append(nb)
        components.append(component)

    return components

def _merge_into_groups(
    high_handover_pairs: list[tuple[str, str]],
    pair_stats: dict[tuple[str, str], _PairStats],
) -> list[list[str]]:
    """Treat high-handover pairs as undirected edges and find connected"""

    if not high_handover_pairs:
        return []

    edges_with_rate: list[tuple[float, str, str]] = []
    for a, b in high_handover_pairs:
        rate = pair_stats[(a, b)].handover_rate if (a, b) in pair_stats else 0.5
        edges_with_rate.append((rate, a, b))
    edges_with_rate.sort(key=lambda x: x[0])

    active_edges: list[tuple[str, str]] = [(a, b) for _, a, b in edges_with_rate]
    components = _find_connected_components(active_edges)

    while any(len(c) > MAX_GROUP_SIZE for c in components) and active_edges:

        oversized_nodes: set[str] = set()
        for c in components:
            if len(c) > MAX_GROUP_SIZE:
                oversized_nodes |= c

        weakest = None
        for i, (rate, a, b) in enumerate(edges_with_rate):
            in_active = (a, b) in active_edges or (b, a) in active_edges
            in_oversized = a in oversized_nodes and b in oversized_nodes
            if in_active and in_oversized:
                weakest = i
                break

        if weakest is None:
            break

        _, wa, wb = edges_with_rate.pop(weakest)

        if (wa, wb) in active_edges:
            active_edges.remove((wa, wb))
        elif (wb, wa) in active_edges:
            active_edges.remove((wb, wa))

        if active_edges:
            components = _find_connected_components(active_edges)
        else:
            components = []
            break

    source_counts: dict[str, int] = defaultdict(int)
    target_counts: dict[str, int] = defaultdict(int)
    for a, b in active_edges:
        source_counts[a] += 1
        target_counts[b] += 1

    def _sort_key(act: str) -> float:
        s = source_counts.get(act, 0)
        t = target_counts.get(act, 0)
        return -(s / (s + t)) if (s + t) > 0 else 0.0

    groups: list[list[str]] = []
    for comp in components:
        ordered = sorted(comp, key=_sort_key)
        groups.append(ordered)

    groups.sort(key=lambda g: len(g), reverse=True)

    return groups

MAX_RESOURCES_PER_ACTIVITY = 5

def _compute_resource_profile(
    group: list[str],
    df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """For each activity in the group, compute the fraction of executions"""

    if "org:resource" not in df.columns:
        return {act: {} for act in group}

    group_set = set(group)
    act_df = df[df["concept:name"].isin(group_set)].copy()

    profile: dict[str, dict[str, float]] = {}
    for act in group:
        sub = act_df[act_df["concept:name"] == act]
        total = len(sub)
        if total == 0:
            profile[act] = {}
            continue

        counts = sub["org:resource"].value_counts()
        top = counts.head(MAX_RESOURCES_PER_ACTIVITY)
        result: dict[str, float] = {
            str(res): round(cnt / total, 4) for res, cnt in top.items()
        }
        remainder = total - top.sum()
        if remainder > 0:
            result["other"] = round(remainder / total, 4)
        profile[act] = result

    return profile

def _compute_resource_overlap(
    group: list[str],
    df: pd.DataFrame,
) -> tuple[float, int]:
    """Compute two objective metrics from the full resource data (not top-N):"""

    if "org:resource" not in df.columns:
        return 0.0, 0

    group_set = set(group)
    act_df = df[df["concept:name"].isin(group_set)]

    resource_sets: dict[str, set[str]] = {}
    for act in group:
        sub = act_df[act_df["concept:name"] == act]
        resources = set(sub["org:resource"].dropna().astype(str).unique())
        resource_sets[act] = resources

    activities = [a for a in group if resource_sets.get(a)]
    if len(activities) < 2:
        return 0.0, 0

    jaccard_sum = 0.0
    pair_count = 0
    for i in range(len(activities)):
        for j in range(i + 1, len(activities)):
            s_a = resource_sets[activities[i]]
            s_b = resource_sets[activities[j]]
            union = s_a | s_b
            inter = s_a & s_b
            if union:
                jaccard_sum += len(inter) / len(union)
            pair_count += 1

    overlap = jaccard_sum / pair_count if pair_count > 0 else 0.0

    resource_to_activities: dict[str, set[str]] = defaultdict(set)
    for act, resources in resource_sets.items():
        for res in resources:
            resource_to_activities[res].add(act)

    cross_trained = sum(
        1 for res, acts in resource_to_activities.items() if len(acts) >= 2
    )

    return round(overlap, 4), cross_trained

def _compute_group_evidence(
    group: list[str],
    pair_stats: dict[tuple[str, str], _PairStats],
    df: pd.DataFrame,
    store: FeatureStore,
) -> CaseAssignmentEvidence:
    """Compute the four evidence fields for a group of activities."""

    group_set = set(group)

    total_transitions = 0
    total_handovers   = 0
    total_wait_sum    = 0.0
    total_wait_count  = 0

    for (a, b), ps in pair_stats.items():
        if a in group_set and b in group_set:
            total_transitions += ps.total
            total_handovers   += ps.handovers
            total_wait_sum    += ps.wait_sum_h
            total_wait_count  += ps.handovers

    continuity = (
        1.0 - (total_handovers / total_transitions)
        if total_transitions > 0 else 1.0
    )
    wait_after = (
        total_wait_sum / total_wait_count
        if total_wait_count > 0 else 0.0
    )

    if "org:resource" not in df.columns:
        ho_mean = 0.0
        rep_rate = 0.0
    else:
        df_sorted = df.sort_values(
            ["case:concept:name", "time:timestamp"]
        )

        case_handovers: dict[str, int] = defaultdict(int)
        case_group_count: dict[str, int] = defaultdict(int)
        case_resource_positions: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

        prev_case = None
        prev_act  = None
        prev_res  = None
        pos_in_group = 0

        for _, row in df_sorted[
            ["case:concept:name", "concept:name", "org:resource"]
        ].iterrows():
            case_id = str(row["case:concept:name"])
            act     = row["concept:name"]
            res     = row["org:resource"]

            if case_id != prev_case:
                prev_case = case_id
                prev_act  = None
                prev_res  = None
                pos_in_group = 0

            if act in group_set:
                case_group_count[case_id] += 1
                if pd.notna(res):
                    case_resource_positions[case_id][res].append(pos_in_group)
                pos_in_group += 1

                if prev_act in group_set and prev_res is not None and pd.notna(res):
                    if res != prev_res:
                        case_handovers[case_id] += 1

            prev_act = act
            prev_res = res if pd.notna(res) else prev_res

        relevant_cases = [
            cid for cid, cnt in case_group_count.items() if cnt >= 2
        ]
        if relevant_cases:
            ho_mean = sum(
                case_handovers.get(cid, 0) for cid in relevant_cases
            ) / len(relevant_cases)
        else:
            ho_mean = 0.0

        rep_count = 0
        for cid in relevant_cases:
            res_pos = case_resource_positions.get(cid, {})
            for res, positions in res_pos.items():
                if len(positions) >= 2:

                    for i in range(len(positions) - 1):
                        if positions[i + 1] - positions[i] > 1:
                            rep_count += 1
                            break
                    else:
                        continue
                    break
        rep_rate = rep_count / len(relevant_cases) if relevant_cases else 0.0

    overlap, cross_trained = _compute_resource_overlap(group, df)

    handover_pairs = []
    for (a, b), ps in pair_stats.items():
        if a in group_set and b in group_set and ps.total >= PAIR_FREQUENCY_MIN:
            handover_pairs.append({
                "source": a,
                "target": b,
                "handover_rate": round(ps.handover_rate, 4),
                "consecutive_count": ps.total,
                "waiting_time_after_handover": round(ps.mean_wait_after_handover, 3),
            })
    handover_pairs.sort(key=lambda x: x["handover_rate"], reverse=True)

    return CaseAssignmentEvidence(
        handover_count_mean=round(ho_mean, 3),
        resource_continuity_rate=round(continuity, 4),
        waiting_time_after_handover_mean=round(wait_after, 3),
        resource_return_rate=round(rep_rate, 4),
        cross_activity_resource_overlap=overlap,
        cross_trained_resource_count=cross_trained,
        resource_profile=_compute_resource_profile(group, df),
        handover_pairs=handover_pairs,
    )

def _build_additional_context(
    group: list[str],
    evidence: CaseAssignmentEvidence,
    pair_stats: dict[tuple[str, str], _PairStats],
    store: FeatureStore,
) -> str:
    """Produce a factual supplementary note from event-log observations."""

    parts: list[str] = []
    group_set = set(group)

    worst_pairs: list[tuple[str, str, float]] = []
    for (a, b), ps in pair_stats.items():
        if a in group_set and b in group_set and ps.total >= PAIR_FREQUENCY_MIN:
            worst_pairs.append((a, b, ps.handover_rate))
    worst_pairs.sort(key=lambda x: x[2], reverse=True)
    if worst_pairs:
        top = worst_pairs[0]
        ps_top = pair_stats[(top[0], top[1])]
        parts.append(
            f"The highest handover rate within this group is "
            f"'{top[0]}' → '{top[1]}': {top[2]:.0%} across "
            f"{ps_top.total} consecutive occurrences "
            f"(mean waiting time after handover: "
            f"{ps_top.mean_wait_after_handover:.1f} hours)."
        )

    ct = evidence.cross_trained_resource_count
    overlap = evidence.cross_activity_resource_overlap
    if ct > 0:
        parts.append(
            f"In the event log, {ct} distinct resource(s) have executed "
            f"at least 2 different target activities (across all cases). "
            f"The pairwise Jaccard overlap of full resource sets across "
            f"target activities is {overlap:.2%}."
        )
    else:
        parts.append(
            f"No resource in the event log has executed more than one "
            f"of the target activities. The pairwise Jaccard overlap of "
            f"full resource sets is {overlap:.2%}."
        )

    rr = evidence.resource_return_rate
    if rr > 0.05:
        parts.append(
            f"In {rr:.0%} of relevant cases, a resource that participated "
            f"earlier in the group re-appears at a later position after "
            f"being replaced by a different resource in between."
        )

    parts.append(f"This group contains {len(group)} activities.")

    return " ".join(parts)

def detect_case_assignment(
    store: FeatureStore,
    df: pd.DataFrame,
) -> list[CaseAssignmentCandidate]:
    """Detect Case Assignment candidates from the event log."""

    pair_stats = _compute_pair_handover_stats(df)
    if not pair_stats:
        print("  No resource information available — cannot detect handovers.")
        return []

    print(f"  Total consecutive activity pairs observed: {len(pair_stats)}")

    high_pairs: list[tuple[str, str]] = []
    for (a, b), ps in pair_stats.items():
        if ps.total >= PAIR_FREQUENCY_MIN and ps.handover_rate >= HANDOVER_RATE_MIN:
            high_pairs.append((a, b))

    print(f"  High-handover pairs (rate >= {HANDOVER_RATE_MIN}, "
          f"freq >= {PAIR_FREQUENCY_MIN}): {len(high_pairs)}")

    if not high_pairs:
        print("  No high-handover pairs found.")
        return []

    groups = _merge_into_groups(high_pairs, pair_stats)

    groups = [g for g in groups if len(g) >= GROUP_SIZE_MIN]

    print(f"  Activity groups after merging: {len(groups)}")

    if not groups:
        return []

    raw: list[tuple[float, CaseAssignmentCandidate]] = []
    counter = 1

    for group in groups:
        evidence = _compute_group_evidence(group, pair_stats, df, store)

        candidate = CaseAssignmentCandidate(
            candidate_id=f"CA-{counter:03d}",
            heuristic="Case Assignment",
            target_activities=group,
            heuristic_context=HEURISTIC_CONTEXT,
            evidence=evidence,
            additional_context=_build_additional_context(
                group, evidence, pair_stats, store
            ),
        )
        counter += 1
        raw.append((-evidence.handover_count_mean, candidate))

    raw.sort(key=lambda x: x[0])
    return [c for _, c in raw]

def print_candidates(candidates: list[CaseAssignmentCandidate]) -> None:
    print(f"\n{'='*70}")
    print(f"CASE ASSIGNMENT CANDIDATES  ({len(candidates)} found)")
    print(f"{'='*70}")

    for c in candidates:
        acts = " → ".join(c.target_activities)
        print(f"\n[{c.candidate_id}]  {acts}")
        print(f"  {'─'*60}")

        e = c.evidence
        print(f"  handover_count_mean              : {e.handover_count_mean:.2f}")
        print(f"  resource_continuity_rate          : {e.resource_continuity_rate:.4f}"
              f"  ({e.resource_continuity_rate:.1%})")
        print(f"  waiting_time_after_handover_mean  : {e.waiting_time_after_handover_mean:.2f} h")
        print(f"  resource_return_rate              : {e.resource_return_rate:.4f}"
              f"  ({e.resource_return_rate:.1%})")
        print(f"  cross_activity_resource_overlap   : {e.cross_activity_resource_overlap:.4f}"
              f"  ({e.cross_activity_resource_overlap:.1%})")
        print(f"  cross_trained_resource_count      : {e.cross_trained_resource_count}")

        print(f"\n  Target activities ({len(c.target_activities)}):")
        for act in c.target_activities:
            print(f"    - {act}")

        rp = e.resource_profile
        if rp:
            print(f"\n  Resource profile:")
            for act in c.target_activities:
                res_map = rp.get(act, {})
                if res_map:
                    res_str = ", ".join(
                        f"{r} ({frac:.0%})" for r, frac in res_map.items()
                    )
                    print(f"    {act}: {res_str}")
                else:
                    print(f"    {act}: (no resource data)")

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

def _candidate_to_dict(c: CaseAssignmentCandidate) -> dict:
    """Convert one candidate to a plain dict for json.dump()."""
    return {
        "candidate_id": c.candidate_id,
        "heuristic": c.heuristic,
        "target_activities": c.target_activities,
        "heuristic_context": c.heuristic_context,
        "evidence": {
            "handover_count_mean":              c.evidence.handover_count_mean,
            "resource_continuity_rate":          c.evidence.resource_continuity_rate,
            "waiting_time_after_handover_mean":  c.evidence.waiting_time_after_handover_mean,
            "resource_return_rate":              c.evidence.resource_return_rate,
            "cross_activity_resource_overlap":   c.evidence.cross_activity_resource_overlap,
            "cross_trained_resource_count":      c.evidence.cross_trained_resource_count,
            "resource_profile":                 c.evidence.resource_profile,
            "handover_pairs":                   c.evidence.handover_pairs,
        },
        "additional_context": c.additional_context,
    }

def save_candidates_json(
    candidates: list[CaseAssignmentCandidate],
    output_path: str,
) -> None:
    """Serialise the full candidate list to a JSON file."""

    import json

    payload = {
        "heuristic":        "Case Assignment",
        "total_candidates": len(candidates),
        "candidates":       [_candidate_to_dict(c) for c in candidates],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(candidates)} candidate(s) → {output_path}")
