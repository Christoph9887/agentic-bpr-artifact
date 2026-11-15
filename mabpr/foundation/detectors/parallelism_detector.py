"""parallelism_detector.py"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from .feature_extraction import FeatureStore

HEURISTIC_CONTEXT: str = (
    "Target problem types: unnecessarily long throughput times caused by "
    "sequential execution of activities that are not strictly dependent. "
    "Applicable structural situations: activities currently executed "
    "sequentially although they have no strong causal dependency and could "
    "potentially be performed independently. "
    "How the data is used: Weak or absent causal dependency between "
    "activities together with consistent sequential ordering and significant "
    "waiting times indicates that the activities may be executed in parallel "
    "without affecting correctness. "
    "Transformation style: replace sequential execution with parallel "
    "branches, typically introducing an AND-split and corresponding "
    "synchronization. "
    "Feasibility: Yes. Activity dependencies and ordering relations can be "
    "derived from event logs to identify independent tasks, which can then "
    "be executed in parallel in the redesigned process."
)

@dataclass
class ProcessNode:
    id: str
    name: str
    type: str
    is_target: bool

@dataclass
class ProcessEdge:
    source: str
    target: str

@dataclass
class ProcessFragment:
    nodes: list[ProcessNode] = field(default_factory=list)
    edges: list[ProcessEdge] = field(default_factory=list)

@dataclass
class ParallelismEvidence:
    activity_a_duration: float
    activity_b_duration: float
    parallel_saving_potential: float

    waiting_time_between: float
    co_occurrence_rate: float

    resource_conflict_ratio: float

    dependency_strength_a_to_b: float
    dependency_strength_b_to_a: float

@dataclass
class ParallelismCandidate:
    candidate_id: str
    heuristic: str
    target_activities: dict
    heuristic_context: str
    process_fragment: ProcessFragment
    evidence: ParallelismEvidence
    additional_context: str

DEPENDENCY_STRONG_UPPER = 0.8

DURATION_OVERLAP_PERCENTILE = 50

WAITING_TIME_PERCENTILE = 50

DURATION_EPSILON = 0.01

CO_OCCURRENCE_MIN = 0.5

RESOURCE_CONFLICT_MAX = 0.85

def _find_sequential_activity_pairs(store: FeatureStore) -> list[tuple[str, str]]:
    """Return all (activity_a, activity_b) pairs where A and B are directly"""

    G    = store.process_graph.graph
    meta = store.process_graph.node_metadata
    a2n  = store.process_graph.activity_to_node

    if G is None or not a2n:
        return []

    n2a: dict[str, str] = {nid: name for name, nid in a2n.items()}

    pairs: list[tuple[str, str]] = []

    for src, tgt in G.edges():
        src_type = meta.get(src, {}).get("type", "")
        tgt_type = meta.get(tgt, {}).get("type", "")
        if src_type == "activity" and tgt_type == "activity":
            src_name = n2a.get(src)
            tgt_name = n2a.get(tgt)
            if src_name and tgt_name:
                pairs.append((src_name, tgt_name))

    return pairs

def _compute_resource_conflict_ratios(
    pairs: list[tuple[str, str]],
    df: pd.DataFrame,
) -> dict[tuple[str, str], float]:
    """For each (A, B) pair, compute the fraction of cases containing both"""

    if "org:resource" not in df.columns:
        return {pair: 0.0 for pair in pairs}

    case_act_resources: dict[str, dict[str, set]] = defaultdict(
        lambda: defaultdict(set)
    )
    for _, row in df[["case:concept:name", "concept:name", "org:resource"]].iterrows():
        case_id = str(row["case:concept:name"])
        act     = row["concept:name"]
        res     = row["org:resource"]
        if pd.notna(res):
            case_act_resources[case_id][act].add(res)

    result: dict[tuple[str, str], float] = {}
    for a, b in pairs:
        co_occur   = 0
        same_res   = 0
        for case_id, act_map in case_act_resources.items():
            if a in act_map and b in act_map:
                co_occur += 1

                if act_map[a] & act_map[b]:
                    same_res += 1
        result[(a, b)] = same_res / co_occur if co_occur > 0 else 0.0

    return result

def _compute_co_occurrence_rates(
    pairs: list[tuple[str, str]],
    df: pd.DataFrame,
    total_cases: int,
) -> dict[tuple[str, str], float]:
    """For each (A, B) pair, compute the fraction of ALL cases where both"""

    if total_cases == 0:
        return {pair: 0.0 for pair in pairs}

    case_activities: dict[str, set] = defaultdict(set)
    for _, row in df[["case:concept:name", "concept:name"]].iterrows():
        case_activities[str(row["case:concept:name"])].add(row["concept:name"])

    result: dict[tuple[str, str], float] = {}
    for a, b in pairs:
        co_count = sum(
            1 for acts in case_activities.values()
            if a in acts and b in acts
        )
        result[(a, b)] = co_count / total_cases

    return result

def _percentile(values: list[float], p: int) -> float:
    """Return the p-th percentile of a list of floats (0 ≤ p ≤ 100)."""
    if not values:
        return 0.0
    sv  = sorted(values)
    idx = (p / 100) * (len(sv) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(sv) - 1)
    return sv[lo] * (1 - (idx - lo)) + sv[hi] * (idx - lo)

def _bfs_fragment_nodes(
    G,
    start_id: str,
    direction: str,
    meta: dict,
    exclude: set,
    max_depth: int = 4,
) -> set:
    """BFS from start_id through the BPMN graph."""

    visited: set = set()
    frontier: set = {start_id}
    depth = 0
    while frontier and depth < max_depth:
        next_frontier: set = set()
        for node_id in frontier:
            neighbors = (
                list(G.predecessors(node_id))
                if direction == "backward"
                else list(G.successors(node_id))
            )
            for nb in neighbors:
                if nb == start_id or nb in visited or nb in exclude:
                    continue
                visited.add(nb)
                nb_type = meta.get(nb, {}).get("type", "unknown")
                if nb_type in (
                    "xor_gateway", "and_gateway", "or_gateway", "event_gateway"
                ):
                    next_frontier.add(nb)
        frontier = next_frontier
        depth += 1
    return visited

def _reachable_forward(G, start_id: str, max_hops: int = 12) -> set:
    """Return all node IDs reachable from start_id following out-edges,"""

    visited: set = set()
    frontier: set = {start_id}
    for _ in range(max_hops):
        next_frontier: set = set()
        for n in frontier:
            for nb in G.successors(n):
                if nb not in visited and nb != start_id:
                    visited.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier
    return visited

def _build_process_fragment(
    activity_a: str,
    activity_b: str,
    store: FeatureStore,
) -> ProcessFragment:
    """Extract a local process fragment centred on BOTH target activities."""

    G    = store.process_graph.graph
    a2n  = store.process_graph.activity_to_node
    meta = store.process_graph.node_metadata

    id_a = a2n.get(activity_a) if (G is not None and a2n) else None
    id_b = a2n.get(activity_b) if (G is not None and a2n) else None

    if G is None or id_a is None or id_b is None:
        raise ValueError(
            f"BPMN graph is required but one or both activities "
            f"({activity_a!r}, {activity_b!r}) were not found in "
            "process_graph.activity_to_node."
        )

    target_ids = {id_a, id_b}

    fwd_reachable_a = _reachable_forward(G, id_a)

    backward_ids = _bfs_fragment_nodes(
        G, id_a, "backward", meta, exclude=fwd_reachable_a
    )

    forward_ids = _bfs_fragment_nodes(
        G, id_b, "forward", meta, exclude=set()
    )

    between_ids = _bfs_fragment_nodes(
        G, id_a, "forward", meta, exclude=set()
    )

    all_bpmn_ids: set = target_ids | backward_ids | forward_ids | between_ids

    fragment_bpmn_edges = [
        (src, tgt)
        for src, tgt in G.edges()
        if src in all_bpmn_ids and tgt in all_bpmn_ids
    ]

    act_ids = sorted(
        [nid for nid in all_bpmn_ids
         if meta.get(nid, {}).get("type") in ("activity", "event")],
        key=lambda nid: (
            0 if nid == id_a else (1 if nid == id_b else 2),
            meta.get(nid, {}).get("name", ""),
        )
    )
    gw_ids = sorted(
        [nid for nid in all_bpmn_ids
         if meta.get(nid, {}).get("type") not in ("activity", "event")],
        key=lambda nid: meta.get(nid, {}).get("name", "")
    )

    local_id_map: dict[str, str] = {}
    for i, nid in enumerate(act_ids):
        local_id_map[nid] = f"n{i + 1}"
    for i, nid in enumerate(gw_ids):
        local_id_map[nid] = f"gw{i + 1}"

    nodes: list[ProcessNode] = []
    for nid in act_ids + gw_ids:
        m = meta.get(nid, {})
        nodes.append(ProcessNode(
            id=local_id_map[nid],
            name=m.get("name", ""),
            type=m.get("type", "unknown"),
            is_target=(nid in target_ids),
        ))

    edges: list[ProcessEdge] = [
        ProcessEdge(
            source=local_id_map[src],
            target=local_id_map[tgt],
        )
        for src, tgt in fragment_bpmn_edges
        if src in local_id_map and tgt in local_id_map
    ]

    return ProcessFragment(nodes=nodes, edges=edges)

def _build_additional_context(
    activity_a: str,
    activity_b: str,
    evidence: ParallelismEvidence,
    store: FeatureStore,
) -> str:
    """Produce a structured natural-language note listing supplementary signals"""

    parts: list[str] = []

    saving = evidence.parallel_saving_potential
    dur_a = evidence.activity_a_duration
    dur_b = evidence.activity_b_duration
    wt = evidence.waiting_time_between
    if saving > 0 and dur_a > DURATION_EPSILON and dur_b > DURATION_EPSILON:
        parts.append(
            f"Executing these activities in parallel could save up to "
            f"{saving:.2f} hours per case from execution overlap alone "
            f"(duration A: {dur_a:.2f} h, duration B: {dur_b:.2f} h)."
        )
    if wt > 1.0:
        parts.append(
            f"Additionally, the average waiting time between them is "
            f"{wt:.1f} hours — this sequential idle time could be reduced "
            "or eliminated through parallelisation."
        )

    rc = evidence.resource_conflict_ratio
    if rc > 0.5:
        parts.append(
            f"In {rc:.0%} of cases where both activities occur, they are "
            "performed by the same resource. Concurrent execution would "
            "require additional staffing or workload redistribution."
        )
    elif rc > 0.2:
        parts.append(
            f"In {rc:.0%} of co-occurring cases, the same resource performs "
            "both activities. Some resource reallocation may be needed."
        )
    else:
        parts.append(
            f"Resource conflict is low ({rc:.0%}): the two activities are "
            "typically performed by different resources, supporting the "
            "feasibility of concurrent execution."
        )

    dep_ab = evidence.dependency_strength_a_to_b
    dep_ba = evidence.dependency_strength_b_to_a
    stronger = max(dep_ab, dep_ba)
    if stronger > 0.8:
        parts.append(
            f"Note: one direction of the directly-follows relation is "
            f"dominant (max strength {stronger:.2f}), which may indicate a "
            "causal dependency between the activities. Domain knowledge is "
            "required to determine if this ordering is logically necessary "
            "or merely a historical convention."
        )
    elif stronger < 0.6:
        parts.append(
            "The directly-follows relation shows no dominant direction "
            f"(max strength {stronger:.2f}), suggesting the activities are "
            "likely independent."
        )

    co = evidence.co_occurrence_rate
    if co > 0.8:
        parts.append(
            f"Both activities co-occur in {co:.0%} of all cases, meaning "
            "parallelisation would impact the vast majority of process "
            "executions."
        )

    for act in (activity_a, activity_b):
        rework = store.control_flow.rework_counts.get(act, 0)
        if rework > 0:
            pct = rework / store.total_cases * 100 if store.total_cases > 0 else 0.0
            parts.append(
                f"Activity '{act}' exhibits rework in {rework} cases "
                f"({pct:.1f}% of all cases). Rework may indicate an implicit "
                "data dependency that is not visible in the control flow."
            )

    parts.append(
        "Whether these activities can be safely executed in parallel depends "
        "on synchronisation constraints, data dependencies, regulatory "
        "sequencing requirements, and resource availability — none of which "
        "can be fully determined from the event log alone."
    )

    return " ".join(parts)

def detect_parallelism(
    store: FeatureStore,
    df: pd.DataFrame,
) -> list[ParallelismCandidate]:
    """Scan the BPMN model for structurally sequential activity pairs and"""

    structural_pairs = _find_sequential_activity_pairs(store)
    if not structural_pairs:
        print("  No structurally sequential activity pairs found in BPMN model.")
        return []

    print(f"  Structural sequential pairs from BPMN: {len(structural_pairs)}")

    resource_conflicts = _compute_resource_conflict_ratios(structural_pairs, df)

    co_occurrence = _compute_co_occurrence_rates(
        structural_pairs, df, store.total_cases
    )

    all_pair_waits: list[float] = [
        wt for wt in store.time.waiting_time_between.values()
        if wt > 0
    ]
    waiting_threshold = _percentile(all_pair_waits, WAITING_TIME_PERCENTILE)

    all_durations: list[float] = [
        d for d in store.time.activity_duration_mean.values()
        if d >= DURATION_EPSILON
    ]
    duration_threshold = _percentile(all_durations, DURATION_OVERLAP_PERCENTILE)

    raw: list[tuple[float, ParallelismCandidate]] = []
    counter = 1

    for activity_a, activity_b in structural_pairs:

        dur_a = store.time.activity_duration_mean.get(activity_a, 0.0)
        dur_b = store.time.activity_duration_mean.get(activity_b, 0.0)
        overlap_saving = min(dur_a, dur_b)

        dep_ab = store.control_flow.dependency_strength.get(
            (activity_a, activity_b), 0.0
        )
        dep_ba = store.control_flow.dependency_strength.get(
            (activity_b, activity_a), 0.0
        )
        waiting = store.time.waiting_time_between.get(
            (activity_a, activity_b), 0.0
        )
        co_rate = co_occurrence.get((activity_a, activity_b), 0.0)
        rc_ratio = resource_conflicts.get((activity_a, activity_b), 0.0)

        stronger_dep = max(dep_ab, dep_ba)
        if stronger_dep >= DEPENDENCY_STRONG_UPPER:
            continue

        has_duration_benefit = overlap_saving > duration_threshold

        has_waiting_benefit = waiting > waiting_threshold
        if not has_duration_benefit and not has_waiting_benefit:
            continue

        if co_rate < CO_OCCURRENCE_MIN:
            continue

        if rc_ratio >= RESOURCE_CONFLICT_MAX:
            continue

        total_benefit = overlap_saving + waiting

        evidence = ParallelismEvidence(
            activity_a_duration=round(dur_a, 4),
            activity_b_duration=round(dur_b, 4),
            parallel_saving_potential=round(overlap_saving, 4),
            waiting_time_between=round(waiting, 3),
            co_occurrence_rate=round(co_rate, 4),
            resource_conflict_ratio=round(rc_ratio, 4),
            dependency_strength_a_to_b=round(dep_ab, 4),
            dependency_strength_b_to_a=round(dep_ba, 4),
        )

        candidate = ParallelismCandidate(
            candidate_id=f"PAR-{counter:03d}",
            heuristic="Parallelism",
            target_activities={
                "activity_a": activity_a,
                "activity_b": activity_b,
            },
            heuristic_context=HEURISTIC_CONTEXT,
            process_fragment=_build_process_fragment(
                activity_a, activity_b, store
            ),
            evidence=evidence,
            additional_context=_build_additional_context(
                activity_a, activity_b, evidence, store
            ),
        )
        counter += 1
        raw.append((-total_benefit, candidate))

    raw.sort(key=lambda x: x[0])
    return [c for _, c in raw]

def print_candidates(candidates: list[ParallelismCandidate]) -> None:
    print(f"\n{'='*70}")
    print(f"PARALLELISM CANDIDATES  ({len(candidates)} found)")
    print(f"{'='*70}")

    for c in candidates:
        a = c.target_activities["activity_a"]
        b = c.target_activities["activity_b"]
        print(f"\n[{c.candidate_id}]  {a}  →  {b}")
        print(f"  {'─'*60}")

        e = c.evidence
        print(f"  activity_a_duration       : {e.activity_a_duration:.4f} h"
              f"  ({e.activity_a_duration * 60:.1f} min)")
        print(f"  activity_b_duration       : {e.activity_b_duration:.4f} h"
              f"  ({e.activity_b_duration * 60:.1f} min)")
        print(f"  parallel_saving_potential : {e.parallel_saving_potential:.4f} h"
              f"  ({e.parallel_saving_potential * 60:.1f} min)")
        print(f"  waiting_time_between      : {e.waiting_time_between:.2f} h")
        print(f"  co_occurrence_rate        : {e.co_occurrence_rate:.4f}"
              f"  ({e.co_occurrence_rate:.1%})")
        print(f"  resource_conflict_ratio   : {e.resource_conflict_ratio:.4f}"
              f"  ({e.resource_conflict_ratio:.1%})")
        print(f"  dependency_strength (A→B) : {e.dependency_strength_a_to_b:.4f}"
              "  (informational)")
        print(f"  dependency_strength (B→A) : {e.dependency_strength_b_to_a:.4f}"
              "  (informational)")

        print(f"\n  Process fragment:")
        for node in c.process_fragment.nodes:
            marker = "  ◀ TARGET" if node.is_target else ""
            print(f"    [{node.id}] {node.name}  ({node.type}){marker}")
        for edge in c.process_fragment.edges:
            print(f"    {edge.source} → {edge.target}")

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

def _candidate_to_dict(c: ParallelismCandidate) -> dict:
    """Convert one candidate to a plain dict that matches the specified"""

    return {
        "candidate_id": c.candidate_id,
        "heuristic": c.heuristic,
        "target_activities": c.target_activities,
        "heuristic_context": c.heuristic_context,
        "process_fragment": {
            "nodes": [
                {
                    "id":        node.id,
                    "name":      node.name,
                    "type":      node.type,
                    "is_target": node.is_target,
                }
                for node in c.process_fragment.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                }
                for edge in c.process_fragment.edges
            ],
        },
        "evidence": {
            "activity_a_duration":         c.evidence.activity_a_duration,
            "activity_b_duration":         c.evidence.activity_b_duration,
            "parallel_saving_potential":   c.evidence.parallel_saving_potential,
            "waiting_time_between":        c.evidence.waiting_time_between,
            "co_occurrence_rate":          c.evidence.co_occurrence_rate,
            "resource_conflict_ratio":     c.evidence.resource_conflict_ratio,
            "dependency_strength_a_to_b":  c.evidence.dependency_strength_a_to_b,
            "dependency_strength_b_to_a":  c.evidence.dependency_strength_b_to_a,
        },
        "additional_context": c.additional_context,
    }

def save_candidates_json(
    candidates: list[ParallelismCandidate],
    output_path: str,
) -> None:
    """Serialise the full candidate list to a JSON file."""

    import json

    payload = {
        "heuristic":        "Parallelism",
        "total_candidates": len(candidates),
        "candidates":       [_candidate_to_dict(c) for c in candidates],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(candidates)} candidate(s) → {output_path}")
