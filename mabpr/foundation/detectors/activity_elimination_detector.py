"""activity_elimination_detector.py"""

from __future__ import annotations

from dataclasses import dataclass, field

from .feature_extraction import FeatureStore

HEURISTIC_CONTEXT: str = (
    "Target problem types: redundant activities, non-value-adding tasks, "
    "excessive verification or control steps. "
    "Applicable structural situations: processes where certain activities "
    "do not influence the final outcome or mainly act as rechecks or "
    "redundant information transfers. "
    "How the data is used: Activities with weak causal dependency to "
    "subsequent tasks, low impact on case completion, or long waiting "
    "times relative to execution time indicate potential non-value-adding "
    "steps and become candidates for elimination. "
    "Transformation style: remove the redundant activity and reconnect "
    "its predecessor and successor tasks (e.g., A → B → C becomes A → C). "
    "Feasibility: Maybe. Potentially redundant activities can be detected "
    "from event logs, but deciding whether they can be safely removed "
    "requires additional compliance or policy information."
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
class EliminationEvidence:
    waiting_time_before: float
    activity_duration: float
    waiting_to_duration_ratio: float
    bypass_ratio: float
    activity_frequency: int
    variant_coverage: float
    predecessor_activities: list
    successor_activities: list

@dataclass
class ActivityEliminationCandidate:
    candidate_id: str
    heuristic: str
    target_activity: str
    heuristic_context: str
    process_fragment: ProcessFragment
    evidence: EliminationEvidence
    additional_context: str

DURATION_EPSILON = 0.01

WAIT_RATIO_PERCENTILE = 50

BYPASS_ABSOLUTE = 0.5

BYPASS_HIGH = 0.85

DURATION_PERCENTILE = 75

COVERAGE_GUARD = 0.95

def _percentile(values: list[float], p: int) -> float:
    """Return the p-th percentile of a list of floats (0 ≤ p ≤ 100)."""
    if not values:
        return 0.0
    sv  = sorted(values)
    idx = (p / 100) * (len(sv) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(sv) - 1)
    return sv[lo] * (1 - (idx - lo)) + sv[hi] * (idx - lo)

def _forward_successors(activity: str, store: FeatureStore) -> list[str]:
    """Return successors reached via FORWARD edges only (dependency_strength > 0.5)."""

    return [
        b for (a, b), strength in store.control_flow.dependency_strength.items()
        if a == activity and strength > 0.5
    ]

def _predecessors(activity: str, store: FeatureStore) -> list[str]:
    """Return all direct predecessors of `activity` from the DFG."""
    return [a for (a, b) in store.control_flow.dfg_frequency if b == activity]

def _activity_neighbors(
    activity: str,
    store: FeatureStore,
    direction: str,
) -> list[dict]:
    """Return the one-hop activity neighbours of `activity` as observed in the"""

    if direction == "predecessors":
        pairs = [
            (a, store.control_flow.dfg_frequency[(a, activity)],
             store.control_flow.dependency_strength.get((a, activity), 0.0))
            for (a, b) in store.control_flow.dfg_frequency
            if b == activity
        ]
    else:
        pairs = [
            (b, store.control_flow.dfg_frequency[(activity, b)],
             store.control_flow.dependency_strength.get((activity, b), 0.0))
            for (a, b), strength in store.control_flow.dependency_strength.items()
            if a == activity and strength > 0.5
        ]

    pairs.sort(key=lambda x: x[1], reverse=True)
    return [
        {
            "name":                name,
            "dfg_frequency":       freq,
            "dependency_strength": round(dep, 4),
        }
        for name, freq, dep in pairs
    ]

def _bypass_ratio(activity: str, store: FeatureStore) -> float:
    """Compute the bypass ratio for `activity`."""

    forward_succs = _forward_successors(activity, store)
    if not forward_succs:
        return 0.0

    ratios: list[float] = []
    for succ in forward_succs:
        succ_freq    = store.control_flow.activity_frequency.get(succ, 0)
        via_activity = store.control_flow.dfg_frequency.get((activity, succ), 0)
        if succ_freq == 0:
            continue
        bypassed = max(0, succ_freq - via_activity)
        ratios.append(bypassed / succ_freq)

    return max(ratios) if ratios else 0.0

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
    target: str,
    store: FeatureStore,
) -> ProcessFragment:
    """Extract a local process fragment centred on the target activity"""

    G    = store.process_graph.graph
    a2n  = store.process_graph.activity_to_node
    meta = store.process_graph.node_metadata

    target_id = a2n.get(target) if (G is not None and a2n) else None

    if G is None or target_id is None:
        raise ValueError(
            f"BPMN graph is required but activity {target!r} "
            "was not found in process_graph.activity_to_node."
        )

    fwd_reachable = _reachable_forward(G, target_id)

    backward_ids = _bfs_fragment_nodes(
        G, target_id, "backward", meta, exclude=fwd_reachable
    )

    forward_ids = _bfs_fragment_nodes(
        G, target_id, "forward", meta, exclude=set()
    )

    all_bpmn_ids: set = {target_id} | backward_ids | forward_ids

    fragment_bpmn_edges = [
        (src, tgt)
        for src, tgt in G.edges()
        if src in all_bpmn_ids and tgt in all_bpmn_ids
    ]

    act_ids = sorted(
        [nid for nid in all_bpmn_ids
         if meta.get(nid, {}).get("type") in ("activity", "event")],
        key=lambda nid: (0 if nid == target_id else 1,
                         meta.get(nid, {}).get("name", ""))
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
            is_target=(nid == target_id),
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
    activity: str,
    evidence: EliminationEvidence,
    store: FeatureStore,
) -> str:
    """Produce a structured natural-language note listing supplementary signals"""

    parts: list[str] = []

    rework = store.control_flow.rework_counts.get(activity, 0)
    if rework > 0:
        pct = rework / store.total_cases * 100 if store.total_cases > 0 else 0.0
        parts.append(
            f"This activity exhibits rework in {rework} cases "
            f"({pct:.1f}% of all cases). "
            "Rework may indicate a validation or error-correction role; "
            "removal could increase downstream error rates."
        )

    cov = evidence.variant_coverage
    if cov < 0.5:
        parts.append(
            f"The activity appears in only {cov:.0%} of cases, meaning the "
            "majority of cases already complete successfully without it. "
            "This low coverage reduces the structural risk of elimination."
        )
    elif cov > 0.8:
        parts.append(
            f"Despite appearing in {cov:.0%} of cases, the activity's high "
            f"bypass ratio ({evidence.bypass_ratio:.2f}) indicates its "
            "successors are frequently reached via alternative paths."
        )

    br = evidence.bypass_ratio
    if br > 0.8:
        parts.append(
            f"The bypass ratio is {br:.2f}: in over {br:.0%} of executions "
            "that reach the successor, this activity was not traversed, "
            "confirming it is structurally non-essential."
        )

    parts.append(
        "Whether this activity can be safely removed depends on compliance "
        "requirements, regulatory obligations, and organisational policy — "
        "none of which can be determined from the event log alone."
    )

    return " ".join(parts)

def detect_activity_elimination(
    store: FeatureStore,
) -> list[ActivityEliminationCandidate]:
    """Scan all activities in the FeatureStore and return those that qualify as"""

    bypass_ratios: dict[str, float] = {
        act: _bypass_ratio(act, store) for act in store.activities
    }

    meaningful_acts: list[str] = [
        act for act in store.activities
        if store.time.activity_duration_mean.get(act, 0.0) >= DURATION_EPSILON
    ]

    meaningful_ratios: list[float] = [
        v for act in meaningful_acts
        if (v := store.time.waiting_to_duration_ratio.get(act)) is not None and v >= 0
    ]
    meaningful_durations: list[float] = [
        store.time.activity_duration_mean[act] for act in meaningful_acts
    ]

    ratio_threshold = _percentile(meaningful_ratios,    WAIT_RATIO_PERCENTILE)
    dur_threshold   = _percentile(meaningful_durations, DURATION_PERCENTILE)

    start_acts: set[str] = {
        act for act in store.activities if not _predecessors(act, store)
    }
    end_acts: set[str] = {
        act for act in store.activities if not _forward_successors(act, store)
    }
    structural = start_acts | end_acts

    raw: list[tuple[float, float, ActivityEliminationCandidate]] = []
    counter = 1

    for activity in store.activities:

        if activity in structural:
            continue

        coverage = store.control_flow.activity_variant_coverage.get(activity, 1.0)
        if coverage >= COVERAGE_GUARD:
            continue

        duration  = store.time.activity_duration_mean.get(activity, 0.0)
        waiting   = store.time.waiting_time_mean.get(activity, 0.0)
        ratio_raw = store.time.waiting_to_duration_ratio.get(activity)
        ratio     = ratio_raw if (ratio_raw is not None and ratio_raw >= 0) else 0.0
        bypass    = bypass_ratios[activity]
        frequency = store.control_flow.activity_frequency.get(activity, 0)

        if duration < DURATION_EPSILON:
            continue

        if ratio <= ratio_threshold:
            continue

        if bypass <= BYPASS_ABSOLUTE:
            continue

        if duration >= dur_threshold and bypass <= BYPASS_HIGH:
            continue

        evidence = EliminationEvidence(
            waiting_time_before=round(waiting, 3),
            activity_duration=round(duration, 4),
            waiting_to_duration_ratio=(
                round(ratio, 1) if ratio_raw is not None else -1.0
            ),
            bypass_ratio=round(bypass, 4),
            activity_frequency=frequency,
            variant_coverage=round(coverage, 4),
            predecessor_activities=_activity_neighbors(activity, store, "predecessors"),
            successor_activities=_activity_neighbors(activity, store, "successors"),
        )

        candidate = ActivityEliminationCandidate(
            candidate_id=f"AE-{counter:03d}",
            heuristic="Activity Elimination",
            target_activity=activity,
            heuristic_context=HEURISTIC_CONTEXT,
            process_fragment=_build_process_fragment(activity, store),
            evidence=evidence,
            additional_context=_build_additional_context(activity, evidence, store),
        )
        counter += 1
        raw.append((-bypass, -ratio, candidate))

    raw.sort(key=lambda x: (x[0], x[1]))
    return [c for _, _, c in raw]

def print_candidates(candidates: list[ActivityEliminationCandidate]) -> None:
    print(f"\n{'='*70}")
    print(f"ACTIVITY ELIMINATION CANDIDATES  ({len(candidates)} found)")
    print(f"{'='*70}")

    for c in candidates:
        print(f"\n[{c.candidate_id}]  {c.target_activity}")
        print(f"  {'─'*60}")

        e = c.evidence
        print(f"  waiting_time_before      : {e.waiting_time_before:.2f} h")
        print(f"  activity_duration        : {e.activity_duration:.4f} h"
              f"  ({e.activity_duration * 60:.1f} min)")
        ratio_str = (
            f"{e.waiting_to_duration_ratio:.1f}x"
            if e.waiting_to_duration_ratio >= 0 else "N/A"
        )
        print(f"  waiting_to_duration_ratio: {ratio_str}")
        print(f"  bypass_ratio             : {e.bypass_ratio:.4f}"
              f"  ({e.bypass_ratio:.1%} of executions bypass this activity)")
        print(f"  activity_frequency       : {e.activity_frequency}")
        print(f"  variant_coverage         : {e.variant_coverage:.1%}")
        print(f"  predecessor_activities   : {e.predecessor_activities}")
        print(f"  successor_activities     : {e.successor_activities}")

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

def _candidate_to_dict(c: ActivityEliminationCandidate) -> dict:
    """Convert one candidate to a plain dict that matches the specified"""

    return {
        "candidate_id": c.candidate_id,
        "heuristic": c.heuristic,
        "target_activity": c.target_activity,
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
            "waiting_time_before":       c.evidence.waiting_time_before,
            "activity_duration":         c.evidence.activity_duration,
            "waiting_to_duration_ratio": c.evidence.waiting_to_duration_ratio,
            "bypass_ratio":              c.evidence.bypass_ratio,
            "activity_frequency":        c.evidence.activity_frequency,
            "variant_coverage":          c.evidence.variant_coverage,
            "predecessor_activities":    c.evidence.predecessor_activities,
            "successor_activities":      c.evidence.successor_activities,
        },
        "additional_context": c.additional_context,
    }

def save_candidates_json(
    candidates: list[ActivityEliminationCandidate],
    output_path: str,
) -> None:
    """Serialise the full candidate list to a JSON file."""

    import json

    payload = {
        "heuristic":        "Activity Elimination",
        "total_candidates": len(candidates),
        "candidates":       [_candidate_to_dict(c) for c in candidates],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(candidates)} candidate(s) → {output_path}")
