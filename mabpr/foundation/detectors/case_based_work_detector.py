"""case_based_work_detector.py"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .feature_extraction import FeatureStore

HEURISTIC_CONTEXT: str = (
    "Target problem types: delays caused by batch processing or periodic "
    "handling of cases, accumulation of waiting cases before execution. "
    "Applicable structural situations: activities executed in batches or "
    "triggered periodically instead of being processed immediately when a "
    "case arrives. "
    "How the data is used: Repeated patterns where multiple cases start the "
    "same activity at nearly identical timestamps or where long waiting "
    "times precede activity execution indicate batch or periodic processing "
    "instead of case-by-case handling. "
    "Transformation style: redesign the process so that cases are processed "
    "individually upon arrival instead of waiting for batch or periodic "
    "execution. "
    "Feasibility: Yes. Batch patterns and periodic execution behavior can "
    "be detected from event log timestamps, and the process can be "
    "redesigned to enable case-by-case execution."
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
class CaseBasedWorkEvidence:
    waiting_time_before: float
    batch_size_mean: float
    batch_interval_mean: float
    periodicity_strength: float
    arrival_rate: float
    execution_rate: float
    arrival_execution_gap: float
    target_remapped_from: list[str] | None = None

@dataclass
class CaseBasedWorkCandidate:
    candidate_id: str
    heuristic: str
    target_activity: str
    heuristic_context: str
    process_fragment: ProcessFragment
    evidence: CaseBasedWorkEvidence
    additional_context: str

WAITING_TIME_PERCENTILE = 75

BATCH_SIZE_MIN = 5.0

def _percentile(values: list[float], p: int) -> float:
    """Return the p-th percentile of a list of floats (0 ≤ p ≤ 100)."""
    if not values:
        return 0.0
    sv  = sorted(values)
    idx = (p / 100) * (len(sv) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(sv) - 1)
    return sv[lo] * (1 - (idx - lo)) + sv[hi] * (idx - lo)

def _compute_arrival_execution_rates(
    activities: list[str],
    df: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """For each activity compute:"""

    obs_start = df["start_timestamp"].min()
    obs_end   = df["time:timestamp"].max()
    full_hours = (obs_end - obs_start).total_seconds() / 3600.0

    if "case:concept:name" in df.columns:
        last_arrival = df.groupby("case:concept:name")["start_timestamp"].min().max()
        op_hours = (last_arrival - obs_start).total_seconds() / 3600.0
        obs_hours = (op_hours if full_hours > 0 and op_hours / full_hours >= 0.10
                     else full_hours)
    else:
        obs_hours = full_hours

    if obs_hours <= 0:
        return {act: (0.0, 0.0) for act in activities}

    result: dict[str, tuple[float, float]] = {}

    for act in activities:
        act_df = df[df["concept:name"] == act]
        if act_df.empty:
            result[act] = (0.0, 0.0)
            continue

        n_arrivals = act_df["case:concept:name"].nunique()
        arrival_rate = n_arrivals / obs_hours

        n_executions = len(act_df)
        execution_rate = n_executions / obs_hours

        result[act] = (arrival_rate, execution_rate)

    return result

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
    evidence: CaseBasedWorkEvidence,
    store: FeatureStore,
    *,
    remapped_from: str | None = None,
) -> str:
    """Produce a structured natural-language note listing supplementary signals"""

    parts: list[str] = []

    if remapped_from is not None:
        parts.append(
            f"Target was remapped from {remapped_from!r} (a lifecycle "
            f"state-change event flagged as batched by pm4py because the "
            f"underlying work being batched leaves a clustered-timestamp "
            f"signature there) to {activity!r}, the work activity that "
            f"consistently precedes it within cases. The W_ activity is "
            f"the real redesign target — the quantitative evidence "
            f"(batch_size, gap, waiting_time) below remains attributed to "
            f"the original A_/O_ event, which is where the batching "
            f"signal is observable."
        )

    ps = evidence.periodicity_strength
    bi = evidence.batch_interval_mean
    if ps > 0.7 and bi > 0:
        parts.append(
            f"The activity shows strong periodicity (strength {ps:.2f}) "
            f"with an average interval of {bi:.1f} hours between batch "
            "executions. This pattern is consistent with scheduled or "
            "system-triggered batch processing (e.g. nightly batch runs, "
            "hourly cron jobs)."
        )
    elif ps > 0.3 and bi > 0:
        parts.append(
            f"The activity shows moderate periodicity (strength {ps:.2f}) "
            f"with an average interval of {bi:.1f} hours. The pattern may "
            "reflect informal batching habits rather than strict system "
            "scheduling."
        )

    gap = evidence.arrival_execution_gap
    if gap > 0:
        parts.append(
            f"The arrival rate ({evidence.arrival_rate:.3f} cases/h) exceeds "
            f"the execution rate ({evidence.execution_rate:.3f} cases/h), "
            f"indicating a backlog accumulation of approximately "
            f"{gap:.3f} cases/h. Switching to case-by-case processing "
            "would require increasing throughput capacity to match demand."
        )
    else:
        parts.append(
            "The execution rate meets or exceeds the arrival rate, "
            "suggesting that case-by-case processing is feasible without "
            "additional capacity."
        )

    bs = evidence.batch_size_mean
    if bs > 10:
        parts.append(
            f"The average batch size is {bs:.1f} cases, which is large. "
            "Transitioning to case-by-case processing would significantly "
            "change the workload pattern for this activity."
        )

    rework = store.control_flow.rework_counts.get(activity, 0)
    if rework > 0:
        pct = rework / store.total_cases * 100 if store.total_cases > 0 else 0.0
        parts.append(
            f"This activity exhibits rework in {rework} cases "
            f"({pct:.1f}% of all cases). Rework in a batched activity "
            "may indicate that batch errors propagate across multiple cases."
        )

    parts.append(
        "Whether batch execution can be replaced by case-by-case processing "
        "depends on system scheduling policies, operational constraints "
        "(e.g. external system availability windows), and whether downstream "
        "systems require batched input — none of which can be determined "
        "from the event log alone."
    )

    return " ".join(parts)

def _find_w_predecessor(
    df: pd.DataFrame,
    activity: str,
    dominance: float = 0.5,
) -> str | None:
    """If *activity* is consistently preceded by a single ``W_*`` work item"""

    case_col = "case:concept:name"
    act_col  = "concept:name"
    if case_col not in df.columns or act_col not in df.columns:
        return None

    sorted_df = df.sort_values([case_col, "start_timestamp"])

    counts: dict[str, int] = {}
    n_occurrences = 0
    for _, group in sorted_df.groupby(case_col, sort=False):
        seq = group[act_col].tolist()
        for i, ev in enumerate(seq):
            if ev != activity:
                continue
            n_occurrences += 1
            for j in range(i - 1, -1, -1):
                if seq[j].startswith("W_"):
                    counts[seq[j]] = counts.get(seq[j], 0) + 1
                    break

    if n_occurrences == 0 or not counts:
        return None

    best_w, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_count / n_occurrences >= dominance:
        return best_w
    return None

def _dedupe_by_target(
    raw: list[tuple[float, "CaseBasedWorkCandidate"]],
) -> list[tuple[float, "CaseBasedWorkCandidate"]]:
    """Consolidate candidates that share the same ``target_activity``"""

    by_target: dict[str, list[tuple[float, CaseBasedWorkCandidate]]] = {}
    for item in raw:
        by_target.setdefault(item[1].target_activity, []).append(item)

    merged: list[tuple[float, CaseBasedWorkCandidate]] = []
    for target, group in by_target.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        group.sort(key=lambda it: (it[1].evidence.arrival_execution_gap,
                                   -it[1].evidence.waiting_time_before))
        winner_key, winner = group[0]

        sources: list[str] = []
        seen: set[str] = set()
        for _, c in group:
            tr = c.evidence.target_remapped_from
            if tr:
                for src in tr:
                    if src not in seen:
                        seen.add(src)
                        sources.append(src)
        winner.evidence.target_remapped_from = sources if sources else None

        n_merged = len(group)
        src_str = (", ".join(repr(s) for s in sources)
                   if sources else "(none — direct W_ detection)")
        merge_note = (
            f"This candidate consolidates batching signals from {n_merged} "
            f"observations on different events that all causally trace back "
            f"to the W_ work activity {target!r}. Source A_/O_ events: "
            f"{src_str}. The evidence fields below are the strongest single "
            f"observation (most negative arrival_execution_gap). A single "
            f"CBW redesign on this W_ target captures the full merged "
            f"signal."
        )
        winner.additional_context = merge_note + "\n\n" + winner.additional_context

        merged.append((winner_key, winner))
    return merged

def detect_case_based_work(
    store: FeatureStore,
    df: pd.DataFrame,
) -> list[CaseBasedWorkCandidate]:
    """Scan activities detected as batched in the FeatureStore and return those"""

    batched_acts = list(store.batch.batched_activities.keys())
    if not batched_acts:
        print("  No batched activities detected in the event log.")
        return []

    print(f"  Batched activities from FeatureStore: {len(batched_acts)}")

    rates = _compute_arrival_execution_rates(batched_acts, df)

    all_waiting_times: list[float] = [
        wt for wt in store.time.waiting_time_mean.values()
        if wt > 0
    ]
    waiting_threshold = _percentile(all_waiting_times, WAITING_TIME_PERCENTILE)

    raw: list[tuple[float, CaseBasedWorkCandidate]] = []
    counter = 1

    for activity in batched_acts:

        waiting   = store.time.waiting_time_mean.get(activity, 0.0)
        bs_mean   = store.batch.batch_size_mean.get(activity, 0.0)
        bi_mean   = store.batch.batch_interval_mean.get(activity, 0.0)
        period    = store.batch.periodicity_strength.get(activity, 0.0)

        arr_rate, exec_rate = rates.get(activity, (0.0, 0.0))
        gap = arr_rate - exec_rate

        if waiting <= waiting_threshold:
            continue

        if bs_mean < BATCH_SIZE_MIN:
            continue

        if gap > 0.5:
            continue

        remapped_from: str | None = None
        target_activity = activity
        if activity.startswith(("A_", "O_")):
            w_pred = _find_w_predecessor(df, activity)
            a2n = store.process_graph.activity_to_node or {}
            if w_pred is not None and w_pred in a2n:
                remapped_from = activity
                target_activity = w_pred

        evidence = CaseBasedWorkEvidence(
            waiting_time_before=round(waiting, 3),
            batch_size_mean=round(bs_mean, 2),
            batch_interval_mean=round(bi_mean, 3),
            periodicity_strength=round(period, 4),
            arrival_rate=round(arr_rate, 6),
            execution_rate=round(exec_rate, 6),
            arrival_execution_gap=round(gap, 6),
            target_remapped_from=[remapped_from] if remapped_from else None,
        )

        candidate = CaseBasedWorkCandidate(
            candidate_id=f"CBW-{counter:03d}",
            heuristic="Case-based Work",
            target_activity=target_activity,
            heuristic_context=HEURISTIC_CONTEXT,
            process_fragment=_build_process_fragment(target_activity, store),
            evidence=evidence,
            additional_context=_build_additional_context(
                target_activity, evidence, store, remapped_from=remapped_from,
            ),
        )
        counter += 1
        raw.append((-waiting, candidate))

    raw = _dedupe_by_target(raw)

    raw.sort(key=lambda x: x[0])
    candidates = [c for _, c in raw]
    for i, c in enumerate(candidates, start=1):
        c.candidate_id = f"CBW-{i:03d}"
    return candidates

def print_candidates(candidates: list[CaseBasedWorkCandidate]) -> None:
    print(f"\n{'='*70}")
    print(f"CASE-BASED WORK CANDIDATES  ({len(candidates)} found)")
    print(f"{'='*70}")

    for c in candidates:
        print(f"\n[{c.candidate_id}]  {c.target_activity}")
        print(f"  {'─'*60}")

        e = c.evidence
        print(f"  waiting_time_before      : {e.waiting_time_before:.2f} h")
        print(f"  batch_size_mean          : {e.batch_size_mean:.2f} cases")
        print(f"  batch_interval_mean      : {e.batch_interval_mean:.2f} h")
        print(f"  periodicity_strength     : {e.periodicity_strength:.4f}"
              f"  ({e.periodicity_strength:.1%})")
        print(f"  arrival_rate             : {e.arrival_rate:.4f} cases/h")
        print(f"  execution_rate           : {e.execution_rate:.4f} cases/h")
        gap_str = f"{e.arrival_execution_gap:+.4f}"
        print(f"  arrival_execution_gap    : {gap_str} cases/h")

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

def _candidate_to_dict(c: CaseBasedWorkCandidate) -> dict:
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
            "waiting_time_before":      c.evidence.waiting_time_before,
            "batch_size_mean":          c.evidence.batch_size_mean,
            "batch_interval_mean":      c.evidence.batch_interval_mean,
            "periodicity_strength":     c.evidence.periodicity_strength,
            "arrival_rate":             c.evidence.arrival_rate,
            "execution_rate":           c.evidence.execution_rate,
            "arrival_execution_gap":    c.evidence.arrival_execution_gap,
            "target_remapped_from":     c.evidence.target_remapped_from,
        },
        "additional_context": c.additional_context,
    }

def save_candidates_json(
    candidates: list[CaseBasedWorkCandidate],
    output_path: str,
) -> None:
    """Serialise the full candidate list to a JSON file."""

    import json

    payload = {
        "heuristic":        "Case-based Work",
        "total_candidates": len(candidates),
        "candidates":       [_candidate_to_dict(c) for c in candidates],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(candidates)} candidate(s) → {output_path}")
