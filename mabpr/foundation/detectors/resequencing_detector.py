"""
resequencing_detector.py

Detects Resequencing candidates from a populated FeatureStore and event log
DataFrame.

Candidate sourcing:
  Activities are sourced from the event log — only activities that have both
  predecessors and successors in the DFG (non-boundary) are considered.
  The detector identifies activities positioned too early in the process
  sequence that could be postponed without affecting correctness.

Evidence signals (from event log):
  - High waiting time before the activity (blocked by upstream)
  - Weak causal dependency to predecessors and successors
  - Early position in the trace (low position_ratio)
  - High ordering variance across cases (position is unstable)
  - High position inconsistency rate (many variants place it later
    than the dominant variant does)
  - Low variant coverage (activity is already optional in many cases)

All thresholds are relative (percentile-based) so they adapt to the log.

Output: List[ResequencingCandidate]
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from .feature_extraction import FeatureStore

HEURISTIC_CONTEXT: str = (
    "Target problem types: inefficient task ordering, unnecessary early "
    "execution of tasks that may later become unnecessary, excessive "
    "waiting caused by suboptimal activity sequence. "
    "Applicable structural situations: activities arranged in a fixed "
    "sequential order although some of them are not required for "
    "subsequent tasks or could be executed later without affecting "
    "correctness. "
    "How the data is used: If activities show weak dependency with "
    "preceding tasks, appear early in the sequence but do not influence "
    "subsequent execution, or frequently become unnecessary due to later "
    "knock-out conditions, their position in the sequence may be "
    "reconsidered. "
    "Transformation style: change the ordering of activities by moving "
    "certain tasks earlier or later in the process sequence while "
    "preserving necessary dependencies. "
    "Feasibility: Yes. Activity ordering relations and dependencies can "
    "be derived from event logs, allowing tasks to be reordered while "
    "preserving necessary constraints."
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
class ResequencingEvidence:
    waiting_time_before: float
    activity_duration: float
    activity_position_ratio: float
    ordering_variance: float
    position_inconsistency_rate: float

    variant_coverage: float

    predecessor_activities: list

    successor_activities: list

    genuine_later_anchor: "str | None" = None
    genuine_later_support: float = 0.0
    genuine_later_position_ratio: float = 0.0

@dataclass
class ResequencingCandidate:
    candidate_id: str
    heuristic: str
    target_activity: str
    heuristic_context: str
    process_fragment: ProcessFragment
    evidence: ResequencingEvidence
    additional_context: str

DURATION_EPSILON = 0.01

POSITION_RATIO_MAX = 0.65

WAITING_TIME_PERCENTILE = 50

POSITION_INCONSISTENCY_MIN = 0.1

ORDERING_VARIANCE_MIN = 0.15

COVERAGE_GUARD = 0.95

COVERAGE_MIN = 0.05

GENUINE_LATER_SUPPORT_MIN = 0.10
GENUINE_LATER_POSITION_MARGIN = 0.10

def _percentile(values: list[float], p: int) -> float:
    """Return the p-th percentile of a list of floats (0 <= p <= 100)."""
    if not values:
        return 0.0
    sv = sorted(values)
    idx = (p / 100) * (len(sv) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sv) - 1)
    return sv[lo] * (1 - (idx - lo)) + sv[hi] * (idx - lo)

def _forward_successors(activity: str, store: FeatureStore) -> list[str]:
    """
    Return successors reached via FORWARD edges only (dependency_strength > 0.5).
    """
    return [
        b for (a, b), strength in store.control_flow.dependency_strength.items()
        if a == activity and strength > 0.5
    ]

def _predecessors(activity: str, store: FeatureStore) -> list[str]:
    """Return all direct predecessors of activity from the DFG."""
    return [a for (a, b) in store.control_flow.dfg_frequency if b == activity]

def _activity_neighbors(
    activity: str,
    store: FeatureStore,
    direction: str,
) -> list[dict]:
    """
    Return the one-hop activity neighbours of activity as observed in the
    event log (DFG), sorted by directly-follows frequency descending.

    direction: "predecessors" -> activities that directly precede activity
               "successors"  -> activities that directly follow  activity

    Only forward edges (dependency_strength > 0.5) are used for successors,
    so loop / rework edges are excluded.  For predecessors all DFG in-edges
    are included because we want to know what actually ran before.
    """
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

@dataclass
class _PositionStats:
    """Internal: position statistics for one activity."""
    position_ratios: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        if not self.position_ratios:
            return 0.5
        return sum(self.position_ratios) / len(self.position_ratios)

    @property
    def cv(self) -> float:
        """Coefficient of variation (std / mean).  0 = perfectly consistent."""
        if len(self.position_ratios) < 2:
            return 0.0
        m = self.mean
        if m == 0:
            return 0.0
        variance = sum((r - m) ** 2 for r in self.position_ratios) / len(
            self.position_ratios
        )
        std = variance ** 0.5
        return std / m

def _compute_position_stats(
    df: pd.DataFrame,
    activities: list[str],
) -> dict[str, _PositionStats]:
    """
    For each activity, compute the list of position_ratio values across
    all cases where the activity appears.

    position_ratio = index_of_activity / (trace_length - 1)
        0.0 = first event in the trace
        1.0 = last event in the trace
        0.5 = used for traces of length 1

    To avoid self-loop / rework inflating positional variance, only the
    FIRST occurrence of each activity per case is recorded.  Repeated
    executions (rework) would otherwise add multiple position entries
    per case — some early, some late — artificially increasing the
    coefficient of variation and making the activity appear more
    "flexible" in its positioning than it really is.
    """
    df_sorted = df.sort_values(
        ["case:concept:name", "time:timestamp"]
    ).copy()

    activity_set = set(activities)
    stats: dict[str, _PositionStats] = {act: _PositionStats() for act in activities}

    for case_id, group in df_sorted.groupby("case:concept:name"):
        acts = group["concept:name"].tolist()
        n = len(acts)
        seen_in_case: set[str] = set()
        for i, act in enumerate(acts):
            if act in activity_set and act not in seen_in_case:
                ratio = i / (n - 1) if n > 1 else 0.5
                stats[act].position_ratios.append(ratio)
                seen_in_case.add(act)

    return stats

def _compute_position_inconsistency_rates(
    store: FeatureStore,
) -> dict[str, float]:
    """
    For each activity, compute the fraction of (activity-containing) variants
    where the activity appears at a LATER relative position than its
    reference baseline.

    Reference baseline:
      - If the activity appears in the dominant variant, use the dominant
        variant's relative position for that activity.
      - If the activity does NOT appear in the dominant variant (common in
        logs like BPI 2012 where the dominant path is short), use the
        MEDIAN relative position of that activity across all containing
        variants (weighted by variant frequency).

    "Later" is defined as: the activity's relative position in a variant
    exceeds the reference baseline by at least one position step
    (1 / (variant_length - 1)).

    Returns {activity: rate} where rate is in [0, 1].
    """
    variant_freq = store.control_flow.variant_frequency
    if not variant_freq:
        return {}

    total_cases = store.total_cases
    if total_cases == 0:
        return {}

    dominant_tuple = max(variant_freq, key=variant_freq.get)

    dominant_positions: dict[str, float] = {}
    dom_len = len(dominant_tuple)
    for i, act in enumerate(dominant_tuple):
        if act not in dominant_positions:
            dominant_positions[act] = i / (dom_len - 1) if dom_len > 1 else 0.5

    activity_positions: dict[str, list[tuple[float, int]]] = defaultdict(list)

    for variant_tuple, count in variant_freq.items():
        v_len = len(variant_tuple)
        if v_len <= 1:
            continue

        first_pos: dict[str, int] = {}
        for i, act in enumerate(variant_tuple):
            if act not in first_pos:
                first_pos[act] = i

        for act, pos_idx in first_pos.items():
            ratio = pos_idx / (v_len - 1)
            activity_positions[act].append((ratio, count))

    reference: dict[str, float] = {}

    for act in store.activities:
        if act in dominant_positions:
            reference[act] = dominant_positions[act]
        elif act in activity_positions:

            pairs = activity_positions[act]

            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            total_weight = sum(c for _, c in pairs_sorted)
            if total_weight == 0:
                reference[act] = 0.5
                continue
            cumulative = 0
            median_ratio = pairs_sorted[-1][0]
            for ratio, count in pairs_sorted:
                cumulative += count
                if cumulative >= total_weight / 2:
                    median_ratio = ratio
                    break
            reference[act] = median_ratio

    activity_later_count: dict[str, int] = defaultdict(int)
    activity_total_cases: dict[str, int] = defaultdict(int)

    for variant_tuple, count in variant_freq.items():
        v_len = len(variant_tuple)
        if v_len <= 1:
            continue

        first_pos: dict[str, int] = {}
        for i, act in enumerate(variant_tuple):
            if act not in first_pos:
                first_pos[act] = i

        for act, pos_idx in first_pos.items():
            if act not in reference:
                continue
            activity_total_cases[act] += count
            variant_ratio = pos_idx / (v_len - 1)
            ref_ratio = reference[act]
            step = 1.0 / (v_len - 1)
            if variant_ratio > ref_ratio + step:
                activity_later_count[act] += count

    rates: dict[str, float] = {}
    for act in store.activities:
        total = activity_total_cases.get(act, 0)
        if total > 0:
            rates[act] = activity_later_count.get(act, 0) / total
        else:
            rates[act] = 0.0

    return rates

def _bfs_fragment_nodes(
    G,
    start_id: str,
    direction: str,
    meta: dict,
    exclude: set,
    max_depth: int = 4,
) -> set:
    """
    BFS from start_id through the BPMN graph.

    direction: "backward" follows in-edges; "forward" follows out-edges.

    Traversal rules:
      - Gateway nodes (XOR/AND/OR/event-based): always traverse through them.
      - Activity and event nodes: include but do NOT traverse further
        (they are the boundary of the fragment).
      - Any node in exclude is skipped entirely.
    """
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
    """
    Return all node IDs reachable from start_id following out-edges,
    with a hop limit to handle BPMN graphs that contain cycles.
    """
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
    """
    Extract a local process fragment centred on the target activity
    directly from the BPMN process graph stored in the FeatureStore.

    Gracefully returns an empty ProcessFragment if the BPMN graph is
    unavailable or the target activity is not found in it.
    """
    G    = store.process_graph.graph
    a2n  = store.process_graph.activity_to_node
    meta = store.process_graph.node_metadata

    target_id = a2n.get(target) if (G is not None and a2n) else None

    if G is None or target_id is None:
        return ProcessFragment()

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
    evidence: ResequencingEvidence,
    store: FeatureStore,
) -> str:
    """
    Produce a structured natural-language note listing supplementary signals
    that cannot be fully resolved from the event log alone.
    """
    parts: list[str] = []

    pos = evidence.activity_position_ratio
    if pos < 0.25:
        pos_label = "early"
    elif pos < 0.5:
        pos_label = "first-half"
    elif pos < 0.75:
        pos_label = "second-half"
    else:
        pos_label = "late"
    parts.append(
        f"This activity currently appears at an average relative position "
        f"of {pos:.2f} in the trace (0 = start, 1 = end), placing it in "
        f"the {pos_label} portion of the process."
    )

    ov = evidence.ordering_variance
    if ov > 0.4:
        parts.append(
            f"The ordering variance (CV) is {ov:.2f}, indicating that the "
            "activity's position varies substantially across cases. "
            "This high variability suggests the process does not strictly "
            "require the activity at its current position, supporting the "
            "feasibility of resequencing."
        )
    elif ov > 0.2:
        parts.append(
            f"The ordering variance (CV) is {ov:.2f}, indicating moderate "
            "variability in when this activity is executed across cases."
        )

    pi = evidence.position_inconsistency_rate
    if pi > 0.3:
        parts.append(
            f"In {pi:.0%} of variants containing this activity, it appears "
            "at a later position than in the dominant variant. This shows "
            "that the process already has a tendency to defer this activity."
        )

    cov = evidence.variant_coverage
    if cov < 0.5:
        parts.append(
            f"The activity appears in only {cov:.0%} of cases. Its low "
            "prevalence suggests it is conditionally required and could "
            "be deferred until the condition is confirmed."
        )

    wt = evidence.waiting_time_before
    dur = evidence.activity_duration
    if dur > 0 and wt / dur > 5.0:
        parts.append(
            f"The waiting time before this activity ({wt:.1f} h) is "
            f"{wt / dur:.0f}x its execution duration ({dur:.2f} h), "
            "suggesting it spends most of its cycle time idle -- a strong "
            "signal that its position in the sequence causes unnecessary "
            "delay."
        )

    rework = store.control_flow.rework_counts.get(activity, 0)
    if rework > 0:
        pct = rework / store.total_cases * 100 if store.total_cases > 0 else 0.0
        parts.append(
            f"This activity exhibits rework in {rework} cases "
            f"({pct:.1f}% of all cases). Rework after an early activity "
            "may indicate that the activity was executed before all "
            "prerequisite information was available."
        )

    parts.append(
        "Whether this activity can be safely moved to a later position "
        "depends on data availability constraints, precedence requirements, "
        "compliance rules, and whether downstream activities rely on its "
        "output -- none of which can be determined from the event log alone."
    )

    return " ".join(parts)

def _is_terminal_activity(act: str) -> bool:
    """
    Terminal / degenerate activities that must not serve as a "later position"
    anchor. An activity appearing after a *_CANCELLED / *_DECLINED step is not
    a genuine resequencing target — the case has already collapsed.
    """
    return act.endswith("_CANCELLED") or act.endswith("_DECLINED")

def _compute_genuine_later_anchors(
    df: pd.DataFrame,
    activities: list[str],
    pos_means: dict[str, float],
) -> dict[str, "tuple[str, float, float] | None"]:
    """
    For each activity, find whether the log contains a *genuine* alternative
    position where the activity runs later behind a stable predecessor.

    An observation of an activity qualifies only if:
      - it has an immediate predecessor in the same case;
      - it is GENUINELY SEQUENTIAL: ``start_timestamp >= predecessor.complete``
        — no temporal overlap with the predecessor. This is the key filter:
        it removes parallel work items that merely *complete* later and get
        sorted after an unrelated event (the BPI-2012 W_/A_ artifact);
      - the predecessor is not the activity itself (excludes rework self-loops);
      - the predecessor is not a terminal/degenerate activity
        (``*_CANCELLED`` / ``*_DECLINED``), which are dead-end anchors.

    Grouping the qualifying observations by predecessor, an anchor survives
    only if it holds in >= ``GENUINE_LATER_SUPPORT_MIN`` of the activity's
    cases AND its mean position ratio exceeds the activity's overall mean
    position by >= ``GENUINE_LATER_POSITION_MARGIN`` (i.e. the activity
    genuinely runs LATER there).

    Returns ``{activity: (best_anchor, support, mean_position_ratio)}`` or
    ``{activity: None}`` when no genuine later position exists.
    """
    result: dict[str, "tuple[str, float, float] | None"] = {
        act: None for act in activities
    }
    if "start_timestamp" not in df.columns:

        return result

    d = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    g = d.groupby("case:concept:name", sort=False)
    d["_prev_act"] = g["concept:name"].shift(1)
    d["_prev_complete"] = g["time:timestamp"].shift(1)

    d["_seq"] = d["start_timestamp"] >= d["_prev_complete"]
    pos_idx = g.cumcount()
    case_len = g["concept:name"].transform("size")
    d["_pos_ratio"] = pos_idx / (case_len - 1).clip(lower=1)

    for activity in set(activities):
        occ_all = d[d["concept:name"] == activity]
        total_cases = occ_all["case:concept:name"].nunique()
        if total_cases == 0:
            continue
        occ = occ_all[
            occ_all["_prev_complete"].notna()
            & occ_all["_seq"]
            & (occ_all["_prev_act"] != activity)
            & ~occ_all["_prev_act"].map(
                _is_terminal_activity, na_action="ignore"
            ).fillna(False)
        ]
        if occ.empty:
            continue
        current_ratio = pos_means.get(activity, 0.5)
        best: "tuple[str, float, float] | None" = None
        for anchor, sub in occ.groupby("_prev_act"):
            support = sub["case:concept:name"].nunique() / total_cases
            mean_ratio = float(sub["_pos_ratio"].mean())
            if support < GENUINE_LATER_SUPPORT_MIN:
                continue
            if mean_ratio < current_ratio + GENUINE_LATER_POSITION_MARGIN:
                continue
            if best is None or mean_ratio > best[2]:
                best = (str(anchor), support, mean_ratio)
        result[activity] = best

    return result

def detect_resequencing(
    store: FeatureStore,
    df: pd.DataFrame,
) -> list[ResequencingCandidate]:
    """
    Scan all activities in the FeatureStore and return those that qualify as
    Resequencing candidates.

    An activity is a candidate if it sits early in the process but shows
    signals that it does not need to be there:
      1. Duration filter: activity is not an instantaneous system event
      2. Coverage guard:  activity does not appear in nearly every case
      3. Position filter: activity appears in the first half of traces
      4. Waiting time:    waiting_time_before > population median
      5. Position inconsistency: a meaningful fraction of variants already
         place this activity later than the dominant variant
      6. Ordering variance: position across cases is not rigidly fixed
      7. Genuine-sequential feasibility: the log actually contains a later,
         non-overlapping position behind a stable, non-terminal predecessor.
         Guards against concurrency/rework artifacts inflating signals 5 & 6.

    Results are sorted by (waiting_time_before DESC).
    """

    pos_stats = _compute_position_stats(df, store.activities)

    pos_means = {
        act: (pos_stats[act].mean if act in pos_stats else 0.5)
        for act in store.activities
    }
    later_anchors = _compute_genuine_later_anchors(df, store.activities, pos_means)

    inconsistency_rates = _compute_position_inconsistency_rates(store)

    all_waiting: list[float] = [
        wt for wt in store.time.waiting_time_mean.values()
        if wt > 0
    ]
    waiting_threshold = _percentile(all_waiting, WAITING_TIME_PERCENTILE)

    start_acts: set[str] = {
        act for act in store.activities if not _predecessors(act, store)
    }
    end_acts: set[str] = {
        act for act in store.activities if not _forward_successors(act, store)
    }
    structural = start_acts | end_acts

    raw: list[tuple[float, ResequencingCandidate]] = []
    counter = 1

    for activity in store.activities:

        if activity in structural:
            continue

        duration = store.time.activity_duration_mean.get(activity, 0.0)
        waiting  = store.time.waiting_time_mean.get(activity, 0.0)
        coverage = store.control_flow.activity_variant_coverage.get(
            activity, 1.0
        )

        p_stats   = pos_stats.get(activity)
        pos_ratio = p_stats.mean if p_stats else 0.5
        pos_cv    = p_stats.cv   if p_stats else 0.0
        incon     = inconsistency_rates.get(activity, 0.0)

        if duration < DURATION_EPSILON:
            continue

        if coverage >= COVERAGE_GUARD:
            continue

        if coverage < COVERAGE_MIN:
            continue

        if pos_ratio > POSITION_RATIO_MAX:
            continue

        if waiting <= waiting_threshold:
            continue

        if incon < POSITION_INCONSISTENCY_MIN:
            continue

        if pos_cv < ORDERING_VARIANCE_MIN:
            continue

        anchor = later_anchors.get(activity)
        if anchor is None:
            continue

        evidence = ResequencingEvidence(
            waiting_time_before=round(waiting, 3),
            activity_duration=round(duration, 4),
            activity_position_ratio=round(pos_ratio, 4),
            ordering_variance=round(pos_cv, 4),
            position_inconsistency_rate=round(incon, 4),
            variant_coverage=round(coverage, 4),
            predecessor_activities=_activity_neighbors(
                activity, store, "predecessors"
            ),
            successor_activities=_activity_neighbors(
                activity, store, "successors"
            ),
            genuine_later_anchor=anchor[0],
            genuine_later_support=round(anchor[1], 4),
            genuine_later_position_ratio=round(anchor[2], 4),
        )

        candidate = ResequencingCandidate(
            candidate_id=f"RSQ-{counter:03d}",
            heuristic="Resequencing",
            target_activity=activity,
            heuristic_context=HEURISTIC_CONTEXT,
            process_fragment=_build_process_fragment(activity, store),
            evidence=evidence,
            additional_context=_build_additional_context(
                activity, evidence, store
            ),
        )
        counter += 1
        raw.append((-waiting, candidate))

    raw.sort(key=lambda x: x[0])
    return [c for _, c in raw]

def print_candidates(candidates: list[ResequencingCandidate]) -> None:
    print(f"\n{'='*70}")
    print(f"RESEQUENCING CANDIDATES  ({len(candidates)} found)")
    print(f"{'='*70}")

    for c in candidates:
        print(f"\n[{c.candidate_id}]  {c.target_activity}")
        print(f"  {'-'*60}")

        e = c.evidence
        print(f"  waiting_time_before          : {e.waiting_time_before:.2f} h")
        print(f"  activity_duration            : {e.activity_duration:.4f} h"
              f"  ({e.activity_duration * 60:.1f} min)")
        print(f"  activity_position_ratio      : {e.activity_position_ratio:.4f}"
              f"  (0=start, 1=end)")
        print(f"  ordering_variance (CV)       : {e.ordering_variance:.4f}")
        print(f"  position_inconsistency_rate  : {e.position_inconsistency_rate:.4f}"
              f"  ({e.position_inconsistency_rate:.1%})")
        print(f"  variant_coverage             : {e.variant_coverage:.4f}"
              f"  ({e.variant_coverage:.1%})")
        print(f"  predecessor_activities       : {e.predecessor_activities}")
        print(f"  successor_activities         : {e.successor_activities}")

        print(f"\n  Process fragment:")
        for node in c.process_fragment.nodes:
            marker = "  < TARGET" if node.is_target else ""
            print(f"    [{node.id}] {node.name}  ({node.type}){marker}")
        for edge in c.process_fragment.edges:
            print(f"    {edge.source} -> {edge.target}")

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

def _candidate_to_dict(c: ResequencingCandidate) -> dict:
    """
    Convert one candidate to a plain dict that matches the specified
    output schema exactly, ready for json.dump().
    """
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
            "waiting_time_before":         c.evidence.waiting_time_before,
            "activity_duration":           c.evidence.activity_duration,
            "activity_position_ratio":     c.evidence.activity_position_ratio,
            "ordering_variance":           c.evidence.ordering_variance,
            "position_inconsistency_rate": c.evidence.position_inconsistency_rate,
            "variant_coverage":            c.evidence.variant_coverage,
            "predecessor_activities":      c.evidence.predecessor_activities,
            "successor_activities":        c.evidence.successor_activities,
        },
        "additional_context": c.additional_context,
    }

def save_candidates_json(
    candidates: list[ResequencingCandidate],
    output_path: str,
) -> None:
    """
    Serialise the full candidate list to a JSON file.

    Top-level structure:
    {
        "heuristic": "Resequencing",
        "total_candidates": <int>,
        "candidates": [ { ... }, ... ]
    }
    """
    import json

    payload = {
        "heuristic":        "Resequencing",
        "total_candidates": len(candidates),
        "candidates":       [_candidate_to_dict(c) for c in candidates],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(candidates)} candidate(s) -> {output_path}")
