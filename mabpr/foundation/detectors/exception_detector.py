"""exception_detector.py"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .feature_extraction import FeatureStore

HEURISTIC_CONTEXT: str = (
    "Target problem types: rare or atypical cases disturbing normal flow, "
    "excessive delays caused by handling exceptional situations within the "
    "main process. "
    "Applicable structural situations: processes where most cases follow a "
    "standard path while a small subset requires special handling but is "
    "currently processed within the same flow. "
    "How the data is used: Rare process variants with significantly "
    "different execution paths or performance patterns compared to the "
    "dominant flow indicate exceptional cases that could be isolated from "
    "the standard process. "
    "Transformation style: redesign the process by separating exceptional "
    "cases from the main flow, typically introducing conditional routing "
    "that directs rare or complex cases to a dedicated exception-handling "
    "branch. "
    "Feasibility: Yes. Exceptional cases can be detected through variant "
    "frequency and deviation patterns derived from the event log, and they "
    "can be separated into dedicated exception branches in the process model."
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
class VariantInfo:
    activity_sequence: list[str]
    frequency: float

@dataclass
class VariantDiff:
    shared_prefix: list[str]
    dominant_suffix: list[str]
    exception_suffix: list[str]
    exception_exclusive_activities: list[str]
    missing_activities: list[str]

@dataclass
class ExceptionEvidence:
    variant_frequency_gap: float
    throughput_time_dominant: float
    throughput_time_exception: float
    throughput_time_gap: float
    deviation_point_waiting_time: float
    exception_case_count: int
    exception_group_frequency: float

    exception_variant_count: int = 1
    throughput_time_exception_max: float = 0.0

@dataclass
class ExceptionCandidate:
    candidate_id: str
    heuristic: str
    divergence_point: str
    heuristic_context: str
    process_fragment: ProcessFragment
    dominant_variant: VariantInfo
    exception_variant: VariantInfo
    variant_diff: VariantDiff
    evidence: ExceptionEvidence
    additional_context: str

EXCEPTION_RATIO_MAX = 0.3

EXCEPTION_CASE_MIN = 10

THROUGHPUT_RATIO_MIN = 1.2

SHARED_PREFIX_MIN = 1

def _build_throughput_index(
    store: FeatureStore,
) -> dict[frozenset, float]:
    """Build a secondary index for throughput_time_per_variant that uses"""

    index: dict[frozenset, float] = {}
    for variant_tuple, tp_time in store.time.throughput_time_per_variant.items():

        counts: dict[str, int] = {}
        elements: list[tuple[str, int]] = []
        for act in variant_tuple:
            counts[act] = counts.get(act, 0) + 1
            elements.append((act, counts[act]))
        key = frozenset(elements)

        if key not in index:
            index[key] = tp_time
    return index

def _lookup_throughput(
    variant_tuple: tuple,
    store: FeatureStore,
    tp_index: dict[frozenset, float],
) -> float:
    """Look up throughput time for a variant, with fuzzy fallback."""

    exact = store.time.throughput_time_per_variant.get(variant_tuple)
    if exact is not None:
        return exact

    counts: dict[str, int] = {}
    elements: list[tuple[str, int]] = []
    for act in variant_tuple:
        counts[act] = counts.get(act, 0) + 1
        elements.append((act, counts[act]))
    key = frozenset(elements)
    return tp_index.get(key, 0.0)

def _find_divergence_point(
    dominant: tuple[str, ...],
    exception: tuple[str, ...],
) -> tuple[str | None, list[str], list[str], list[str]]:
    """Compare two activity sequences and find where they diverge."""

    shared: list[str] = []
    min_len = min(len(dominant), len(exception))

    for i in range(min_len):
        if dominant[i] == exception[i]:
            shared.append(dominant[i])
        else:
            break

    divergence = shared[-1] if shared else None
    dominant_suffix = list(dominant[len(shared):])
    exception_suffix = list(exception[len(shared):])

    return divergence, shared, dominant_suffix, exception_suffix

def _compute_variant_diff(
    dominant: tuple[str, ...],
    exception: tuple[str, ...],
    shared: list[str],
    dominant_suffix: list[str],
    exception_suffix: list[str],
) -> VariantDiff:
    """Build a VariantDiff comparing dominant and exception variants."""
    dominant_set = set(dominant)
    exception_set = set(exception)

    return VariantDiff(
        shared_prefix=shared,
        dominant_suffix=dominant_suffix,
        exception_suffix=exception_suffix,
        exception_exclusive_activities=sorted(exception_set - dominant_set),
        missing_activities=sorted(dominant_set - exception_set),
    )

def _bfs_fragment_nodes(
    G,
    start_id: str,
    direction: str,
    meta: dict,
    exclude: set,
    max_depth: int = 4,
) -> set:
    """BFS through gateways, stopping at activities/events."""
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
    """Return all node IDs reachable forward with hop limit."""
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
    """Extract a local process fragment centred on the divergence point"""

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
    divergence_point: str,
    evidence: ExceptionEvidence,
    variant_diff: VariantDiff,
    store: FeatureStore,
) -> str:
    """Produce a structured natural-language note listing supplementary signals"""

    parts: list[str] = []

    if evidence.throughput_time_gap > 0:
        ratio = (
            evidence.throughput_time_exception / evidence.throughput_time_dominant
            if evidence.throughput_time_dominant > 0 else 0.0
        )
        parts.append(
            f"The exception path takes {evidence.throughput_time_exception:.1f} "
            f"hours on average, which is {ratio:.1f}x slower than the dominant "
            f"path ({evidence.throughput_time_dominant:.1f} hours). Isolating "
            "these cases would remove slow outliers from the main flow."
        )

    excl = variant_diff.exception_exclusive_activities
    if excl:
        parts.append(
            f"The exception path includes {len(excl)} activities not present "
            f"in the dominant flow: {', '.join(excl)}. These activities "
            "represent the additional handling required for exceptional cases."
        )

    miss = variant_diff.missing_activities
    if miss:
        parts.append(
            f"The exception path skips {len(miss)} activities present in the "
            f"dominant flow: {', '.join(miss)}. This may indicate that "
            "exceptional cases follow a shortened or alternative procedure."
        )

    rework_acts: list[str] = []
    for act in excl:
        rework = store.control_flow.rework_counts.get(act, 0)
        if rework > 0:
            rework_acts.append(act)
    if rework_acts:
        parts.append(
            f"Activities with rework in the exception path: "
            f"{', '.join(rework_acts)}. This suggests the exception path "
            "may involve error correction or repeated attempts."
        )

    gf = evidence.exception_group_frequency
    if gf > 0:
        parts.append(
            f"All exception variants diverging at '{divergence_point}' "
            f"account for {gf:.1%} of total cases "
            f"({evidence.exception_case_count} cases). "
            "This represents the total volume that would be routed to "
            "the dedicated exception branch."
        )

    parts.append(
        "Whether these exception cases correspond to known business rules, "
        "whether a dedicated handling process already exists elsewhere, "
        "and whether service level agreements apply differently to "
        "exception cases cannot be determined from the event log alone."
    )

    return " ".join(parts)

def detect_exception(
    store: FeatureStore,
    report: dict | None = None,
) -> list[ExceptionCandidate]:
    """Detect Exception candidates by comparing rare variants against the"""

    variant_freq = store.control_flow.variant_frequency
    if not variant_freq:
        print("  No variants found in the event log.")
        return []

    total_cases = store.total_cases
    if total_cases == 0:
        return []

    dominant_tuple = max(variant_freq, key=variant_freq.get)
    dominant_count = variant_freq[dominant_tuple]
    dominant_freq  = dominant_count / total_cases

    print(f"  Dominant variant: {len(dominant_tuple)} activities, "
          f"{dominant_count} cases ({dominant_freq:.1%})")
    print(f"  Total variants: {len(variant_freq)}")

    tp_index = _build_throughput_index(store)

    ActionKey = tuple
    action_groups: dict[ActionKey, list[tuple]] = defaultdict(list)
    unactionable: list[dict] = []

    for variant_tuple, count in variant_freq.items():
        if variant_tuple == dominant_tuple:
            continue

        freq = count / total_cases

        if freq > dominant_freq * EXCEPTION_RATIO_MAX:
            continue

        if count < EXCEPTION_CASE_MIN:
            continue

        div_point, shared, dom_suffix, exc_suffix = _find_divergence_point(
            dominant_tuple, variant_tuple
        )

        if len(shared) < SHARED_PREFIX_MIN:
            continue

        if div_point is None:
            continue

        tp_dominant   = _lookup_throughput(dominant_tuple, store, tp_index)
        tp_exception  = _lookup_throughput(variant_tuple, store, tp_index)

        if tp_dominant > 0 and tp_exception / tp_dominant < THROUGHPUT_RATIO_MIN:
            continue

        diff = _compute_variant_diff(
            dominant_tuple, variant_tuple, shared, dom_suffix, exc_suffix
        )

        if not diff.exception_exclusive_activities:
            unactionable.append({
                "divergence_point": div_point,
                "case_count": count,
                "frequency": round(freq, 6),
                "throughput_time_exception": round(tp_exception, 3),
                "missing_activities": diff.missing_activities,

                "pattern": ("skips_dominant_activities" if diff.missing_activities
                            else "reordering_or_rework"),
                "reason": ("no exception-exclusive activities: the Exception redesign "
                           "has nothing to lift out, and the handler locates the "
                           "exception branch by exactly these activity names"),
            })
            continue

        key = (div_point, frozenset(diff.exception_exclusive_activities))
        action_groups[key].append((
            variant_tuple, count, freq, tp_exception, shared, diff
        ))

    print(f"  Actionable exception patterns (divergence point x isolatable "
          f"activity set): {len(action_groups)}")
    if unactionable:
        skipping = sum(1 for u in unactionable if u["pattern"] == "skips_dominant_activities")
        print(f"  Rare-and-slow variants the Exception redesign cannot act on: "
              f"{len(unactionable)} ({skipping} skip dominant activities, "
              f"{len(unactionable) - skipping} reorder/rework only)")
    if report is not None:
        report["unactionable_variants"] = unactionable

    if not action_groups:
        return []

    tp_dominant = _lookup_throughput(dominant_tuple, store, tp_index)

    raw: list[tuple[float, ExceptionCandidate]] = []
    counter = 1

    for (div_point, _exclusive), group in action_groups.items():

        group_total_count = sum(item[1] for item in group)
        group_total_freq  = sum(item[2] for item in group)

        group.sort(key=lambda x: x[1], reverse=True)
        rep_tuple, rep_count, rep_freq, rep_tp, rep_shared, rep_diff = group[0]

        tp_exception = (sum(item[1] * item[3] for item in group) / group_total_count
                        if group_total_count else rep_tp)
        tp_gap = tp_exception - tp_dominant
        waiting_at_div = store.time.waiting_time_mean.get(div_point, 0.0)

        evidence = ExceptionEvidence(
            variant_frequency_gap=round(dominant_freq - rep_freq, 6),
            throughput_time_dominant=round(tp_dominant, 3),
            throughput_time_exception=round(tp_exception, 3),
            throughput_time_gap=round(tp_gap, 3),
            deviation_point_waiting_time=round(waiting_at_div, 3),
            exception_case_count=group_total_count,
            exception_group_frequency=round(group_total_freq, 6),
            exception_variant_count=len(group),
            throughput_time_exception_max=round(max(item[3] for item in group), 3),
        )

        candidate = ExceptionCandidate(
            candidate_id=f"EXC-{counter:03d}",
            heuristic="Exception",
            divergence_point=div_point,
            heuristic_context=HEURISTIC_CONTEXT,
            process_fragment=_build_process_fragment(div_point, store),
            dominant_variant=VariantInfo(
                activity_sequence=list(dominant_tuple),
                frequency=round(dominant_freq, 6),
            ),
            exception_variant=VariantInfo(
                activity_sequence=list(rep_tuple),
                frequency=round(rep_freq, 6),
            ),
            variant_diff=rep_diff,
            evidence=evidence,
            additional_context=_build_additional_context(
                div_point, evidence, rep_diff, store
            ),
        )
        counter += 1
        raw.append((-tp_gap, candidate))

    raw.sort(key=lambda x: x[0])
    return [c for _, c in raw]

def print_candidates(candidates: list[ExceptionCandidate]) -> None:
    print(f"\n{'='*70}")
    print(f"EXCEPTION CANDIDATES  ({len(candidates)} found)")
    print(f"{'='*70}")

    for c in candidates:
        print(f"\n[{c.candidate_id}]  divergence at: {c.divergence_point}")
        print(f"  {'─'*60}")

        e = c.evidence
        print(f"  variant_frequency_gap        : {e.variant_frequency_gap:.4f}")
        print(f"  throughput_time_dominant      : {e.throughput_time_dominant:.1f} h")
        print(f"  throughput_time_exception     : {e.throughput_time_exception:.1f} h")
        print(f"  throughput_time_gap           : {e.throughput_time_gap:+.1f} h")
        print(f"  deviation_point_waiting_time  : {e.deviation_point_waiting_time:.2f} h")
        print(f"  exception_case_count          : {e.exception_case_count}")
        print(f"  exception_group_frequency     : {e.exception_group_frequency:.4f}"
              f"  ({e.exception_group_frequency:.1%})")

        print(f"\n  Dominant variant ({c.dominant_variant.frequency:.1%}):")
        print(f"    {' → '.join(c.dominant_variant.activity_sequence)}")

        print(f"\n  Exception variant ({c.exception_variant.frequency:.1%}):")
        print(f"    {' → '.join(c.exception_variant.activity_sequence)}")

        d = c.variant_diff
        print(f"\n  Variant diff:")
        print(f"    shared_prefix  : {' → '.join(d.shared_prefix) if d.shared_prefix else '(none)'}")
        print(f"    dominant_suffix: {' → '.join(d.dominant_suffix) if d.dominant_suffix else '(none)'}")
        print(f"    exception_suffix: {' → '.join(d.exception_suffix) if d.exception_suffix else '(none)'}")
        if d.exception_exclusive_activities:
            print(f"    exception-only activities: {', '.join(d.exception_exclusive_activities)}")
        if d.missing_activities:
            print(f"    missing from exception  : {', '.join(d.missing_activities)}")

        print(f"\n  Process fragment:")
        for node in c.process_fragment.nodes:
            marker = "  ◀ DIVERGENCE" if node.is_target else ""
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

def _candidate_to_dict(c: ExceptionCandidate) -> dict:
    """Convert one candidate to a plain dict for json.dump()."""
    return {
        "candidate_id": c.candidate_id,
        "heuristic": c.heuristic,
        "divergence_point": c.divergence_point,
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
        "dominant_variant": {
            "activity_sequence": c.dominant_variant.activity_sequence,
            "frequency":         c.dominant_variant.frequency,
        },
        "exception_variant": {
            "activity_sequence": c.exception_variant.activity_sequence,
            "frequency":         c.exception_variant.frequency,
        },
        "variant_diff": {
            "shared_prefix":                  c.variant_diff.shared_prefix,
            "dominant_suffix":                c.variant_diff.dominant_suffix,
            "exception_suffix":               c.variant_diff.exception_suffix,
            "exception_exclusive_activities": c.variant_diff.exception_exclusive_activities,
            "missing_activities":             c.variant_diff.missing_activities,
        },
        "evidence": {
            "variant_frequency_gap":        c.evidence.variant_frequency_gap,
            "throughput_time_dominant":      c.evidence.throughput_time_dominant,
            "throughput_time_exception":     c.evidence.throughput_time_exception,
            "throughput_time_gap":           c.evidence.throughput_time_gap,
            "deviation_point_waiting_time":  c.evidence.deviation_point_waiting_time,
            "exception_case_count":         c.evidence.exception_case_count,
            "exception_group_frequency":    c.evidence.exception_group_frequency,
            "exception_variant_count":      c.evidence.exception_variant_count,
            "throughput_time_exception_max": c.evidence.throughput_time_exception_max,
        },
        "additional_context": c.additional_context,
    }

def save_candidates_json(
    candidates: list[ExceptionCandidate],
    output_path: str,
    report: dict | None = None,
) -> None:
    """Serialise the full candidate list to a JSON file."""

    import json

    payload = {
        "heuristic":        "Exception",
        "total_candidates": len(candidates),
        "candidates":       [_candidate_to_dict(c) for c in candidates],
    }
    if report is not None:
        payload["unactionable_variants"] = report.get("unactionable_variants", [])

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(candidates)} candidate(s) → {output_path}")
