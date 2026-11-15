"""BPMN graph mutation operations layered on pm4py."""

from __future__ import annotations

from dataclasses import dataclass

from pm4py.objects.bpmn.obj import BPMN

@dataclass(frozen=True)
class WrapperContext:
    """Shape of one XOR wrapper layer captured around an activity X."""

    had_skip: bool
    had_loop: bool

@dataclass(frozen=True)
class WrapperLayerIds:
    """IDs of one XOR wrapper layer around an activity X."""

    gw_a_id:           str
    gw_b_id:           str
    wrap_in_flow_id:   str
    forward_flow_id:   str
    skip_flow_id:      str | None
    loop_flow_id:      str | None

def _predecessors(bpmn: BPMN, node) -> list:
    """Direct predecessors of *node* (one hop up via incoming sequence flows)."""
    return [f.get_source() for f in bpmn.get_flows() if f.get_target() is node]

def _successors(bpmn: BPMN, node) -> list:
    """Direct successors of *node* (one hop down via outgoing sequence flows)."""
    return [f.get_target() for f in bpmn.get_flows() if f.get_source() is node]

def _has_direct_edge(bpmn: BPMN, source, target) -> bool:
    """True iff there is a sequence flow ``source → target`` in *bpmn*."""
    return any(
        f.get_source() is source and f.get_target() is target
        for f in bpmn.get_flows()
    )

def _remove_node_and_flows(bpmn: BPMN, node) -> None:
    """Remove *node* and every sequence flow that touches it (pm4py's"""

    incident = [
        f for f in bpmn.get_flows()
        if f.get_source() is node or f.get_target() is node
    ]
    for flow in incident:
        bpmn.remove_flow(flow)
    bpmn.remove_node(node)

def _process_of(bpmn: BPMN, *nodes):
    """The process id a NEW flow must carry."""

    for node in nodes:
        pid = getattr(node, "get_process", lambda: None)()
        if pid:
            return pid
    for f in bpmn.get_flows():
        pid = f.get_process()
        if pid:
            return pid
    for nd in bpmn.get_nodes():
        pid = nd.get_process()
        if pid:
            return pid
    return None

def _add_flow_unique(bpmn: BPMN, source, target) -> None:
    """Add a sequence flow source→target unless it already exists or would be"""

    if source is target:
        return
    if any(
        f.get_source() is source and f.get_target() is target
        for f in bpmn.get_flows()
    ):
        return
    bpmn.add_flow(BPMN.SequenceFlow(source=source, target=target,
                                    process=_process_of(bpmn, source, target)))

def _try_collapse_empty_xor_pair(bpmn: BPMN, gw_a, gw_b) -> bool:
    """If *gw_a* and *gw_b* are XOR gateways with no node on any *local*"""

    if not (
        isinstance(gw_a, BPMN.ExclusiveGateway)
        and isinstance(gw_b, BPMN.ExclusiveGateway)
    ):
        return False

    for s in _successors(bpmn, gw_a):
        if s is gw_b:
            continue
        if _has_direct_edge(bpmn, s, gw_b):
            return False

    for s in _successors(bpmn, gw_b):
        if s is gw_a:
            continue
        if _has_direct_edge(bpmn, s, gw_a):
            return False

    inter_edges = [
        f for f in bpmn.get_flows()
        if (f.get_source() is gw_a and f.get_target() is gw_b)
        or (f.get_source() is gw_b and f.get_target() is gw_a)
    ]
    for f in inter_edges:
        bpmn.remove_flow(f)

    a_preds = _predecessors(bpmn, gw_a)
    a_succs = _successors(bpmn, gw_a)
    b_preds = _predecessors(bpmn, gw_b)
    b_succs = _successors(bpmn, gw_b)

    if not a_succs and not b_preds:

        bridge_from, bridge_to = a_preds, b_succs
    elif not a_preds and not b_succs:

        bridge_from, bridge_to = b_preds, a_succs
    else:

        for f in inter_edges:
            bpmn.add_flow(f)
        return False

    for p in bridge_from:
        for s in bridge_to:
            if p is s:
                continue
            bpmn.add_flow(BPMN.SequenceFlow(source=p, target=s,
                                            process=_process_of(bpmn, p, s)))

    _remove_node_and_flows(bpmn, gw_a)
    _remove_node_and_flows(bpmn, gw_b)

    for p in bridge_from:
        for s in bridge_to:
            if p is s:
                continue
            _try_collapse_empty_xor_pair(bpmn, p, s)

    return True

_NOT_A_TIMER = (BPMN.MessageIntermediateCatchEvent,
                BPMN.CancelIntermediateCatchEvent,
                BPMN.ErrorIntermediateCatchEvent)

def past_timers(bpmn: BPMN, node, forward: bool = True, limit: int = 4):
    """The next real node past any fitted extraneous-delay timers. READ ONLY."""

    if node is None:
        return node
    cur = node
    for _ in range(limit):
        flows = [f for f in bpmn.get_flows()
                 if (f.get_source() if forward else f.get_target()) is cur]
        if len(flows) != 1:
            return cur
        nxt = flows[0].get_target() if forward else flows[0].get_source()
        if not isinstance(nxt, BPMN.IntermediateCatchEvent) or isinstance(nxt, _NOT_A_TIMER):
            return cur

        ins = [f for f in bpmn.get_flows() if f.get_target() is nxt]
        outs = [f for f in bpmn.get_flows() if f.get_source() is nxt]
        if len(ins) != 1 or len(outs) != 1:
            return cur
        cur = nxt
    return cur

def strip_preceding_timers(bpmn: BPMN, target) -> list:
    """Remove the extraneous-delay timer(s) immediately in front of *target*,"""

    removed = []
    for pred in list(_predecessors(bpmn, target)):
        if not isinstance(pred, BPMN.IntermediateCatchEvent):
            continue
        if isinstance(pred, _NOT_A_TIMER):
            continue
        ins = _predecessors(bpmn, pred)
        outs = _successors(bpmn, pred)
        if len(ins) != 1 or len(outs) != 1 or outs[0] is not target:
            continue
        _remove_node_and_flows(bpmn, pred)
        if ins[0] is not target:
            _add_flow_unique(bpmn, ins[0], target)
        removed.append(str(pred.get_id()))
    return removed

def drop_event_distributions(params: dict, timer_ids) -> tuple:
    """Drop ``event_distribution`` entries for *timer_ids*. Returns"""

    ids = {str(t) for t in (timer_ids or ())}
    if not ids:
        return 0, 0.0
    kept, hours, n = [], 0.0, 0
    for e in params.get("event_distribution", []) or []:
        if str(e.get("event_id")) in ids:
            n += 1
            prm = e.get("distribution_params") or []
            if prm:
                v = prm[0].get("value") if isinstance(prm[0], dict) else prm[0]
                hours += float(v or 0.0) / 3600.0
        else:
            kept.append(e)
    params["event_distribution"] = kept
    return n, round(hours, 4)

def delete_activity(bpmn: BPMN, activity_name: str) -> None:
    """Remove the activity named *activity_name* from *bpmn*, in place."""

    matches = [n for n in bpmn.get_nodes() if n.get_name() == activity_name]
    if not matches:
        raise ValueError(f"No node named {activity_name!r} in BPMN")

    for target in matches:

        preds = _predecessors(bpmn, target)
        succs = _successors(bpmn, target)

        sandwich = (
            len(preds) == 1 and isinstance(preds[0], BPMN.ExclusiveGateway)
            and len(succs) == 1 and isinstance(succs[0], BPMN.ExclusiveGateway)
        )
        gw_a = preds[0] if sandwich else None
        gw_b = succs[0] if sandwich else None

        only_edge_through_target = False
        if sandwich:
            a_succs = _successors(bpmn, gw_a)
            b_preds = _predecessors(bpmn, gw_b)
            only_edge_through_target = (
                len(a_succs) == 1 and a_succs[0] is target
                and len(b_preds) == 1 and b_preds[0] is target
                and not _has_direct_edge(bpmn, gw_b, gw_a)
            )

        _remove_node_and_flows(bpmn, target)

        for p in preds:
            for s in succs:
                _add_flow_unique(bpmn, p, s)

        if sandwich and not only_edge_through_target:
            _try_collapse_empty_xor_pair(bpmn, gw_a, gw_b)

def insert_activity_after(
    bpmn: BPMN,
    activity_name: str,
    anchor_name: str,
    task_id: str | None = None,
) -> None:
    """Insert a fresh Task named *activity_name* immediately after *anchor_name*."""

    matches = [n for n in bpmn.get_nodes() if n.get_name() == anchor_name]
    if not matches:
        raise ValueError(f"No anchor node named {anchor_name!r} in BPMN")
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} nodes named {anchor_name!r} in BPMN; ambiguous"
        )
    anchor = matches[0]

    out_flows = [f for f in bpmn.get_flows() if f.get_source() is anchor]
    if len(out_flows) == 0:
        raise ValueError(
            f"Anchor {anchor_name!r} has no outgoing flow — nothing to split"
        )
    if len(out_flows) > 1:
        raise ValueError(
            f"Anchor {anchor_name!r} has {len(out_flows)} outgoing flows; "
            "insertion point not unique"
        )
    target_flow = out_flows[0]
    v = target_flow.get_target()
    process = target_flow.get_process()

    if task_id is not None:
        new_node = BPMN.Task(name=activity_name, id=task_id, process=process)
    else:
        new_node = BPMN.Task(name=activity_name, process=process)
    bpmn.add_node(new_node)

    bpmn.remove_flow(target_flow)
    bpmn.add_flow(BPMN.SequenceFlow(source=anchor, target=new_node, process=process))
    bpmn.add_flow(BPMN.SequenceFlow(source=new_node, target=v, process=process))

def insert_activity_before(
    bpmn: BPMN,
    activity_name: str,
    anchor_name: str,
    task_id: str | None = None,
) -> None:
    """Insert a fresh Task named *activity_name* immediately before *anchor_name*."""

    matches = [n for n in bpmn.get_nodes() if n.get_name() == anchor_name]
    if not matches:
        raise ValueError(f"No anchor node named {anchor_name!r} in BPMN")
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} nodes named {anchor_name!r} in BPMN; ambiguous"
        )
    anchor = matches[0]

    in_flows = [f for f in bpmn.get_flows() if f.get_target() is anchor]
    if len(in_flows) == 0:
        raise ValueError(
            f"Anchor {anchor_name!r} has no incoming flow — nothing to split"
        )
    if len(in_flows) > 1:
        raise ValueError(
            f"Anchor {anchor_name!r} has {len(in_flows)} incoming flows; "
            "insertion point not unique"
        )
    target_flow = in_flows[0]
    pred = target_flow.get_source()
    process = target_flow.get_process()

    if task_id is not None:
        new_node = BPMN.Task(name=activity_name, id=task_id, process=process)
    else:
        new_node = BPMN.Task(name=activity_name, process=process)
    bpmn.add_node(new_node)

    bpmn.remove_flow(target_flow)
    bpmn.add_flow(BPMN.SequenceFlow(source=pred, target=new_node, process=process))
    bpmn.add_flow(BPMN.SequenceFlow(source=new_node, target=anchor, process=process))

def parallelize_activities(
    bpmn: BPMN,
    activity_a_name: str,
    activity_b_name: str,
) -> None:
    """Convert a sequential pair (A → B) into a parallel block."""

    a_matches = [n for n in bpmn.get_nodes() if n.get_name() == activity_a_name]
    if not a_matches:
        raise ValueError(f"No node named {activity_a_name!r} in BPMN")
    if len(a_matches) > 1:
        raise ValueError(
            f"{len(a_matches)} nodes named {activity_a_name!r}; ambiguous"
        )
    A = a_matches[0]

    b_matches = [n for n in bpmn.get_nodes() if n.get_name() == activity_b_name]
    if not b_matches:
        raise ValueError(f"No node named {activity_b_name!r} in BPMN")
    if len(b_matches) > 1:
        raise ValueError(
            f"{len(b_matches)} nodes named {activity_b_name!r}; ambiguous"
        )
    B = b_matches[0]

    a_succs = _successors(bpmn, A)
    if len(a_succs) != 1 or a_succs[0] is not B:
        raise ValueError(
            f"{activity_a_name!r} must have exactly one successor and it "
            f"must be {activity_b_name!r}; got {len(a_succs)} successor(s)"
        )

    b_preds = _predecessors(bpmn, B)
    if len(b_preds) != 1 or b_preds[0] is not A:
        raise ValueError(
            f"{activity_b_name!r} must have exactly one predecessor and it "
            f"must be {activity_a_name!r}; got {len(b_preds)} predecessor(s)"
        )

    a_preds = _predecessors(bpmn, A)
    if len(a_preds) != 1:
        raise ValueError(
            f"{activity_a_name!r} must have exactly one predecessor "
            f"(the anchor before the parallel block); got {len(a_preds)}"
        )
    pred = a_preds[0]

    b_succs = _successors(bpmn, B)
    if len(b_succs) != 1:
        raise ValueError(
            f"{activity_b_name!r} must have exactly one successor "
            f"(the anchor after the parallel block); got {len(b_succs)}"
        )
    succ = b_succs[0]

    pred_to_a = next(
        f for f in bpmn.get_flows()
        if f.get_source() is pred and f.get_target() is A
    )
    a_to_b = next(
        f for f in bpmn.get_flows()
        if f.get_source() is A and f.get_target() is B
    )
    b_to_succ = next(
        f for f in bpmn.get_flows()
        if f.get_source() is B and f.get_target() is succ
    )
    process = (
        pred_to_a.get_process()
        or a_to_b.get_process()
        or b_to_succ.get_process()
    )

    old_pred_to_a_id = str(pred_to_a.get_id())

    and_split = BPMN.ParallelGateway(name="", process=process)
    and_join = BPMN.ParallelGateway(name="", process=process)
    bpmn.add_node(and_split)
    bpmn.add_node(and_join)

    bpmn.remove_flow(pred_to_a)
    bpmn.remove_flow(a_to_b)
    bpmn.remove_flow(b_to_succ)

    bpmn.add_flow(BPMN.SequenceFlow(
        source=pred, target=and_split, id=old_pred_to_a_id, process=process,
    ))
    bpmn.add_flow(BPMN.SequenceFlow(source=and_split, target=A,         process=process))
    bpmn.add_flow(BPMN.SequenceFlow(source=and_split, target=B,         process=process))
    bpmn.add_flow(BPMN.SequenceFlow(source=A,         target=and_join,  process=process))
    bpmn.add_flow(BPMN.SequenceFlow(source=B,         target=and_join,  process=process))
    bpmn.add_flow(BPMN.SequenceFlow(source=and_join,  target=succ,      process=process))

def encapsulate_chain_as_subprocess(
    bpmn: BPMN,
    chain_node_ids: list[str],
    subprocess_name: str,
) -> "BPMN.Task":
    """Replace a single-in / single-out chain of nodes with a fresh"""

    if not chain_node_ids:
        raise ValueError("chain_node_ids must be non-empty")

    id_to_node = {n.get_id(): n for n in bpmn.get_nodes()}
    chain_nodes: list = []
    chain_id_set: set[str] = set()
    for nid in chain_node_ids:
        if nid not in id_to_node:
            raise ValueError(f"chain node id {nid!r} not found in BPMN")
        if nid in chain_id_set:
            raise ValueError(f"chain node id {nid!r} appears more than once")
        chain_nodes.append(id_to_node[nid])
        chain_id_set.add(nid)

    head = chain_nodes[0]
    tail = chain_nodes[-1]

    external_in_edges = [
        f for f in bpmn.get_flows()
        if f.get_source().get_id() not in chain_id_set
        and f.get_target().get_id() in chain_id_set
    ]
    external_out_edges = [
        f for f in bpmn.get_flows()
        if f.get_source().get_id() in chain_id_set
        and f.get_target().get_id() not in chain_id_set
    ]

    if len(external_in_edges) != 1:
        raise ValueError(
            f"chain must have exactly one external predecessor edge; "
            f"got {len(external_in_edges)}"
        )
    if len(external_out_edges) != 1:
        raise ValueError(
            f"chain must have exactly one external successor edge; "
            f"got {len(external_out_edges)}"
        )

    in_edge = external_in_edges[0]
    out_edge = external_out_edges[0]

    if in_edge.get_target() is not head:
        raise ValueError(
            "external predecessor edge does not point to chain[0]; "
            "chain ordering is inconsistent with BPMN topology"
        )
    if out_edge.get_source() is not tail:
        raise ValueError(
            "external successor edge does not start from chain[-1]; "
            "chain ordering is inconsistent with BPMN topology"
        )

    external_pred = in_edge.get_source()
    external_succ = out_edge.get_target()
    process = in_edge.get_process() or out_edge.get_process()

    old_in_edge_id = str(in_edge.get_id())

    subprocess_node = BPMN.Task(name=subprocess_name, process=process)
    bpmn.add_node(subprocess_node)

    for node in chain_nodes:
        _remove_node_and_flows(bpmn, node)

    bpmn.add_flow(BPMN.SequenceFlow(
        source=external_pred, target=subprocess_node,
        id=old_in_edge_id, process=process,
    ))
    bpmn.add_flow(BPMN.SequenceFlow(
        source=subprocess_node, target=external_succ, process=process,
    ))

    return subprocess_node

def detect_xor_wrapper_cascade(
    bpmn: BPMN,
    activity_name: str,
) -> list[WrapperContext]:
    """Walk the recursive XOR-wrapper cascade around *activity_name* and"""

    matches = [n for n in bpmn.get_nodes() if n.get_name() == activity_name]
    if len(matches) != 1:
        return []

    inside_set: set = {matches[0]}

    wrappers: list[WrapperContext] = []

    while True:

        op_seen: set = set()
        outer_preds: list = []
        for n in inside_set:
            for p in _predecessors(bpmn, n):
                if p in inside_set or id(p) in op_seen:
                    continue
                op_seen.add(id(p))
                outer_preds.append(p)

        os_seen: set = set()
        outer_succs: list = []
        for n in inside_set:
            for s in _successors(bpmn, n):
                if s in inside_set or id(s) in os_seen:
                    continue
                os_seen.add(id(s))
                outer_succs.append(s)

        if len(outer_preds) != 1 or len(outer_succs) != 1:
            break
        gw_a = outer_preds[0]
        gw_b = outer_succs[0]

        if not (
            isinstance(gw_a, BPMN.ExclusiveGateway)
            and isinstance(gw_b, BPMN.ExclusiveGateway)
        ):
            break

        forward_violation = False
        for s in _successors(bpmn, gw_a):
            if s is gw_b or s in inside_set:
                continue
            if _has_direct_edge(bpmn, s, gw_b):
                forward_violation = True
                break
        if forward_violation:
            break

        backward_violation = False
        for s in _successors(bpmn, gw_b):
            if s is gw_a or s in inside_set:
                continue
            if _has_direct_edge(bpmn, s, gw_a):
                backward_violation = True
                break
        if backward_violation:
            break

        a_outer_preds = [
            p for p in _predecessors(bpmn, gw_a)
            if p is not gw_b and p not in inside_set
        ]
        b_outer_succs = [
            s for s in _successors(bpmn, gw_b)
            if s is not gw_a and s not in inside_set
        ]
        if len(a_outer_preds) != 1 or len(b_outer_succs) != 1:

            had_skip = _has_direct_edge(bpmn, gw_a, gw_b)
            had_loop = _has_direct_edge(bpmn, gw_b, gw_a)
            if had_skip or had_loop:
                wrappers.append(WrapperContext(had_skip=had_skip, had_loop=had_loop))
            break

        had_skip = _has_direct_edge(bpmn, gw_a, gw_b)
        had_loop = _has_direct_edge(bpmn, gw_b, gw_a)
        if not (had_skip or had_loop):
            break

        wrappers.append(WrapperContext(had_skip=had_skip, had_loop=had_loop))

        inside_set.add(gw_a)
        inside_set.add(gw_b)

    return wrappers

def detect_xor_wrapper_cascade_with_ids(
    bpmn: BPMN,
    activity_name: str,
) -> list[WrapperLayerIds]:
    """Walk the same XOR-wrapper cascade as"""

    matches = [n for n in bpmn.get_nodes() if n.get_name() == activity_name]
    if len(matches) != 1:
        return []

    inside_set: set = {matches[0]}
    layers: list[WrapperLayerIds] = []

    while True:
        op_seen: set = set()
        outer_preds: list = []
        for n in inside_set:
            for p in _predecessors(bpmn, n):
                if p in inside_set or id(p) in op_seen:
                    continue
                op_seen.add(id(p))
                outer_preds.append(p)

        os_seen: set = set()
        outer_succs: list = []
        for n in inside_set:
            for s in _successors(bpmn, n):
                if s in inside_set or id(s) in os_seen:
                    continue
                os_seen.add(id(s))
                outer_succs.append(s)

        if len(outer_preds) != 1 or len(outer_succs) != 1:
            break
        gw_a = outer_preds[0]
        gw_b = outer_succs[0]
        if not (
            isinstance(gw_a, BPMN.ExclusiveGateway)
            and isinstance(gw_b, BPMN.ExclusiveGateway)
        ):
            break

        forward_violation = False
        for s in _successors(bpmn, gw_a):
            if s is gw_b or s in inside_set:
                continue
            if _has_direct_edge(bpmn, s, gw_b):
                forward_violation = True
                break
        if forward_violation:
            break

        backward_violation = False
        for s in _successors(bpmn, gw_b):
            if s is gw_a or s in inside_set:
                continue
            if _has_direct_edge(bpmn, s, gw_a):
                backward_violation = True
                break
        if backward_violation:
            break

        a_outer_preds = [
            p for p in _predecessors(bpmn, gw_a)
            if p is not gw_b and p not in inside_set
        ]
        b_outer_succs = [
            s for s in _successors(bpmn, gw_b)
            if s is not gw_a and s not in inside_set
        ]

        had_skip = _has_direct_edge(bpmn, gw_a, gw_b)
        had_loop = _has_direct_edge(bpmn, gw_b, gw_a)

        if (len(a_outer_preds) != 1 or len(b_outer_succs) != 1) and not (had_skip or had_loop):
            break
        if not (had_skip or had_loop):
            break

        skip_flow_id = None
        loop_flow_id = None
        wrap_in_flow_id = None
        forward_flow_id = None
        for f in bpmn.get_flows():
            src, tgt = f.get_source(), f.get_target()
            if src is gw_a and tgt is gw_b:
                skip_flow_id = str(f.get_id())
            elif src is gw_b and tgt is gw_a:
                loop_flow_id = str(f.get_id())
            elif src is gw_a and tgt is not gw_b:

                wrap_in_flow_id = str(f.get_id())
            elif src is gw_b and tgt is not gw_a:

                forward_flow_id = str(f.get_id())

        if wrap_in_flow_id is None or forward_flow_id is None:

            break

        layers.append(WrapperLayerIds(
            gw_a_id=str(gw_a.get_id()),
            gw_b_id=str(gw_b.get_id()),
            wrap_in_flow_id=wrap_in_flow_id,
            forward_flow_id=forward_flow_id,
            skip_flow_id=skip_flow_id,
            loop_flow_id=loop_flow_id,
        ))

        if len(a_outer_preds) != 1 or len(b_outer_succs) != 1:

            break

        inside_set.add(gw_a)
        inside_set.add(gw_b)

    return layers

def insert_activity_after_with_nested_xor_wraps(
    bpmn: BPMN,
    activity_name: str,
    anchor_name: str,
    wrappers: list[WrapperContext],
    wrapper_ids: list[WrapperLayerIds] | None = None,
    task_id: str | None = None,
) -> None:
    """Insert *activity_name* after *anchor_name*, then wrap it in N"""

    insert_activity_after(bpmn, activity_name, anchor_name, task_id=task_id)

    if not wrappers:
        return

    anchor_matches = [n for n in bpmn.get_nodes() if n.get_name() == anchor_name]
    if len(anchor_matches) != 1:

        raise ValueError(
            f"Anchor {anchor_name!r} no longer unique after plain insert"
        )
    anchor = anchor_matches[0]

    a_succs_after = _successors(bpmn, anchor)
    if len(a_succs_after) != 1:
        raise ValueError(
            f"Anchor {anchor_name!r} unexpectedly has "
            f"{len(a_succs_after)} successors after plain insert"
        )
    new_x = a_succs_after[0]
    x_succs = _successors(bpmn, new_x)
    if len(x_succs) != 1:
        raise ValueError(
            f"Inserted activity {activity_name!r} unexpectedly has "
            f"{len(x_succs)} successors after plain insert"
        )
    v = x_succs[0]

    reuse_per_layer: list[WrapperLayerIds | None] = []
    if wrapper_ids is not None:
        existing_node_ids = {str(n.get_id()) for n in bpmn.get_nodes()}
        for i in range(len(wrappers)):
            cand = wrapper_ids[i] if i < len(wrapper_ids) else None
            if cand is None:
                reuse_per_layer.append(None)
                continue
            if cand.gw_a_id in existing_node_ids or cand.gw_b_id in existing_node_ids:
                reuse_per_layer.append(None)
            else:
                reuse_per_layer.append(cand)
    else:
        reuse_per_layer = [None] * len(wrappers)

    current_head = new_x
    current_tail = new_x

    n = len(wrappers)
    for i, layer in enumerate(wrappers):
        r = reuse_per_layer[i]

        anchor_to_head = next(
            f for f in bpmn.get_flows()
            if f.get_source() is anchor and f.get_target() is current_head
        )
        tail_to_v = next(
            f for f in bpmn.get_flows()
            if f.get_source() is current_tail and f.get_target() is v
        )
        process = anchor_to_head.get_process() or tail_to_v.get_process()

        new_split = BPMN.ExclusiveGateway(
            id=(r.gw_a_id if r else ""), name="", process=process,
        )
        new_join = BPMN.ExclusiveGateway(
            id=(r.gw_b_id if r else ""), name="", process=process,
        )
        bpmn.add_node(new_split)
        bpmn.add_node(new_join)

        bpmn.remove_flow(anchor_to_head)
        bpmn.add_flow(BPMN.SequenceFlow(
            source=anchor, target=new_split, process=process,
        ))
        bpmn.add_flow(BPMN.SequenceFlow(
            source=new_split, target=current_head,
            id=(r.wrap_in_flow_id if r else ""),
            process=process,
        ))

        bpmn.remove_flow(tail_to_v)
        prev_forward_id = (
            reuse_per_layer[i - 1].forward_flow_id
            if i > 0 and reuse_per_layer[i - 1] is not None
            else ""
        )
        bpmn.add_flow(BPMN.SequenceFlow(
            source=current_tail, target=new_join,
            id=prev_forward_id, process=process,
        ))
        boundary_forward_id = (
            r.forward_flow_id if (i == n - 1 and r) else ""
        )
        bpmn.add_flow(BPMN.SequenceFlow(
            source=new_join, target=v,
            id=boundary_forward_id, process=process,
        ))

        if layer.had_skip:
            bpmn.add_flow(BPMN.SequenceFlow(
                source=new_split, target=new_join,
                id=(r.skip_flow_id if (r and r.skip_flow_id) else ""),
                process=process,
            ))
        if layer.had_loop:
            bpmn.add_flow(BPMN.SequenceFlow(
                source=new_join, target=new_split,
                id=(r.loop_flow_id if (r and r.loop_flow_id) else ""),
                process=process,
            ))

        current_head = new_split
        current_tail = new_join

def insert_activity_before_with_nested_xor_wraps(
    bpmn: BPMN,
    activity_name: str,
    anchor_name: str,
    wrappers: list[WrapperContext],
    wrapper_ids: list[WrapperLayerIds] | None = None,
    task_id: str | None = None,
) -> None:
    """Mirror of :func:`insert_activity_after_with_nested_xor_wraps`:"""

    insert_activity_before(bpmn, activity_name, anchor_name, task_id=task_id)

    if not wrappers:
        return

    anchor_matches = [n for n in bpmn.get_nodes() if n.get_name() == anchor_name]
    if len(anchor_matches) != 1:
        raise ValueError(
            f"Anchor {anchor_name!r} no longer unique after plain insert"
        )
    anchor = anchor_matches[0]

    a_preds_after = _predecessors(bpmn, anchor)
    if len(a_preds_after) != 1:
        raise ValueError(
            f"Anchor {anchor_name!r} unexpectedly has "
            f"{len(a_preds_after)} predecessors after plain insert"
        )
    new_x = a_preds_after[0]
    x_preds = _predecessors(bpmn, new_x)
    if len(x_preds) != 1:
        raise ValueError(
            f"Inserted activity {activity_name!r} unexpectedly has "
            f"{len(x_preds)} predecessors after plain insert"
        )
    pred = x_preds[0]

    reuse_per_layer: list[WrapperLayerIds | None] = []
    if wrapper_ids is not None:
        existing_node_ids = {str(n.get_id()) for n in bpmn.get_nodes()}
        for i in range(len(wrappers)):
            cand = wrapper_ids[i] if i < len(wrapper_ids) else None
            if cand is None:
                reuse_per_layer.append(None)
                continue
            if cand.gw_a_id in existing_node_ids or cand.gw_b_id in existing_node_ids:
                reuse_per_layer.append(None)
            else:
                reuse_per_layer.append(cand)
    else:
        reuse_per_layer = [None] * len(wrappers)

    current_head = new_x
    current_tail = new_x

    n = len(wrappers)
    for i, layer in enumerate(wrappers):
        r = reuse_per_layer[i]

        pred_to_head = next(
            f for f in bpmn.get_flows()
            if f.get_source() is pred and f.get_target() is current_head
        )
        tail_to_anchor = next(
            f for f in bpmn.get_flows()
            if f.get_source() is current_tail and f.get_target() is anchor
        )
        process = pred_to_head.get_process() or tail_to_anchor.get_process()

        new_split = BPMN.ExclusiveGateway(
            id=(r.gw_a_id if r else ""), name="", process=process,
        )
        new_join = BPMN.ExclusiveGateway(
            id=(r.gw_b_id if r else ""), name="", process=process,
        )
        bpmn.add_node(new_split)
        bpmn.add_node(new_join)

        bpmn.remove_flow(pred_to_head)
        bpmn.add_flow(BPMN.SequenceFlow(
            source=pred, target=new_split, process=process,
        ))
        bpmn.add_flow(BPMN.SequenceFlow(
            source=new_split, target=current_head,
            id=(r.wrap_in_flow_id if r else ""),
            process=process,
        ))

        bpmn.remove_flow(tail_to_anchor)
        prev_forward_id = (
            reuse_per_layer[i - 1].forward_flow_id
            if i > 0 and reuse_per_layer[i - 1] is not None
            else ""
        )
        bpmn.add_flow(BPMN.SequenceFlow(
            source=current_tail, target=new_join,
            id=prev_forward_id, process=process,
        ))
        boundary_forward_id = (
            r.forward_flow_id if (i == n - 1 and r) else ""
        )
        bpmn.add_flow(BPMN.SequenceFlow(
            source=new_join, target=anchor,
            id=boundary_forward_id, process=process,
        ))

        if layer.had_skip:
            bpmn.add_flow(BPMN.SequenceFlow(
                source=new_split, target=new_join,
                id=(r.skip_flow_id if (r and r.skip_flow_id) else ""),
                process=process,
            ))
        if layer.had_loop:
            bpmn.add_flow(BPMN.SequenceFlow(
                source=new_join, target=new_split,
                id=(r.loop_flow_id if (r and r.loop_flow_id) else ""),
                process=process,
            ))

        current_head = new_split
        current_tail = new_join

def lazy_reconcile_gateway_probabilities(
    params: dict,
    bpmn_after: BPMN,
) -> dict:
    """Reconcile ``gateway_branching_probabilities`` against *bpmn_after*."""

    node_ids = {str(n.get_id()) for n in bpmn_after.get_nodes()}
    flow_ids = {str(f.get_id()) for f in bpmn_after.get_flows()}

    outgoing_by_source: dict = {}
    for f in bpmn_after.get_flows():
        outgoing_by_source.setdefault(str(f.get_source().get_id()), set()).add(str(f.get_id()))

    gbp = params.get("gateway_branching_probabilities", [])

    surviving: list[dict] = []
    gateways_dropped = 0
    paths_dropped = 0
    paths_renormalized = 0
    paths_remapped = 0
    ambiguous_gateways: list = []

    for entry in gbp:
        gid = str(entry.get("gateway_id"))
        if gid not in node_ids:
            gateways_dropped += 1
            continue

        outgoing = outgoing_by_source.get(gid, set())
        kept_probs: list[dict] = []
        orphaned: list[dict] = []
        for p in entry.get("probabilities", []):
            if str(p.get("path_id")) in flow_ids and str(p.get("path_id")) in outgoing:
                kept_probs.append(p)
            else:
                orphaned.append(p)

        unmatched = sorted(outgoing - {str(p.get("path_id")) for p in kept_probs})
        if len(orphaned) == 1 and len(unmatched) == 1:
            orphaned[0]["path_id"] = unmatched[0]
            kept_probs.append(orphaned[0])
            paths_remapped += 1
        else:
            if orphaned and unmatched:

                ambiguous_gateways.append(gid)
            paths_dropped += len(orphaned)

        if not kept_probs:
            gateways_dropped += 1
            continue

        total = sum(p.get("value", 0.0) for p in kept_probs)
        if total > 0 and abs(total - 1.0) > 1e-9:
            for p in kept_probs:
                p["value"] = p["value"] / total
            paths_renormalized += len(kept_probs)

        entry["probabilities"] = kept_probs
        surviving.append(entry)

    params["gateway_branching_probabilities"] = surviving

    decision_gateway_ids = {
        str(n.get_id()) for n in bpmn_after.get_nodes()
        if isinstance(n, (BPMN.ExclusiveGateway, BPMN.InclusiveGateway))
    }
    covered = {str(e["gateway_id"]) for e in surviving}
    uncovered_decision_gateways = sorted(decision_gateway_ids - covered)

    return {
        "gateways_dropped": gateways_dropped,
        "paths_dropped": paths_dropped,
        "paths_remapped": paths_remapped,
        "paths_renormalized": paths_renormalized,
        "ambiguous_rewired_gateways": ambiguous_gateways,
        "uncovered_decision_gateways": uncovered_decision_gateways,
    }
