"""ACTIVITY COMPOSITION — the Phase-3 handler."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mabpr.foundation.transform.bpmn_primitives import (
    delete_activity, drop_event_distributions, lazy_reconcile_gateway_probabilities,
    strip_preceding_timers)

_MEAN_AT = {"fix": 0, "fixed": 0, "expon": 0, "exponential": 0,
            "norm": 0, "normal": 0, "gamma": 0, "lognorm": 0, "lognormal": 0}
_STD_AT = {"norm": 1, "normal": 1}

@dataclass
class ActivityCompositionResult:
    """Per-candidate outcome of Phase-3 AC."""
    candidate_id: str
    target_activities: list = field(default_factory=list)
    composite_name: str = ""
    kept_node_id: str = ""
    merged_node_id: str = ""
    timer_ids: list = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

def _sum_distributions(da: dict, db: dict) -> dict:
    """Moment-sum two duration distributions. See the module docstring."""
    na = str(da.get("distribution_name", "fix")).lower()
    nb = str(db.get("distribution_name", "fix")).lower()
    pa = [float(p.get("value", 0.0)) for p in da.get("distribution_params") or []]
    pb = [float(p.get("value", 0.0)) for p in db.get("distribution_params") or []]

    def mean_of(name, p):
        i = _MEAN_AT.get(name)
        return p[i] if (i is not None and len(p) > i) else 0.0

    if na != nb:
        keep, kp = (da, pa) if mean_of(na, pa) >= mean_of(nb, pb) else (db, pb)
        name = str(keep.get("distribution_name", "fix")).lower()
        out = [dict(p) for p in keep.get("distribution_params") or []]
        i = _MEAN_AT.get(name)
        if i is not None and len(out) > i:
            out[i] = {"value": mean_of(na, pa) + mean_of(nb, pb)}
        return {"distribution_name": keep.get("distribution_name", "fix"),
                "distribution_params": out, "mixed_families": [na, nb]}

    out = []
    for k in range(max(len(pa), len(pb))):
        va = pa[k] if k < len(pa) else 0.0
        vb = pb[k] if k < len(pb) else 0.0
        if _STD_AT.get(na) == k:
            out.append({"value": math.sqrt(va * va + vb * vb)})
        else:
            out.append({"value": va + vb})
    return {"distribution_name": da.get("distribution_name", "fix"),
            "distribution_params": out}

def apply_activity_composition(candidate: dict, df, bpmn) -> ActivityCompositionResult:
    """Merge B into A in the BPMN. MUTATES *bpmn*."""

    cid = candidate["candidate_id"]
    targets = list(candidate.get("target_activities") or [])
    if len(targets) != 2:
        return ActivityCompositionResult(
            cid, target_activities=targets, skipped=True,
            skip_reason=f"AC merges exactly 2 consecutive activities; got {len(targets)}")
    a_name, b_name = targets[0], targets[1]
    composite = candidate.get("composite_name") or f"{a_name} + {b_name}"

    def find(name):
        return [n for n in bpmn.get_nodes() if n.get_name() == name]

    na, nb = find(a_name), find(b_name)
    for name, hits in ((a_name, na), (b_name, nb)):
        if len(hits) == 0:
            return ActivityCompositionResult(
                cid, target_activities=targets, skipped=True,
                skip_reason=(f"{name!r} is no longer in the model — an earlier "
                             f"candidate removed it, so there is no pair to merge"))
        if len(hits) > 1:
            return ActivityCompositionResult(
                cid, target_activities=targets, skipped=True,
                skip_reason=f"{name!r} resolves to {len(hits)} nodes; the merge is ambiguous")

    a_node, b_node = na[0], nb[0]
    a_id, b_id = str(a_node.get_id()), str(b_node.get_id())

    adjacent = any(str(f.get_source().get_id()) == a_id
                   and str(f.get_target().get_id()) == b_id
                   for f in bpmn.get_flows())

    timers = [] if adjacent else _timer_chain(bpmn, a_id, b_id)
    if timers:
        removed = strip_preceding_timers(bpmn, b_node)
        adjacent = any(str(f.get_source().get_id()) == a_id
                       and str(f.get_target().get_id()) == b_id
                       for f in bpmn.get_flows())
        if adjacent:
            timers = removed
        else:

            timers = []

    if not adjacent:

        between = _nodes_between(bpmn, a_id, b_id)
        what = ", ".join(between) if between else "no path at all"
        return ActivityCompositionResult(
            cid, target_activities=targets, skipped=True,
            skip_reason=(f"{a_name!r} does not flow directly into {b_name!r}; "
                         f"what sits between them is: {what}"))

    try:
        delete_activity(bpmn, b_name)
    except ValueError as e:
        return ActivityCompositionResult(
            cid, target_activities=targets, skipped=True,
            skip_reason=f"could not remove {b_name!r} to fold it into the composite: {e}")

    a_node.set_name(composite)

    return ActivityCompositionResult(
        candidate_id=cid, target_activities=targets, composite_name=composite,
        kept_node_id=a_id, merged_node_id=b_id, timer_ids=list(timers))

def _timer_chain(bpmn, a_id: str, b_id: str, limit: int = 4) -> list:
    """The timers on a chain ``A -> t1 -> ... -> tn -> B``, or [] if that is not"""

    out, inc = {}, {}
    for f in bpmn.get_flows():
        out.setdefault(str(f.get_source().get_id()), []).append(str(f.get_target().get_id()))
        inc.setdefault(str(f.get_target().get_id()), []).append(str(f.get_source().get_id()))
    by_id = {str(n.get_id()): n for n in bpmn.get_nodes()}

    chain, cur = [], a_id
    for _ in range(limit):
        nxt = out.get(cur, [])
        if len(nxt) != 1:
            return []
        cur = nxt[0]
        if cur == b_id:
            return chain if chain else []
        node = by_id.get(cur)
        if node is None or "IntermediateCatchEvent" not in type(node).__name__:
            return []
        if len(inc.get(cur, [])) != 1 or len(out.get(cur, [])) != 1:
            return []
        chain.append(cur)
    return []

def _nodes_between(bpmn, a_id: str, b_id: str, max_depth: int = 4) -> list:
    """Describe the nodes on the shortest path from *a_id* to *b_id*, if short."""

    from pm4py.objects.bpmn.obj import BPMN

    succ: dict = {}
    for f in bpmn.get_flows():
        succ.setdefault(str(f.get_source().get_id()), []).append(f.get_target())

    def describe(node) -> str:
        if isinstance(node, BPMN.Task):
            return f"the task {node.get_name()!r}"
        if isinstance(node, BPMN.IntermediateCatchEvent):
            return ("an intermediate catch event (a fitted extraneous-delay "
                    "timer, whose duration lives in the params, not here)")
        return f"a {type(node).__name__}"

    frontier = [(a_id, [])]
    seen = {a_id}
    for _ in range(max_depth):
        nxt = []
        for nid, path in frontier:
            for node in succ.get(nid, []):
                tid = str(node.get_id())
                if tid == b_id:
                    return path or ["nothing — they are adjacent"]
                if tid in seen:
                    continue
                seen.add(tid)
                nxt.append((tid, path + [describe(node)]))
        frontier = nxt
    return []

def adapt_params_after_ac(params: dict, result: ActivityCompositionResult,
                          bpmn_after) -> dict:
    """Fold B's service times into A, then drop everything that referenced B."""
    if result.skipped or not result.merged_node_id:
        return {"applied": False, "reason": result.skip_reason}

    a_id, b_id = result.kept_node_id, result.merged_node_id
    trd = params.get("task_resource_distribution", [])
    a_entry = next((e for e in trd if str(e.get("task_id")) == a_id), None)
    b_entry = next((e for e in trd if str(e.get("task_id")) == b_id), None)

    folded, added = 0, 0
    if a_entry is not None and b_entry is not None:
        b_by_res = {str(r.get("resource_id")): r for r in b_entry.get("resources", [])}
        for r in a_entry.get("resources", []):
            rid = str(r.get("resource_id"))
            if rid in b_by_res:
                merged = _sum_distributions(r, b_by_res[rid])
                r["distribution_name"] = merged["distribution_name"]
                r["distribution_params"] = merged["distribution_params"]
                folded += 1

        present = {str(r.get("resource_id")) for r in a_entry.get("resources", [])}
        for rid, r in b_by_res.items():
            if rid not in present:
                a_entry["resources"].append(dict(r))
                added += 1

    before = len(trd)
    params["task_resource_distribution"] = [
        e for e in trd if str(e.get("task_id")) != b_id]
    trd_dropped = before - len(params["task_resource_distribution"])

    assigned_removed, assigned_added = 0, 0
    for prof in params.get("resource_profiles", []):
        for res in prof.get("resource_list", []):
            tasks = list(res.get("assignedTasks", []))
            had_b = b_id in tasks
            tasks = [t for t in tasks if t != b_id]
            assigned_removed += int(had_b)

            if had_b and a_id not in tasks:
                tasks.append(a_id)
                assigned_added += 1
            res["assignedTasks"] = tasks

    ea = params.get("event_attributes", [])
    before = len(ea)
    params["event_attributes"] = [e for e in ea if str(e.get("event_id")) != b_id]
    ea_dropped = before - len(params["event_attributes"])

    n_timers, timer_h = drop_event_distributions(params, result.timer_ids or [])

    gw = lazy_reconcile_gateway_probabilities(params, bpmn_after)

    return {
        "applied": True,
        "composite_name": result.composite_name,

        "timers_removed": n_timers,
        "timer_h_removed_per_occurrence": timer_h,
        "kept_node_id": a_id,
        "merged_node_id": b_id,
        "distributions_folded": folded,
        "distributions_carried_over": added,
        "task_resource_distribution_dropped": trd_dropped,
        "assigned_tasks_removed": assigned_removed,
        "assigned_tasks_added": assigned_added,
        "event_attributes_dropped": ea_dropped,

        "setup_saving_assumed_h": 0.0,
        "gateway_reconciliation": gw,
    }
