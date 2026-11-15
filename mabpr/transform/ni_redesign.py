"""NUMERICAL INVOLVEMENT — the Phase-3 handler. Params-only, the widening one."""

from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class InvolvementResult:
    candidate_id: str
    target_activities: list = field(default_factory=list)
    task_ids: dict = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""

def apply_numerical_involvement(candidate: dict, df, bpmn) -> InvolvementResult:
    """Resolve the run to BPMN task ids. The BPMN is NOT mutated."""
    cid = candidate.get("candidate_id", "")
    frag = candidate.get("process_fragment") or {}
    acts = list(frag.get("target_activities")
                or candidate.get("target_activities")
                or (candidate.get("evidence") or {}).get("target_activities") or [])
    if len(acts) < 2:
        return InvolvementResult(cid, acts, skipped=True,
                                 skip_reason=("Numerical Involvement needs at least "
                                              "two activities to put in one pair of "
                                              "hands"))
    if bpmn is None:
        return InvolvementResult(cid, acts, skipped=True,
                                 skip_reason="no model to resolve the run against")
    ids, gone = {}, []
    for a in acts:
        m = [n for n in bpmn.get_nodes() if n.get_name() == a]
        if len(m) == 1:
            ids[a] = str(m[0].get_id())
        else:
            gone.append(a)
    if len(ids) < 2:
        return InvolvementResult(
            cid, acts, ids, skipped=True,
            skip_reason=(f"fewer than two of the run's activities are still in the "
                         f"model (missing or ambiguous: {gone}) — an earlier "
                         f"candidate removed them, so there is no stretch left to "
                         f"consolidate"))
    return InvolvementResult(cid, acts, ids)

def adapt_params_after_ni(params: dict, result: InvolvementResult) -> dict:
    """Give every resource in the run's union the right to perform all of it."""
    if result.skipped:
        return {"applied": False, "reason": result.skip_reason}

    entries = {str(e.get("task_id")): e
               for e in params.get("task_resource_distribution") or []}
    mine = {name: entries[tid] for name, tid in result.task_ids.items()
            if tid in entries}
    if len(mine) < 2:
        return {"applied": False,
                "reason": (f"only {len(mine)} of the run's activities have a "
                           f"task_resource_distribution entry; there is no allocation "
                           f"to widen")}

    union, added = set(), 0
    for e in mine.values():
        union |= {str(r.get("resource_id")) for r in e.get("resources") or []}
    for name, e in mine.items():
        have = {str(r.get("resource_id")) for r in e.get("resources") or []}

        pool = e.get("resources") or []
        if not pool:
            continue
        template = max(pool, key=lambda r: sum(
            1 for x in pool
            if x.get("distribution_name") == r.get("distribution_name")))
        for rid in sorted(union - have):
            clone = {k: v for k, v in template.items()}
            clone["resource_id"] = rid
            e["resources"].append(clone)
            added += 1

    assigned = 0
    tids = set(result.task_ids.values())
    for prof in params.get("resource_profiles") or []:
        for r in prof.get("resource_list") or []:
            if str(r.get("id")) not in union:
                continue
            cur = list(r.get("assignedTasks") or [])
            new = [t for t in tids if t not in cur]
            if new:
                r["assignedTasks"] = cur + new
                assigned += len(new)

    if not added and not assigned:
        return {"applied": False,
                "reason": (f"every resource in the run already performs all of "
                           f"{sorted(mine)} — the case is not passed between "
                           f"different pools here, so widening changes nothing")}
    return {
        "applied": True,
        "target_activities": sorted(mine),
        "pool_union": len(union),

        "task_resources_added": added,
        "assigned_tasks_added": assigned,
        "ca_pool_resource_count": len(union),
    }
