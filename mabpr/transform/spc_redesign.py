"""SPECIALIZE — the Phase-3 handler. Params-only, the mirror image of XR."""

from __future__ import annotations

from dataclasses import dataclass, field

from mabpr.detectors.extra_resources_detector import normalize_resource_id

@dataclass
class SpecializeResult:
    """Per-candidate outcome of Phase-3 SPC."""
    candidate_id: str
    target_activity: str = ""
    target_task_id: str = ""
    specialists: list = field(default_factory=list)
    sizing: dict = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""

def apply_specialize(candidate: dict, df, bpmn) -> SpecializeResult:
    """Resolve the target activity to its BPMN task id. The BPMN is not mutated."""

    cid = candidate["candidate_id"]
    name = candidate.get("target_activity") or ""
    ev = candidate.get("evidence") or {}
    specialists = list(ev.get("specialists") or [])

    if not name:
        return SpecializeResult(cid, skipped=True,
                                skip_reason="SPC needs a target_activity")
    if len(specialists) < 2:
        return SpecializeResult(
            cid, target_activity=name, skipped=True,
            skip_reason=(f"SPC needs at least 2 named specialists; got "
                         f"{len(specialists)} — a pool of one is a single point "
                         f"of failure, not a specialisation"))

    matches = [n for n in bpmn.get_nodes() if n.get_name() == name]
    if len(matches) == 0:
        return SpecializeResult(
            cid, target_activity=name, skipped=True,
            skip_reason=(f"target activity {name!r} is no longer in the model — an "
                         f"earlier candidate removed it, so there is no allocation "
                         f"left to specialise"))
    if len(matches) > 1:
        return SpecializeResult(
            cid, target_activity=name, skipped=True,
            skip_reason=(f"target activity {name!r} resolves to {len(matches)} BPMN "
                         f"nodes; the allocation to narrow is ambiguous"))

    return SpecializeResult(
        candidate_id=cid, target_activity=name,
        target_task_id=str(matches[0].get_id()), specialists=specialists,
        sizing={"specialist_count": ev.get("specialist_count"),
                "utilisation_after": ev.get("utilisation_after"),
                "speedup": ev.get("speedup"),
                "separation_z": ev.get("separation_z")})

def adapt_params_after_spc(params: dict, result: SpecializeResult) -> dict:
    """Narrow the activity's allocation to the named specialists."""
    if result.skipped or not result.target_task_id:
        return {"applied": False, "reason": result.skip_reason}

    tid = result.target_task_id
    wanted = {normalize_resource_id(r) for r in result.specialists}

    entry = next((e for e in params.get("task_resource_distribution", [])
                  if str(e.get("task_id")) == tid), None)
    if entry is None:
        return {"applied": False,
                "reason": f"no task_resource_distribution entry for task {tid} "
                          f"({result.target_activity!r})"}

    before = list(entry.get("resources", []))
    keep = [r for r in before
            if normalize_resource_id(r.get("resource_id")) in wanted]
    found = {normalize_resource_id(r.get("resource_id")) for r in keep}
    missing = sorted(wanted - found)

    if not keep:

        return {"applied": False,
                "reason": (f"none of the {len(wanted)} named specialists appear in "
                           f"the params for {result.target_activity!r} "
                           f"(looked for {sorted(wanted)}); refusing to leave the "
                           f"activity with no resource able to perform it"),
                "specialists_missing": missing}

    entry["resources"] = keep
    removed = len(before) - len(keep)

    unassigned = 0
    for prof in params.get("resource_profiles", []):
        for res in prof.get("resource_list", []):
            if normalize_resource_id(res.get("id")) in wanted:
                continue
            tasks = list(res.get("assignedTasks", []))
            if tid in tasks:
                res["assignedTasks"] = [t for t in tasks if t != tid]
                unassigned += 1

    return {
        "applied": True,
        "target_activity": result.target_activity,
        "target_task_id": tid,
        "resources_before": len(before),
        "resources_after": len(keep),
        "resources_removed": removed,
        "assigned_tasks_removed": unassigned,

        "specialists_missing": missing,
        "sizing": result.sizing,
    }
