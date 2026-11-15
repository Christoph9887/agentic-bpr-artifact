"""EXTRA RESOURCES — the Phase-3 handler. Params-only, one field."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

TARGET_UTILISATION = 0.80

MAX_EXTRA_PER_RESOURCE = 3

@dataclass
class ExtraResourcesResult:
    """Per-candidate outcome of Phase-3 XR."""

    candidate_id: str
    target_activity: str = ""
    target_task_id: str = ""
    extra_per_resource: int = 0
    sizing: dict = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""

def size_extra_resources(evidence: dict, *, target_utilisation: float = TARGET_UTILISATION
                         ) -> dict:
    """How many extra servers the queue needs, and the arithmetic behind it."""

    arrival = float(evidence.get("arrival_rate_per_h") or 0.0)
    service_h = float(evidence.get("service_time_mean") or 0.0)
    servers = int(evidence.get("resources_serving") or 0)
    saturation = float(evidence.get("calendar_utilisation") or 0.0)
    if arrival <= 0 or service_h <= 0 or servers <= 0:
        return {"extra_per_resource": 0,
                "why": "arrival rate, service time and pool size must all be "
                       "measured and positive to size a queue"}

    if saturation > 0:
        needed = math.ceil(servers * saturation / target_utilisation)
        load = servers * saturation
        basis = (f"{servers} servers running at {saturation:.2f} of the hours they "
                 f"are open; reaching {target_utilisation:.0%} needs "
                 f"{servers} x {saturation:.2f} / {target_utilisation:.2f} = "
                 f"{needed} servers")
    else:

        load = arrival * service_h
        needed = math.ceil(load / target_utilisation)
        basis = (f"no resource calendar available, so sized on WALL-CLOCK load: "
                 f"arrivals {arrival:.3f}/h x service {service_h:.4f} h = "
                 f"{load:.3f} erlangs, needing {needed} servers at "
                 f"{target_utilisation:.0%} — this under-provisions whenever the "
                 f"pool is not staffed around the clock")
    extra_total = max(0, needed - servers)

    extra_per_resource = min(MAX_EXTRA_PER_RESOURCE,
                             math.ceil(extra_total / servers)) if extra_total else 0
    return {
        "extra_per_resource": int(extra_per_resource),
        "offered_load_erlangs": round(load, 4),
        "sized_on": "calendar" if saturation > 0 else "wall_clock",
        "servers_now": servers,
        "servers_needed": int(needed),
        "target_utilisation": target_utilisation,
        "current_utilisation_implied": round(load / servers, 4),
        "why": (f"{basis}, against the {servers} in the pool now, so each "
                f"resource entry's `amount` goes up by {extra_per_resource}"),
    }

def apply_extra_resources(candidate: dict, df, bpmn) -> ExtraResourcesResult:
    """Resolve the target activity NAME to its BPMN task id and size the addition."""

    cid = candidate["candidate_id"]
    name = candidate.get("target_activity") or ""
    if not name:
        return ExtraResourcesResult(cid, skipped=True,
                                    skip_reason="XR needs a target_activity")

    matches = [n for n in bpmn.get_nodes() if n.get_name() == name]
    if len(matches) == 0:

        return ExtraResourcesResult(
            cid, target_activity=name, skipped=True,
            skip_reason=(f"target activity {name!r} is no longer in the model — an "
                         f"earlier candidate removed it, so there is no queue left "
                         f"to add capacity to"))
    if len(matches) > 1:
        return ExtraResourcesResult(
            cid, target_activity=name, skipped=True,
            skip_reason=(f"target activity {name!r} resolves to {len(matches)} BPMN "
                         f"nodes; the pool to grow is ambiguous"))

    sizing = size_extra_resources(candidate.get("evidence") or {})
    if not sizing.get("extra_per_resource"):
        return ExtraResourcesResult(
            cid, target_activity=name, target_task_id=str(matches[0].get_id()),
            sizing=sizing, skipped=True,
            skip_reason=(f"the measured load needs no extra capacity: "
                         f"{sizing.get('why', 'sizing produced 0')}"))

    return ExtraResourcesResult(
        candidate_id=cid, target_activity=name,
        target_task_id=str(matches[0].get_id()),
        extra_per_resource=int(sizing["extra_per_resource"]), sizing=sizing)

def adapt_params_after_xr(params: dict, result: ExtraResourcesResult) -> dict:
    """Grow the pool serving the target task. Mutates *params*, returns a report."""

    if result.skipped or not result.target_task_id:
        return {"applied": False, "reason": result.skip_reason}

    tid = result.target_task_id

    serving: set = set()
    for entry in params.get("task_resource_distribution", []):
        if str(entry.get("task_id")) != tid:
            continue
        for r in entry.get("resources", []):
            rid = str(r.get("resource_id"))
            if rid and rid != "NOT_SET":
                serving.add(rid)

    grown, before, after = 0, 0, 0
    for prof in params.get("resource_profiles", []):
        for res in prof.get("resource_list", []):
            if str(res.get("id")) not in serving:
                continue
            was = int(res.get("amount", 1) or 1)
            res["amount"] = was + result.extra_per_resource
            before += was
            after += res["amount"]
            grown += 1

    if not grown:

        return {"applied": False,
                "reason": (f"no resource profile entry serves task {tid} "
                           f"({result.target_activity!r}), so there is no pool to grow")}

    return {
        "applied": True,
        "target_activity": result.target_activity,
        "target_task_id": tid,
        "resources_grown": grown,
        "extra_per_resource": result.extra_per_resource,
        "pool_size_before": before,
        "pool_size_after": after,

        "sizing": result.sizing,
    }
