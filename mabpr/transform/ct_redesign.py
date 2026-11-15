"""CASE TYPES — the Phase-3 handler. Params-only; the BPMN is never touched."""

from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class CaseTypesResult:
    """Per-candidate outcome of Phase-3 CT."""
    candidate_id: str
    attribute: str = ""
    routes: list = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

def apply_case_types(candidate: dict, df, bpmn) -> CaseTypesResult:
    """Resolve the candidate against the model. The BPMN is NOT mutated."""

    cid = candidate.get("candidate_id", "")
    frag = candidate.get("process_fragment") or {}
    ev = candidate.get("evidence") or {}
    attribute = frag.get("attribute") or ev.get("attribute") or ""
    routes = list(frag.get("routes") or ev.get("routes") or [])

    if not attribute:
        return CaseTypesResult(cid, skipped=True,
                               skip_reason="CT needs a case attribute to route on")
    if not routes:
        return CaseTypesResult(
            cid, attribute=attribute, skipped=True,
            skip_reason=("CT needs at least one gateway to condition; the "
                         "candidate carries none"))

    if bpmn is not None:
        present = {n.get_name() for n in bpmn.get_nodes() if n.get_name()}
        alive = [r for r in routes if r.get("target_activity") in present]
        gone = sorted({r.get("target_activity") for r in routes
                       if r.get("target_activity") not in present})
        if not alive:
            return CaseTypesResult(
                cid, attribute=attribute, skipped=True,
                skip_reason=(f"every activity this case type routes to is gone from "
                             f"the model ({gone}) — an earlier candidate removed "
                             f"them, so there is no routing left to condition"))
        routes = alive
    return CaseTypesResult(candidate_id=cid, attribute=attribute, routes=routes)

def adapt_params_after_ct(params: dict, result: CaseTypesResult) -> dict:
    """Attach the case type's rule, and its complement, to every claimed gateway."""
    if result.skipped:
        return {"applied": False, "reason": result.skip_reason}

    declared = {str(c.get("name")) for c in (params.get("case_attributes") or [])}
    if result.attribute not in declared:
        return {"applied": False,
                "reason": (f"{result.attribute!r} is not in params.case_attributes "
                           f"(have {sorted(declared)}); Prosimos would never "
                           f"generate it, so the rule could not fire and the "
                           f"candidate would read as applied-with-no-effect")}

    by_gateway = {str(g.get("gateway_id")): g
                  for g in params.get("gateway_branching_probabilities") or []}
    rules = params.setdefault("branch_rules", [])
    conditioned, missing = [], []

    for i, r in enumerate(result.routes):
        gid = str(r.get("gateway_id"))
        g = by_gateway.get(gid)
        if g is None:
            missing.append(gid)
            continue
        paths = {str(p.get("path_id")): p for p in g.get("probabilities") or []}
        p_in = paths.get(str(r.get("conditioned_path_id")))
        p_out = paths.get(str(r.get("complement_path_id")))
        if p_in is None or p_out is None:
            missing.append(gid)
            continue

        comp = str(r.get("comparison"))
        inv = _INVERT.get(comp)
        if inv is None:
            missing.append(gid)
            continue
        in_id = f"{result.candidate_id}-{i}-in"
        out_id = f"{result.candidate_id}-{i}-out"
        rules.append({"id": in_id, "rules": [[
            {"attribute": result.attribute, "comparison": comp,
             "value": str(r.get("value"))}]]})
        rules.append({"id": out_id, "rules": [[
            {"attribute": result.attribute, "comparison": inv,
             "value": str(r.get("value"))}]]})
        p_in["condition_id"] = in_id
        p_out["condition_id"] = out_id
        conditioned.append({"gateway_id": gid,
                            "target_activity": r.get("target_activity"),
                            "rule": f"{result.attribute} {comp} {r.get('value')}",
                            "pooled_probability_replaced": r.get("pooled_probability")})

    if not conditioned:
        return {"applied": False,
                "reason": (f"none of the {len(result.routes)} claimed gateways could "
                           f"be conditioned (missing from params or their paths no "
                           f"longer match): {missing}")}
    return {
        "applied": True,
        "attribute": result.attribute,
        "gateways_conditioned": len(conditioned),
        "activities_routed": sorted({c["target_activity"] for c in conditioned
                                     if c["target_activity"]}),
        "routes": conditioned,

        "gateways_missing": missing,
        "branch_rules_added": 2 * len(conditioned),
    }

_INVERT = {"<=": ">", ">": "<=", "<": ">=", ">=": "<", "=": "!=", "!=": "="}
