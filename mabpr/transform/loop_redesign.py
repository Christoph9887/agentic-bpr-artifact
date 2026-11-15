"""CONTACT REDUCTION + INTERFACING — the Phase-3 handler. Params-only, shared."""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_LOOP_P = 0.001
_MARK = "_mabpr_loop_edited_by"

@dataclass
class LoopResult:
    """Per-candidate outcome of Phase-3 CR / IF."""
    candidate_id: str
    heuristic: str = ""
    target_activity: str = ""
    gateway_id: str = ""
    loop_path_id: str = ""
    exit_path_id: str = ""
    loop_probability_after: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

def apply_loop_reduction(candidate: dict, df, bpmn) -> LoopResult:
    """Resolve the candidate against the model. The BPMN is NOT mutated."""

    cid = candidate.get("candidate_id", "")
    heur = candidate.get("heuristic", "")
    frag = candidate.get("process_fragment") or {}
    ev = candidate.get("evidence") or {}
    act = frag.get("target_activity") or candidate.get("target_activity") or ""
    gid = frag.get("gateway_id") or ev.get("gateway_id") or ""
    loop_path = frag.get("loop_path_id") or ""
    exit_path = frag.get("exit_path_id") or ""
    after = frag.get("loop_probability_after")
    if after is None:
        after = ev.get("loop_probability_after")

    if not (gid and loop_path and exit_path):
        return LoopResult(cid, heur, act, skipped=True,
                          skip_reason=("the candidate carries no gateway/path ids, so "
                                       "there is no loop probability to reduce"))
    if not isinstance(after, (int, float)):
        return LoopResult(cid, heur, act, gid, skipped=True,
                          skip_reason="the candidate carries no target probability")
    if bpmn is not None:
        present = {n.get_name() for n in bpmn.get_nodes() if n.get_name()}
        if act and act not in present:
            return LoopResult(
                cid, heur, act, gid, skipped=True,
                skip_reason=(f"{act!r} is no longer in the model — an earlier "
                             f"candidate removed it, so its rework loop no longer "
                             f"exists"))
    return LoopResult(cid, heur, act, gid, loop_path, exit_path, float(after))

def adapt_params_after_loop(params: dict, result: LoopResult) -> dict:
    """Lower the loop path's probability and give the difference to the exit."""
    if result.skipped:
        return {"applied": False, "reason": result.skip_reason}

    g = next((x for x in params.get("gateway_branching_probabilities") or []
              if str(x.get("gateway_id")) == result.gateway_id), None)
    if g is None:
        return {"applied": False,
                "reason": (f"gateway {result.gateway_id} is not in the params; the "
                           f"loop probability this candidate measured does not exist "
                           f"in what is being simulated")}
    owner = g.get(_MARK)
    if owner and owner != result.candidate_id:
        return {"applied": False,
                "reason": (f"gateway {result.gateway_id} was already re-rated by "
                           f"{owner} in this plan. Contact Reduction and Interfacing "
                           f"move the SAME probability, so applying both would leave "
                           f"whichever ran last and the plan would express one "
                           f"candidate while claiming two."),
                "collides_with": owner}

    paths = {str(p.get("path_id")): p for p in g.get("probabilities") or []}
    lp, ep = paths.get(result.loop_path_id), paths.get(result.exit_path_id)
    if lp is None or ep is None:
        return {"applied": False,
                "reason": (f"gateway {result.gateway_id} no longer has both the loop "
                           f"and exit paths this candidate named")}

    before = float(lp.get("value", 0.0))
    after = max(MIN_LOOP_P, min(before, float(result.loop_probability_after)))
    if before - after < 1e-6:
        return {"applied": False,
                "reason": (f"the loop is already at {before:.4f}; the measured target "
                           f"{result.loop_probability_after:.4f} would not lower it, "
                           f"so this candidate is a no-op on this model")}

    total = sum(float(p.get("value", 0.0)) for p in g.get("probabilities") or [])
    lp["value"] = after
    ep["value"] = float(ep.get("value", 0.0)) + (before - after)
    g[_MARK] = result.candidate_id
    return {
        "applied": True,
        "heuristic": result.heuristic,
        "target_activity": result.target_activity,
        "gateway_id": result.gateway_id,
        "loop_probability_before": round(before, 4),
        "loop_probability_after": round(after, 4),

        "loop_probability_reduced": round(before - after, 4),
        "exit_probability_after": round(float(ep["value"]), 4),
        "paths_renormalized": len(g.get("probabilities") or []),
        "sum_before": round(total, 6),
        "sum_after": round(sum(float(p.get("value", 0.0))
                               for p in g.get("probabilities") or []), 6),
    }
