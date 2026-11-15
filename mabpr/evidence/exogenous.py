"""EXOGENOUS SAVING — the part of a candidate's benefit the simulator is blind to."""

from __future__ import annotations

RECIPES: dict[str, tuple] = {
    "Activity Elimination": (
        "waiting_time_before",
        "the wait before the deleted activity goes with the activity"),
    "Case-based Work": (
        "batch_interval_mean",
        "processing case by case removes the wait for the batch to fill"),

    "Contact Reduction": (
        "waiting_time_before",
        "consolidating the exchange removes the idle wait after every repeat "
        "contact beyond the first"),
    "Interfacing": (
        "waiting_time_before",
        "a standardised interface removes the idle wait after the submissions "
        "that currently come back and force earlier work to be redone"),
}

_ABSORBING_FIELDS = ("timers_removed", "batch_component_h_removed")

def absorbed_by_edit(edit: dict | None) -> str | None:
    """Why this candidate's exogenous claim is NOT claimable after the edit that ran."""

    for field in _ABSORBING_FIELDS:
        if (edit or {}).get(field):
            return (f"the edit removed or shrank the extraneous-delay timer that holds "
                    f"this wait ({field}={(edit or {})[field]}), so the simulation "
                    f"measured it and the four columns contain it. Reporting the log "
                    f"estimate as well would count the same hours twice.")
    return None

def _dig(evidence: dict, field: str):
    """The field, or the same field one level down (detectors nest some blocks)."""
    if field in evidence:
        return evidence[field]
    for v in evidence.values():
        if isinstance(v, dict) and field in v:
            return v[field]
    return None

def exogenous_saving(candidate: dict, occurrences_per_case: float | None = None,
                     baseline_case_hours: float | None = None,
                     edit: dict | None = None) -> dict:
    """{hours, field, why, simulator_blind} for one candidate, or {} when its"""

    heuristic = candidate.get("heuristic") or ""
    recipe = RECIPES.get(heuristic)
    if recipe is None:
        return {}
    if absorbed_by_edit(edit):
        return {}
    field, why = recipe
    hours = _dig(candidate.get("evidence") or {}, field)
    if not isinstance(hours, (int, float)) or hours <= 0:
        return {}
    out = {
        "hours_per_occurrence": round(float(hours), 4),
        "field": field,
        "why": why,
        "source": "detector evidence (measured in the LOG, not simulated)",
        "note": ("The simulator cannot see this: the mined model has no overnight, "
                 "handover or batching delay. It is an ESTIMATE of what the "
                 "redesign additionally removes, and it is NOT comparable with the "
                 "simulated columns — never add it into a ranking. Its own "
                 "reliability is untested: injecting it into the baseline and "
                 "re-simulating returns the amount injected."),
    }

    if occurrences_per_case:
        out["occurrences_per_case"] = round(float(occurrences_per_case), 4)
        est = float(hours) * float(occurrences_per_case)
        out["hours_per_case_estimate"] = round(est, 4)
        if baseline_case_hours and est > baseline_case_hours:

            out["EXCEEDS_WHOLE_CASE"] = (
                f"{est:.1f} h/case estimated against a {baseline_case_hours:.2f} h "
                f"baseline case. The detector measures wall-clock gaps in the real "
                f"log (overnight, weekend, backlog), so this is an upper bound on "
                f"what the redesign could touch, NOT a saving the case can realise. "
                f"Use it as evidence that the effect exists and the simulator is "
                f"blind to it; never rank candidates by its size.")
    return out
