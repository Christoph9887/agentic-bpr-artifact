"""TRIAGE — a Phase-3 handler that REFUSES, on purpose, and says exactly why."""

from __future__ import annotations

from dataclasses import dataclass

_REASON = (
    "Triage needs the per-type versions of an activity to be separate BPMN nodes "
    "behind a conditioned split — a Prosimos task holds exactly one service "
    "distribution, so there is no parameter that can carry 'this case type needs "
    "more effort here'. That transformation is not built, because the detector "
    "finds no candidate on either dataset (every activity x case-type effort gap "
    "is inside the noise, |z| < 3). This candidate is therefore refused rather "
    "than half-applied, and the refusal is recorded so the loop stops re-proposing "
    "it. See mabpr/transform/triage_redesign.py for what a real handler needs."
)

@dataclass
class TriageResult:
    candidate_id: str
    target_activity: str = ""
    skipped: bool = True
    skip_reason: str = _REASON

def apply_triage(candidate: dict, df, bpmn) -> TriageResult:
    """Always a refusal, with the reason above. *df* / *bpmn* are unused."""
    return TriageResult(candidate.get("candidate_id", ""),
                        candidate.get("target_activity", ""))

def adapt_params_after_triage(params: dict, result: TriageResult) -> dict:
    """No parameter is touched; the refusal is passed through unchanged."""
    return {"applied": False, "reason": result.skip_reason,
            "target_activity": result.target_activity,
            "not_implementable_as_params": True}
