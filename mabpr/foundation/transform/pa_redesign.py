"""Parallelism — heuristic-level redesign module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from pm4py.objects.bpmn.obj import BPMN

from mabpr.foundation.transform.bpmn_primitives import (
    lazy_reconcile_gateway_probabilities,
    parallelize_activities,
)

@dataclass
class RewriteSummary:
    """Output of the Structural Rewriter on success."""
    applied:     bool
    activity_a:  str
    activity_b:  str
    case_label:  str

@dataclass
class ParallelismResult:
    """Per-candidate outcome of Phase 3."""
    candidate_id:  str
    activity_a:    str
    activity_b:    str
    rewrite:       Optional[RewriteSummary]
    skipped:       bool
    skip_reason:   str = ""

def rewrite_bpmn(
    bpmn: BPMN,
    activity_a: str,
    activity_b: str,
) -> tuple[Optional[RewriteSummary], str]:
    """Apply Parallelism on *bpmn* (MUTATES the object): wrap the"""

    try:
        parallelize_activities(bpmn, activity_a, activity_b)
    except ValueError as e:
        return None, str(e)

    return (
        RewriteSummary(
            applied=True,
            activity_a=activity_a,
            activity_b=activity_b,
            case_label="parallelized",
        ),
        "",
    )

def apply_parallelism(
    candidate: dict,
    df: pd.DataFrame,
    bpmn: BPMN,
) -> ParallelismResult:
    """Apply Phase 3 Parallelism to a single detector candidate."""

    candidate_id = candidate["candidate_id"]
    target = candidate["target_activities"]
    activity_a = target["activity_a"]
    activity_b = target["activity_b"]

    rewrite, reason = rewrite_bpmn(bpmn, activity_a, activity_b)
    if rewrite is None:
        return ParallelismResult(
            candidate_id=candidate_id,
            activity_a=activity_a,
            activity_b=activity_b,
            rewrite=None,
            skipped=True,
            skip_reason=f"rewriter skipped: {reason}",
        )

    return ParallelismResult(
        candidate_id=candidate_id,
        activity_a=activity_a,
        activity_b=activity_b,
        rewrite=rewrite,
        skipped=False,
    )

def result_to_json(result: ParallelismResult) -> dict:
    """Convert a single ParallelismResult to a plain dict for JSON export."""
    return {
        "candidate_id": result.candidate_id,
        "activity_a":   result.activity_a,
        "activity_b":   result.activity_b,
        "skipped":      result.skipped,
        "skip_reason":  result.skip_reason,
        "rewrite":      asdict(result.rewrite) if result.rewrite else None,
    }

def save_result_json(result: ParallelismResult, path: str | Path) -> None:
    """Persist a single ParallelismResult to *path* as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_to_json(result), f, indent=2, ensure_ascii=False)
    print(f"Saved Phase 3 PAR result to {path}")

def adapt_params_after_par(
    params: dict,
    result: ParallelismResult,
    bpmn_after: BPMN,
) -> dict:
    """Sync a Prosimos params JSON dict to reflect a Parallelism rewrite."""

    if result.skipped:
        return {"applied": False, "reason": result.skip_reason}

    gw_report = lazy_reconcile_gateway_probabilities(params, bpmn_after)
    return {"applied": True, **gw_report}
