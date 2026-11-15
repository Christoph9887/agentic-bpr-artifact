"""Activity Elimination — heuristic-level redesign module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from pm4py.objects.bpmn.obj import BPMN

from mabpr.foundation.transform.bpmn_primitives import (
    delete_activity,
    drop_event_distributions,
    lazy_reconcile_gateway_probabilities,
    strip_preceding_timers,
)

@dataclass
class RewriteSummary:
    """Output of the Structural Rewriter on success."""
    applied:          bool
    activity:         str
    deleted_node_id:  str
    case_label:       str

    removed_timer_ids: tuple = ()

@dataclass
class ActivityEliminationResult:
    """Per-candidate outcome of Phase 3."""
    candidate_id:  str
    activity:      str
    rewrite:       Optional[RewriteSummary]
    skipped:       bool
    skip_reason:   str = ""

def rewrite_bpmn(
    bpmn: BPMN,
    activity: str,
) -> tuple[Optional[RewriteSummary], str]:
    """Apply Activity Elimination on *bpmn* (MUTATES the object): remove"""

    target = next(
        (n for n in bpmn.get_nodes() if n.get_name() == activity),
        None,
    )
    if target is None:
        return None, f"no node named {activity!r}"
    deleted_node_id = str(target.get_id())

    removed_timers = strip_preceding_timers(bpmn, target)

    try:
        delete_activity(bpmn, activity)
    except ValueError as e:
        return None, str(e)

    return (
        RewriteSummary(
            applied=True,
            activity=activity,
            deleted_node_id=deleted_node_id,
            case_label="eliminated",
            removed_timer_ids=tuple(removed_timers),
        ),
        "",
    )

def apply_activity_elimination(
    candidate: dict,
    df: pd.DataFrame,
    bpmn: BPMN,
) -> ActivityEliminationResult:
    """Apply Phase 3 Activity Elimination to a single detector candidate."""

    candidate_id = candidate["candidate_id"]
    activity = candidate["target_activity"]

    rewrite, reason = rewrite_bpmn(bpmn, activity)
    if rewrite is None:
        return ActivityEliminationResult(
            candidate_id=candidate_id,
            activity=activity,
            rewrite=None,
            skipped=True,
            skip_reason=f"rewriter skipped: {reason}",
        )

    return ActivityEliminationResult(
        candidate_id=candidate_id,
        activity=activity,
        rewrite=rewrite,
        skipped=False,
    )

def result_to_json(result: ActivityEliminationResult) -> dict:
    """Convert a single ActivityEliminationResult to a plain dict for JSON export."""
    return {
        "candidate_id": result.candidate_id,
        "activity":     result.activity,
        "skipped":      result.skipped,
        "skip_reason":  result.skip_reason,
        "rewrite":      asdict(result.rewrite) if result.rewrite else None,
    }

def save_result_json(result: ActivityEliminationResult, path: str | Path) -> None:
    """Persist a single ActivityEliminationResult to *path* as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_to_json(result), f, indent=2, ensure_ascii=False)
    print(f"Saved Phase 3 AE result to {path}")

def adapt_params_after_ae(
    params: dict,
    result: ActivityEliminationResult,
    bpmn_after: BPMN,
) -> dict:
    """Sync a Prosimos params JSON dict to reflect an Activity Elimination."""

    if result.skipped or result.rewrite is None:
        return {"applied": False, "reason": "result is skipped or has no rewrite"}

    deleted_id = result.rewrite.deleted_node_id

    trd = params.get("task_resource_distribution", [])
    before = len(trd)
    params["task_resource_distribution"] = [
        e for e in trd if e.get("task_id") != deleted_id
    ]
    trd_dropped = before - len(params["task_resource_distribution"])

    assigned_removed = 0
    for prof in params.get("resource_profiles", []):
        for res in prof.get("resource_list", []):
            tasks = res.get("assignedTasks", [])
            before = len(tasks)
            res["assignedTasks"] = [t for t in tasks if t != deleted_id]
            assigned_removed += before - len(res["assignedTasks"])

    ea = params.get("event_attributes", [])
    before = len(ea)
    params["event_attributes"] = [
        e for e in ea if e.get("event_id") != deleted_id
    ]
    ea_dropped = before - len(params["event_attributes"])

    timer_ids = getattr(result.rewrite, "removed_timer_ids", ()) or ()
    n_timers, timer_h = drop_event_distributions(params, timer_ids)

    bp = params.get("batch_processing") or []
    kept = [r for r in bp if str(r.get("task_id")) != deleted_id]
    batch_rules_dropped = len(bp) - len(kept)
    if batch_rules_dropped:
        params["batch_processing"] = kept

    gw_report = lazy_reconcile_gateway_probabilities(params, bpmn_after)

    return {
        "applied": True,
        "deleted_node_id": deleted_id,
        "task_resource_distribution_dropped": trd_dropped,
        "assigned_tasks_removed": assigned_removed,
        "event_attributes_dropped": ea_dropped,
        "timers_removed": n_timers,
        "batch_rules_dropped": batch_rules_dropped,
        "timer_h_removed_per_occurrence": timer_h,
        **gw_report,
    }
