"""Case-based Work — heuristic-level redesign module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from pm4py.objects.bpmn.obj import BPMN

from mabpr.evidence.batch_delay import scale_distribution

@dataclass
class CaseBasedWorkResult:
    """Per-candidate outcome of Phase 3 CBW."""

    candidate_id:        str
    target_activity:     str
    target_task_id:      Optional[str]   = None
    batch_size_mean:     Optional[float] = None
    batch_interval_mean: Optional[float] = None
    skipped:             bool            = False
    skip_reason:         str             = ""

    removed_timer_ids:   tuple           = ()

    delay_split:         Optional[dict]  = None

def apply_case_based_work(
    candidate:  dict,
    df:         pd.DataFrame,
    bpmn:       BPMN,
) -> CaseBasedWorkResult:
    """Resolve target_activity NAME to UUID via the BPMN. CBW does not"""

    candidate_id        = candidate["candidate_id"]
    target_activity     = candidate.get("target_activity")
    evidence            = candidate.get("evidence", {}) or {}
    batch_size_mean     = evidence.get("batch_size_mean")
    batch_interval_mean = evidence.get("batch_interval_mean")

    if not target_activity:
        return CaseBasedWorkResult(
            candidate_id=candidate_id,
            target_activity=str(target_activity),
            skipped=True,
            skip_reason="candidate has no target_activity",
        )
    if batch_size_mean is None:
        return CaseBasedWorkResult(
            candidate_id=candidate_id,
            target_activity=target_activity,
            skipped=True,
            skip_reason=(
                "candidate evidence is missing batch_size_mean; "
                "cannot synthesize a Prosimos batching rule"
            ),
        )

    matches = [n for n in bpmn.get_nodes() if n.get_name() == target_activity]
    if len(matches) == 0:
        return CaseBasedWorkResult(
            candidate_id=candidate_id,
            target_activity=target_activity,
            batch_size_mean=batch_size_mean,
            skipped=True,
            skip_reason=f"target activity {target_activity!r} not found in BPMN",
        )
    if len(matches) > 1:
        return CaseBasedWorkResult(
            candidate_id=candidate_id,
            target_activity=target_activity,
            batch_size_mean=batch_size_mean,
            skipped=True,
            skip_reason=(
                f"target activity {target_activity!r} has {len(matches)} matches "
                "in BPMN; not unique"
            ),
        )

    return CaseBasedWorkResult(
        candidate_id=candidate_id,
        target_activity=target_activity,
        target_task_id=str(matches[0].get_id()),
        batch_size_mean=float(batch_size_mean),
        batch_interval_mean=(float(batch_interval_mean)
                             if batch_interval_mean is not None else None),
        skipped=False,
        delay_split=(evidence.get("extraneous_delay_split") or None),
    )

def result_to_json(result: CaseBasedWorkResult) -> dict:
    """Convert a CaseBasedWorkResult to a plain dict for JSON export."""
    return asdict(result)

def save_result_json(result: CaseBasedWorkResult, path: str | Path) -> None:
    """Persist a single CaseBasedWorkResult to *path* as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_to_json(result), f, indent=2, ensure_ascii=False)
    print(f"Saved Phase 3 CBW result to {path}")

def adapt_params_after_cbw(
    params: dict,
    result: CaseBasedWorkResult,
    bpmn,
) -> dict:
    """Remove the part of the wait in front of the target that batching causes —"""

    if result.skipped or not result.target_task_id:
        return {"applied": False, "reason": result.skip_reason}

    tid = str(result.target_task_id)

    split = result.delay_split or {}
    component = float(split.get("batch_component_h") or 0.0)
    timer_h = float(split.get("timer_h") or 0.0)
    if component > 0 and timer_h > 0:
        factor = max(0.0, (timer_h - component) / timer_h)
        event_id = str(split.get("event_id"))
        ed = params.get("event_distribution") or []
        for idx, e in enumerate(ed):
            if str(e.get("event_id")) == event_id:
                ed[idx] = scale_distribution(e, factor)
                params["event_distribution"] = ed
                return {
                    "applied":        True,
                    "channel":        "extraneous_delay_batch_share",
                    "task_id":        tid,
                    "rules_removed":  0,
                    "timers_removed": 0,
                    "timer_h_before": round(timer_h, 4),
                    "timer_h_after":  round(timer_h - component, 4),
                    "batch_component_h_removed": round(component, 4),
                    "share_of_timer_removed": round(component / timer_h, 4),
                    "delta_h_batched_vs_not": split.get("delta_h"),
                    "delta_z": split.get("z"),
                    "note": (
                        f"scaled the extraneous-delay timer in front of "
                        f"{result.target_activity!r} by {factor:.4f}: "
                        f"{timer_h:.2f} h -> {timer_h - component:.2f} h per "
                        f"occurrence. Only the batching share is removed "
                        f"({100 * component / timer_h:.1f}% of the timer, from a "
                        f"{split.get('delta_h')} h gap between batched and "
                        f"non-batched executions at z={split.get('z')}); the rest "
                        f"of the wait is left in place because this heuristic does "
                        f"not claim it."),
                }

    bp = params.get("batch_processing") or []
    kept = [r for r in bp if str(r.get("task_id")) != tid]
    rules_removed = len(bp) - len(kept)
    if rules_removed:
        params["batch_processing"] = kept
        return {
            "applied":        True,
            "channel":        "batching_rule",
            "rules_removed":  rules_removed,
            "task_id":        tid,
            "timers_removed": 0,
            "note": (f"removed {rules_removed} batch_processing rule(s) for "
                     f"{result.target_activity!r}. Exact: the extraneous-delay "
                     "timer is left in place, so the measured difference is the "
                     "batching alone."),
        }

    why = (split.get("reason")
           or f"its split attributes {split.get('batch_component_h')} h to batching"
           ) if split else (
        "no batching split is attached to this candidate — run the detectors so "
        "`extraneous_delay_split` is computed for its target")
    return {
        "applied": False,
        "channel": "refused_no_batch_component",
        "reason": (f"nothing in this model is attributable to batching in front of "
                   f"{result.target_activity!r}: {why}. The extraneous-delay timer "
                   f"is deliberately NOT removed — it holds every unexplained wait "
                   f"before the activity, of which batching is only a part."),
    }
