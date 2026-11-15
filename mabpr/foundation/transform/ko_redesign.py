"""Knock-out — heuristic-level redesign module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from pm4py.objects.bpmn.obj import BPMN

from mabpr.foundation.transform.bpmn_primitives import (
    WrapperLayerIds,
    delete_activity,
    detect_xor_wrapper_cascade,
    detect_xor_wrapper_cascade_with_ids,
    insert_activity_after_with_nested_xor_wraps,
    insert_activity_before_with_nested_xor_wraps,
    lazy_reconcile_gateway_probabilities,
    past_timers,
)

@dataclass
class MoveSummary:
    """One re-anchor performed during KO reordering."""

    activity:         str
    anchor:           str
    placement:        str
    n_wrappers:       int
    case_label:       str
    old_wrapper_ids:  list[WrapperLayerIds] = field(default_factory=list)

@dataclass
class KnockoutResult:
    """Per-candidate outcome of Phase 3 KO."""
    candidate_id:      str
    target_activities: list[str]
    optimal_ordering:  list[str]
    moves:             list[MoveSummary] = field(default_factory=list)
    skipped:           bool = False
    skip_reason:       str  = ""

def rewrite_bpmn(
    bpmn: BPMN,
    optimal_ordering: list[str],
    direction: str = "forward",
) -> tuple[list[MoveSummary], str]:
    """Reorder KO checks in *bpmn* so they appear in *optimal_ordering*"""

    if direction not in ("forward", "reverse"):
        return [], (
            f"unknown direction {direction!r}; expected 'forward' or 'reverse'"
        )
    if len(optimal_ordering) < 2:
        return [], "optimal_ordering has fewer than 2 checks; nothing to reorder"

    moves: list[MoveSummary] = []

    if direction == "forward":
        pair_indices = range(1, len(optimal_ordering))
    else:
        pair_indices = range(len(optimal_ordering) - 1, 0, -1)

    for i in pair_indices:
        prev = optimal_ordering[i - 1]
        curr = optimal_ordering[i]

        if direction == "forward":
            moving, anchor_name, placement = curr, prev, "after"
        else:
            moving, anchor_name, placement = prev, curr, "before"

        moving_matches = [n for n in bpmn.get_nodes() if n.get_name() == moving]
        anchor_matches = [n for n in bpmn.get_nodes() if n.get_name() == anchor_name]
        if not moving_matches:
            return moves, f"check {moving!r} not in BPMN"
        if not anchor_matches:
            return moves, f"check {anchor_name!r} not in BPMN"
        if len(moving_matches) > 1:
            return moves, f"check {moving!r} not unique in BPMN"
        if len(anchor_matches) > 1:
            return moves, f"check {anchor_name!r} not unique in BPMN"

        moving_node = moving_matches[0]
        anchor_node = anchor_matches[0]

        if placement == "after":
            side = past_timers(bpmn, anchor_node, forward=True)
            adj_flows = [f for f in bpmn.get_flows() if f.get_source() is side]
            already = (
                len(adj_flows) == 1 and adj_flows[0].get_target() is moving_node
            )
            adj_label, adj_count = "outgoing", len(adj_flows)
        else:
            side = past_timers(bpmn, anchor_node, forward=False)
            adj_flows = [f for f in bpmn.get_flows() if f.get_target() is side]
            already = (
                len(adj_flows) == 1 and adj_flows[0].get_source() is moving_node
            )
            adj_label, adj_count = "incoming", len(adj_flows)

        if already:
            moves.append(MoveSummary(
                activity=moving,
                anchor=anchor_name,
                placement=placement,
                n_wrappers=0,
                case_label="already_in_place",
            ))
            continue

        if adj_count != 1:
            return moves, (
                f"check {anchor_name!r} has {adj_count} {adj_label} flows; "
                "insertion point not unique"
            )

        wrappers = detect_xor_wrapper_cascade(bpmn, moving)
        old_wrapper_ids = detect_xor_wrapper_cascade_with_ids(bpmn, moving)

        old_task_matches = [n for n in bpmn.get_nodes() if n.get_name() == moving]
        old_task_id = str(old_task_matches[0].get_id()) if len(old_task_matches) == 1 else None

        delete_activity(bpmn, moving)
        try:
            if placement == "after":
                insert_activity_after_with_nested_xor_wraps(
                    bpmn, moving, anchor_name, wrappers,
                    wrapper_ids=old_wrapper_ids,
                    task_id=old_task_id,
                )
            else:
                insert_activity_before_with_nested_xor_wraps(
                    bpmn, moving, anchor_name, wrappers,
                    wrapper_ids=old_wrapper_ids,
                    task_id=old_task_id,
                )
        except ValueError as e:
            return moves, (
                f"move {moving!r} {placement} {anchor_name!r} failed: {e}"
            )

        moves.append(MoveSummary(
            activity=moving,
            anchor=anchor_name,
            placement=placement,
            n_wrappers=len(wrappers),
            case_label="moved",
            old_wrapper_ids=old_wrapper_ids,
        ))

    return moves, ""

def apply_knockout(
    candidate: dict,
    df: pd.DataFrame,
    bpmn: BPMN,
    direction: str = "forward",
) -> KnockoutResult:
    """Apply Phase 3 Knock-out to a single detector candidate."""

    candidate_id      = candidate["candidate_id"]
    target_activities = list(candidate.get("target_activities", []))
    evidence          = candidate.get("evidence", {})
    optimal_ordering  = list(evidence.get("optimal_ordering", []))
    current_ordering  = list(evidence.get("current_ordering", []))

    if current_ordering and current_ordering == optimal_ordering:
        return KnockoutResult(
            candidate_id=candidate_id,
            target_activities=target_activities,
            optimal_ordering=optimal_ordering,
            moves=[],
            skipped=True,
            skip_reason=(
                "current ordering already matches optimal ordering "
                "(efficiency_gap = 0); no reordering needed"
            ),
        )

    if len(optimal_ordering) < 2:
        return KnockoutResult(
            candidate_id=candidate_id,
            target_activities=target_activities,
            optimal_ordering=optimal_ordering,
            moves=[],
            skipped=True,
            skip_reason=(
                "optimal_ordering has fewer than 2 checks; "
                "nothing to reorder"
            ),
        )

    moves, reason = rewrite_bpmn(bpmn, optimal_ordering, direction=direction)
    if reason:
        return KnockoutResult(
            candidate_id=candidate_id,
            target_activities=target_activities,
            optimal_ordering=optimal_ordering,
            moves=moves,
            skipped=True,
            skip_reason=f"rewriter skipped: {reason}",
        )

    return KnockoutResult(
        candidate_id=candidate_id,
        target_activities=target_activities,
        optimal_ordering=optimal_ordering,
        moves=moves,
        skipped=False,
    )

def result_to_json(result: KnockoutResult) -> dict:
    """Convert a single KnockoutResult to a plain dict for JSON export."""
    return {
        "candidate_id":      result.candidate_id,
        "target_activities": result.target_activities,
        "optimal_ordering":  result.optimal_ordering,
        "skipped":           result.skipped,
        "skip_reason":       result.skip_reason,
        "moves":             [asdict(m) for m in result.moves],
    }

def save_result_json(result: KnockoutResult, path: str | Path) -> None:
    """Persist a single KnockoutResult to *path* as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_to_json(result), f, indent=2, ensure_ascii=False)
    print(f"Saved Phase 3 KO result to {path}")

def adapt_params_after_ko(
    params: dict,
    result: KnockoutResult,
    bpmn_after: BPMN,
) -> dict:
    """Sync a Prosimos params JSON dict to reflect a KO reordering."""

    if result.skipped:
        return {"applied": False, "reason": result.skip_reason}

    gw_report = lazy_reconcile_gateway_probabilities(params, bpmn_after)
    return {"applied": True, **gw_report}
