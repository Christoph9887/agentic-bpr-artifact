"""Resequencing — heuristic-level redesign module."""

from __future__ import annotations

import json
from collections import defaultdict
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
    lazy_reconcile_gateway_probabilities,
    past_timers,
)

SUPPORT_MIN: float = 0.10

POSITION_MARGIN: float = 0.10

HIGH_CONF_SUPPORT: float = 0.30
HIGH_CONF_MARGIN:  float = 0.20

@dataclass
class PlacementProposal:
    """Output of the Empirical Placement Miner."""

    predecessor:               str
    support_fraction:          float
    occurrence_count:          int
    empirical_position_ratio:  float
    current_position_ratio:    float
    confidence:                str

@dataclass
class RewriteSummary:
    """Output of the Structural Rewriter on success."""

    applied:             bool
    target_flow_source:  str
    target_flow_target:  str
    case_label:          str
    old_wrapper_ids:     list[WrapperLayerIds] = field(default_factory=list)

@dataclass
class ResequencingResult:
    """Per-candidate outcome of Phase 3."""
    candidate_id:  str
    activity:      str
    placement:     Optional[PlacementProposal]
    rewrite:       Optional[RewriteSummary]
    skipped:       bool
    skip_reason:   str = ""

def mine_empirical_placement(
    activity: str,
    df: pd.DataFrame,
    current_position_ratio: float,
    support_min:     float = SUPPORT_MIN,
    position_margin: float = POSITION_MARGIN,
) -> Optional[PlacementProposal]:
    """Single-anchor miner: aggregate per-case observations of "X directly"""

    df_sorted = df.sort_values(["case:concept:name", "time:timestamp"])

    per_A_cases:  dict[str, set] = defaultdict(set)
    per_A_ratios: dict[str, list[float]] = defaultdict(list)
    total_containing_cases = 0

    for case_id, group in df_sorted.groupby("case:concept:name", sort=False):
        acts = group["concept:name"].tolist()
        n = len(acts)
        if n <= 1:
            continue
        if activity not in acts:
            continue

        total_containing_cases += 1

        for i, act in enumerate(acts):
            if act != activity or i == 0:
                continue
            A = acts[i - 1]
            ratio = i / (n - 1)
            per_A_cases[A].add(case_id)
            per_A_ratios[A].append(ratio)

    if total_containing_cases == 0 or not per_A_ratios:
        return None

    survivors: list[tuple[str, float, int, float]] = []
    for A, ratios in per_A_ratios.items():
        support    = len(per_A_cases[A]) / total_containing_cases
        mean_ratio = sum(ratios) / len(ratios)
        if support < support_min:
            continue
        if mean_ratio < current_position_ratio + position_margin:
            continue
        survivors.append((A, support, len(ratios), mean_ratio))

    if not survivors:
        return None

    survivors.sort(key=lambda x: (x[3], x[1]), reverse=True)
    A, support, count, mean_ratio = survivors[0]

    confidence = (
        "high"
        if support >= HIGH_CONF_SUPPORT
        and (mean_ratio - current_position_ratio) >= HIGH_CONF_MARGIN
        else "medium"
    )

    return PlacementProposal(
        predecessor=A,
        support_fraction=round(support, 4),
        occurrence_count=count,
        empirical_position_ratio=round(mean_ratio, 4),
        current_position_ratio=round(current_position_ratio, 4),
        confidence=confidence,
    )

def rewrite_bpmn(
    bpmn: BPMN,
    activity: str,
    placement: PlacementProposal,
) -> tuple[Optional[RewriteSummary], str]:
    """Apply Resequencing on *bpmn* (MUTATES the object) so that"""

    activity_nodes = [n for n in bpmn.get_nodes() if n.get_name() == activity]
    if not activity_nodes:
        return None, f"activity {activity!r} not in BPMN"

    anchor_name = placement.predecessor
    anchor_matches = [n for n in bpmn.get_nodes() if n.get_name() == anchor_name]
    if not anchor_matches:
        return None, f"predecessor {anchor_name!r} not in BPMN"
    if len(anchor_matches) > 1:
        return None, (
            f"predecessor {anchor_name!r} has {len(anchor_matches)} matching "
            "nodes; not unique"
        )
    anchor = anchor_matches[0]

    out_flows = [f for f in bpmn.get_flows() if f.get_source() is anchor]
    if len(out_flows) == 0:
        return None, f"predecessor {anchor_name!r} has no outgoing flow"
    if len(out_flows) > 1:
        return None, (
            f"predecessor {anchor_name!r} has {len(out_flows)} outgoing flows; "
            "insertion point not unique"
        )

    _skipped = past_timers(bpmn, anchor, forward=True)
    _fl = [f for f in bpmn.get_flows() if f.get_source() is _skipped]
    _after_anchor = _fl[0].get_target() if len(_fl) == 1 else None
    if out_flows[0].get_target() in activity_nodes or _after_anchor in activity_nodes:
        return None, (
            f"activity is already placed immediately after "
            f"{anchor_name!r} in the current BPMN"
        )

    v = out_flows[0].get_target()
    v_label = (v.get_name()
               or (_after_anchor.get_name() if _after_anchor is not None else "")
               or f"<{type(v).__name__}>")

    wrappers = detect_xor_wrapper_cascade(bpmn, activity)
    old_wrapper_ids = detect_xor_wrapper_cascade_with_ids(bpmn, activity)

    old_task_matches = [n for n in bpmn.get_nodes() if n.get_name() == activity]
    old_task_id = str(old_task_matches[0].get_id()) if len(old_task_matches) == 1 else None

    delete_activity(bpmn, activity)
    try:
        insert_activity_after_with_nested_xor_wraps(
            bpmn, activity, anchor_name, wrappers,
            wrapper_ids=old_wrapper_ids,
            task_id=old_task_id,
        )
    except ValueError as e:

        return None, f"insertion failed after delete: {e}"

    return (
        RewriteSummary(
            applied=True,
            target_flow_source=anchor_name,
            target_flow_target=v_label,
            case_label="after_predecessor",
            old_wrapper_ids=old_wrapper_ids,
        ),
        "",
    )

def apply_resequencing(
    candidate: dict,
    df: pd.DataFrame,
    bpmn: BPMN,
) -> ResequencingResult:
    """Apply Phase 3 Resequencing to a single detector candidate."""

    activity      = candidate["target_activity"]
    candidate_id  = candidate["candidate_id"]
    current_ratio = candidate["evidence"]["activity_position_ratio"]

    placement = mine_empirical_placement(
        activity=activity,
        df=df,
        current_position_ratio=current_ratio,
    )
    if placement is None:
        return ResequencingResult(
            candidate_id=candidate_id,
            activity=activity,
            placement=None,
            rewrite=None,
            skipped=True,
            skip_reason="no empirical placement with sufficient support",
        )

    rewrite, reason = rewrite_bpmn(bpmn, activity, placement)
    if rewrite is None:
        return ResequencingResult(
            candidate_id=candidate_id,
            activity=activity,
            placement=placement,
            rewrite=None,
            skipped=True,
            skip_reason=f"rewriter skipped: {reason}",
        )

    return ResequencingResult(
        candidate_id=candidate_id,
        activity=activity,
        placement=placement,
        rewrite=rewrite,
        skipped=False,
    )

def result_to_json(result: ResequencingResult) -> dict:
    """Convert a single ResequencingResult to a plain dict for JSON export."""
    return {
        "candidate_id": result.candidate_id,
        "activity":     result.activity,
        "skipped":      result.skipped,
        "skip_reason":  result.skip_reason,
        "placement":    asdict(result.placement) if result.placement else None,
        "rewrite":      asdict(result.rewrite)   if result.rewrite   else None,
    }

def save_result_json(result: ResequencingResult, path: str | Path) -> None:
    """Persist a single ResequencingResult to *path* as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_to_json(result), f, indent=2, ensure_ascii=False)
    print(f"Saved Phase 3 RSQ result to {path}")

def adapt_params_after_rsq(
    params: dict,
    result: ResequencingResult,
    bpmn_after: BPMN,
) -> dict:
    """Sync a Prosimos params JSON dict to reflect a Resequencing move."""

    if result.skipped:
        return {"applied": False, "reason": result.skip_reason}

    gw_report = lazy_reconcile_gateway_probabilities(params, bpmn_after)
    return {"applied": True, **gw_report}
