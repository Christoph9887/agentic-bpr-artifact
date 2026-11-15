"""Exception — heuristic-level redesign module."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from pm4py.objects.bpmn.obj import BPMN

from mabpr.foundation.transform.bpmn_primitives import (
    encapsulate_chain_as_subprocess,
    lazy_reconcile_gateway_probabilities,
)

MAX_HOPS_TO_SPLIT: int = 5

MAX_BRANCH_LENGTH: int = 50

DEFAULT_SUBPROCESS_NAME: str = "Exception Handler"

@dataclass
class ExceptionLocation:
    """Result of :func:`locate_exception_branch`."""

    divergence_node_id:                str
    divergence_node_name:              str
    split_gateway_id:                  str
    join_gateway_id:                   str
    exception_branch_node_ids:         list[str]
    exception_branch_activity_names:   list[str]
    dominant_branch_node_ids:          list[str]
    dominant_branch_activity_names:    list[str]
    other_branch_count:                int

@dataclass
class EncapsulationSummary:
    """Result of :func:`encapsulate_exception_branch` on success."""
    applied:                       bool
    subprocess_node_id:            str
    subprocess_name:               str
    encapsulated_node_count:       int
    encapsulated_activity_names:   list[str]

@dataclass
class ExceptionResult:
    """Per-candidate outcome of Phase 3 EXC."""
    candidate_id:    str
    divergence_point: str
    location:        Optional[ExceptionLocation]   = None
    encapsulation:   Optional[EncapsulationSummary] = None
    skipped:         bool                          = False
    skip_reason:     str                           = ""

def _find_node_by_name(bpmn: BPMN, name: str):
    matches = [n for n in bpmn.get_nodes() if n.get_name() == name]
    if len(matches) != 1:
        return None
    return matches[0]

def _successors(bpmn: BPMN, node):
    return [f.get_target() for f in bpmn.get_flows() if f.get_source() is node]

def _predecessors(bpmn: BPMN, node):
    return [f.get_source() for f in bpmn.get_flows() if f.get_target() is node]

def _walk_to_first_split(
    bpmn: BPMN,
    start_node,
    max_hops: int,
):
    """Forward-walk from *start_node*, stepping through 1-out nodes until"""

    current = start_node
    for _ in range(max_hops + 1):
        outs = _successors(bpmn, current)
        if (
            isinstance(current, BPMN.ExclusiveGateway)
            and len(outs) >= 2
            and current is not start_node
        ):
            return current
        if len(outs) != 1:
            return None
        current = outs[0]
    return None

def _describe_blocker(bpmn: BPMN, branch_start, max_length: int = 12) -> str:
    """Name the node that stops the branch walk, for an honest refusal message."""

    current = branch_start
    seen: set = set()
    for _ in range(max_length):
        nid = current.get_id()
        if nid in seen:
            return "a node it had already visited (a loop)"
        seen.add(nid)
        outs = _successors(bpmn, current)
        if len(outs) == 0:
            return f"an end of the flow at {type(current).__name__}"
        if len(outs) > 1:
            kinds = sorted({type(n).__name__ for n in _successors(bpmn, current)})
            return (f"a {type(current).__name__} with {len(outs)} outgoing flows "
                    f"leading to {kinds}")
        current = outs[0]
    return f"no join within {max_length} hops"

def _collect_branch_until_join(
    bpmn:           BPMN,
    branch_start,
    split_node,
    candidate_join_ids: Optional[set[str]] = None,
    max_length:     int = MAX_BRANCH_LENGTH,
):
    """Walk forward from *branch_start*, gathering nodes that belong to"""

    branch: list = []
    visited_ids: set[str] = set()
    current = branch_start

    for _ in range(max_length):
        nid = current.get_id()
        if nid in visited_ids:
            return None, None
        visited_ids.add(nid)

        if (
            isinstance(current, BPMN.ExclusiveGateway)
            and len(_predecessors(bpmn, current)) >= 2
            and current is not split_node
        ):
            return [n_id for n_id in branch], current

        if candidate_join_ids is not None and nid in candidate_join_ids:
            return [n_id for n_id in branch], current

        branch.append(nid)

        outs = _successors(bpmn, current)
        if len(outs) != 1:

            return None, None
        current = outs[0]

    return None, None

def _activity_names_in(bpmn: BPMN, node_ids: list[str]) -> list[str]:
    """Return the ``name`` of every node in *node_ids* whose type is a"""

    id_to_node = {n.get_id(): n for n in bpmn.get_nodes()}
    out: list[str] = []
    for nid in node_ids:
        n = id_to_node.get(nid)
        if n is None:
            continue
        if isinstance(n, BPMN.Activity) and not isinstance(n, BPMN.SubProcess):
            name = n.get_name()
            if name:
                out.append(name)
    return out

def locate_exception_branch(
    bpmn:                            BPMN,
    divergence_point:                str,
    exception_exclusive_activities:  list[str],
    dominant_suffix:                 list[str],
    max_hops_to_split:               int = MAX_HOPS_TO_SPLIT,
) -> tuple[Optional[ExceptionLocation], str]:
    """Locate the BPMN's exception branch downstream of *divergence_point*."""

    div_node = _find_node_by_name(bpmn, divergence_point)
    if div_node is None:
        return None, f"divergence_point {divergence_point!r} not unique in BPMN"

    split = _walk_to_first_split(bpmn, div_node, max_hops_to_split)
    if split is None:
        return None, (
            f"no XOR split found within {max_hops_to_split} hops "
            f"of {divergence_point!r}"
        )

    branches: list[tuple[list[str], object]] = []
    for s in _successors(bpmn, split):
        nodes, join = _collect_branch_until_join(bpmn, s, split)
        if nodes is None:
            return None, (
                f"branch from split {split.get_id()!r} does not cleanly "
                f"terminate at a XOR join — the branch reaches "
                f"{_describe_blocker(bpmn, s)}, so this divergence point has no "
                f"single-entry single-exit region to encapsulate"
            )
        branches.append((nodes, join))

    join_ids = {j.get_id() for _, j in branches}
    if len(join_ids) != 1:
        return None, (
            f"branches converge to {len(join_ids)} different joins; "
            "structure not a clean XOR sandwich"
        )
    join = branches[0][1]

    exception_set = set(exception_exclusive_activities)
    dominant_first = dominant_suffix[0] if dominant_suffix else None

    exc_branches:     list[list[str]] = []
    dominant_marked:  list[list[str]] = []
    empty_branches:   list[list[str]] = []
    other_count = 0

    for branch_ids, _ in branches:
        names = _activity_names_in(bpmn, branch_ids)
        if any(a in exception_set for a in names):
            exc_branches.append(branch_ids)
        elif dominant_first is not None and dominant_first in names:
            dominant_marked.append(branch_ids)
        elif not names:
            empty_branches.append(branch_ids)
        else:
            other_count += 1

    if len(exc_branches) == 0:
        return None, (
            "no branch contains any of the exception_exclusive_activities "
            f"{exception_exclusive_activities!r}"
        )
    if len(exc_branches) > 1:
        return None, (
            f"{len(exc_branches)} branches contain exception activities; "
            "exception location is ambiguous"
        )

    if len(dominant_marked) == 1:
        dominant_ids = dominant_marked[0]
    elif len(dominant_marked) == 0 and len(empty_branches) == 1:
        dominant_ids = empty_branches[0]
    elif len(dominant_marked) > 1:
        return None, (
            f"{len(dominant_marked)} branches carry the dominant suffix "
            f"head {dominant_first!r}; dominant branch is ambiguous"
        )
    else:
        return None, (
            "no dominant branch identifiable: none carry "
            f"{dominant_first!r}, and {len(empty_branches)} empty "
            "branches available (need exactly 1)"
        )

    other_count += len(empty_branches) - (
        1 if (len(dominant_marked) == 0 and len(empty_branches) == 1) else 0
    )
    other_count += max(0, len(dominant_marked) - 1)

    exception_ids = exc_branches[0]
    location = ExceptionLocation(
        divergence_node_id=div_node.get_id(),
        divergence_node_name=divergence_point,
        split_gateway_id=split.get_id(),
        join_gateway_id=join.get_id(),
        exception_branch_node_ids=exception_ids,
        exception_branch_activity_names=_activity_names_in(bpmn, exception_ids),
        dominant_branch_node_ids=dominant_ids,
        dominant_branch_activity_names=_activity_names_in(bpmn, dominant_ids),
        other_branch_count=other_count,
    )
    return location, ""

def encapsulate_exception_branch(
    bpmn:             BPMN,
    location:         ExceptionLocation,
    subprocess_name:  str = DEFAULT_SUBPROCESS_NAME,
) -> tuple[Optional[EncapsulationSummary], str]:
    """Replace the exception branch's node chain with a single"""

    chain_ids = list(location.exception_branch_node_ids)
    if not chain_ids:
        return None, "exception branch is empty; nothing to encapsulate"

    encapsulated_names = list(location.exception_branch_activity_names)

    try:
        sub = encapsulate_chain_as_subprocess(bpmn, chain_ids, subprocess_name)
    except ValueError as e:
        return None, str(e)

    return (
        EncapsulationSummary(
            applied=True,
            subprocess_node_id=sub.get_id(),
            subprocess_name=sub.get_name(),
            encapsulated_node_count=len(chain_ids),
            encapsulated_activity_names=encapsulated_names,
        ),
        "",
    )

def apply_exception(
    candidate:        dict,
    df:               pd.DataFrame,
    bpmn:             BPMN,
    subprocess_name:  str = DEFAULT_SUBPROCESS_NAME,
) -> ExceptionResult:
    """Apply Phase 3 Exception to a single detector candidate."""

    candidate_id     = candidate["candidate_id"]
    divergence_point = candidate["divergence_point"]
    variant_diff     = candidate.get("variant_diff", {}) or {}
    exception_exclusive_activities = list(
        variant_diff.get("exception_exclusive_activities", [])
    )
    dominant_suffix = list(variant_diff.get("dominant_suffix", []))

    if not exception_exclusive_activities:
        return ExceptionResult(
            candidate_id=candidate_id,
            divergence_point=divergence_point,
            skipped=True,
            skip_reason=(
                "candidate has no exception_exclusive_activities; "
                "nothing to isolate"
            ),
        )

    location, reason = locate_exception_branch(
        bpmn,
        divergence_point=divergence_point,
        exception_exclusive_activities=exception_exclusive_activities,
        dominant_suffix=dominant_suffix,
    )
    if location is None:
        return ExceptionResult(
            candidate_id=candidate_id,
            divergence_point=divergence_point,
            skipped=True,
            skip_reason=f"locator skipped: {reason}",
        )

    encapsulation, reason = encapsulate_exception_branch(
        bpmn, location, subprocess_name=subprocess_name,
    )
    if encapsulation is None:
        return ExceptionResult(
            candidate_id=candidate_id,
            divergence_point=divergence_point,
            location=location,
            skipped=True,
            skip_reason=f"encapsulation skipped: {reason}",
        )

    return ExceptionResult(
        candidate_id=candidate_id,
        divergence_point=divergence_point,
        location=location,
        encapsulation=encapsulation,
        skipped=False,
    )

def result_to_json(result: ExceptionResult) -> dict:
    """Convert a single ExceptionResult to a plain dict for JSON export."""
    return {
        "candidate_id":     result.candidate_id,
        "divergence_point": result.divergence_point,
        "skipped":          result.skipped,
        "skip_reason":      result.skip_reason,
        "location":         asdict(result.location)      if result.location      else None,
        "encapsulation":    asdict(result.encapsulation) if result.encapsulation else None,
    }

def save_result_json(result: ExceptionResult, path: str | Path) -> None:
    """Persist a single ExceptionResult to *path* as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_to_json(result), f, indent=2, ensure_ascii=False)
    print(f"Saved Phase 3 EXC result to {path}")

def _fitted_mean_s(name: str, params: list) -> float:
    """The mean of a Simod-fitted duration distribution, in seconds."""

    vals = [float(x.get("value", 0.0)) for x in params if isinstance(x, dict)]
    if not vals:
        return 0.0
    if name.lower() in ("uniform", "default") and len(vals) >= 2:
        return (vals[0] + vals[1]) / 2.0
    return vals[0]

def adapt_params_after_exc(
    params: dict,
    result: ExceptionResult,
    bpmn_after: BPMN,
) -> dict:
    """Sync a Prosimos params JSON dict to reflect an Exception isolation."""

    if result.skipped or result.location is None:
        return {"applied": False, "reason": result.skip_reason or "no location"}

    chain_uuids = set(result.location.exception_branch_node_ids)
    subprocess_uuid = (result.encapsulation.subprocess_node_id
                       if result.encapsulation and result.encapsulation.applied
                       else None)

    chain_resource_ids: list[str] = []
    seen_rids: set[str] = set()
    chain_total_duration_s: float = 0.0
    for entry in params.get("task_resource_distribution", []):
        if entry.get("task_id") not in chain_uuids:
            continue
        for r in entry.get("resources", []):
            rid = r.get("resource_id")
            if rid and rid not in seen_rids:
                chain_resource_ids.append(rid)
                seen_rids.add(rid)

        for r in entry.get("resources", []):
            dp = r.get("distribution_params", [])
            if dp and isinstance(dp[0], dict) and "value" in dp[0]:
                chain_total_duration_s += _fitted_mean_s(
                    str(r.get("distribution_name", "fix")), dp)
                break

    trd = params.get("task_resource_distribution", [])
    before = len(trd)
    params["task_resource_distribution"] = [
        e for e in trd if e.get("task_id") not in chain_uuids
    ]
    trd_dropped = before - len(params["task_resource_distribution"])

    assigned_removed = 0
    for prof in params.get("resource_profiles", []):
        for res in prof.get("resource_list", []):
            tasks = res.get("assignedTasks", [])
            before = len(tasks)
            res["assignedTasks"] = [t for t in tasks if t not in chain_uuids]
            assigned_removed += before - len(res["assignedTasks"])

    ea = params.get("event_attributes", [])
    before = len(ea)
    params["event_attributes"] = [
        e for e in ea if e.get("event_id") not in chain_uuids
    ]
    ea_dropped = before - len(params["event_attributes"])

    subprocess_registered = False
    if subprocess_uuid and chain_resource_ids:
        new_trd_entry = {
            "task_id": subprocess_uuid,
            "resources": [
                {
                    "resource_id":         rid,
                    "distribution_name":   "fix",
                    "distribution_params": [{"value": chain_total_duration_s}],
                }
                for rid in chain_resource_ids
            ],
        }
        params["task_resource_distribution"].append(new_trd_entry)

        assigned_added = 0
        for prof in params.get("resource_profiles", []):
            for res in prof.get("resource_list", []):
                if res.get("id") in seen_rids:
                    res.setdefault("assignedTasks", []).append(subprocess_uuid)
                    assigned_added += 1
        subprocess_registered = True
    else:
        assigned_added = 0

    gw_report = lazy_reconcile_gateway_probabilities(params, bpmn_after)

    return {
        "applied": True,
        "chain_uuid_count": len(chain_uuids),
        "task_resource_distribution_dropped": trd_dropped,
        "assigned_tasks_removed": assigned_removed,
        "event_attributes_dropped": ea_dropped,
        "subprocess_registered": subprocess_registered,
        "subprocess_uuid": subprocess_uuid,
        "subprocess_resource_count": len(chain_resource_ids),
        "subprocess_duration_s": chain_total_duration_s,
        "subprocess_assigned_added": assigned_added,
        **gw_report,
    }
