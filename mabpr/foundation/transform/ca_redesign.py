"""Case Assignment — heuristic-level redesign module."""

from __future__ import annotations

import os

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from pm4py.objects.bpmn.obj import BPMN

@dataclass
class CaseAssignmentResult:
    """Per-candidate outcome of Phase 3 CA."""

    candidate_id:        str
    target_activities:   list[str]                = field(default_factory=list)
    target_task_ids:     list[str]                = field(default_factory=list)
    skipped:             bool                     = False
    skip_reason:         str                      = ""

    missing_target_activities: list[str]          = field(default_factory=list)

def apply_case_assignment(
    candidate:  dict,
    df:         pd.DataFrame,
    bpmn:       BPMN,
) -> CaseAssignmentResult:
    """Resolve target activity NAMES to UUIDs via the BPMN. CA does"""

    candidate_id = candidate["candidate_id"]
    target_activities = list(candidate.get("target_activities") or [])

    if len(target_activities) < 2:
        return CaseAssignmentResult(
            candidate_id=candidate_id,
            target_activities=target_activities,
            skipped=True,
            skip_reason=(
                "CA needs at least 2 target activities to consolidate; "
                f"got {len(target_activities)}"
            ),
        )

    target_task_ids: list[str] = []
    missing: list[str] = []
    for name in target_activities:
        matches = [n for n in bpmn.get_nodes() if n.get_name() == name]
        if len(matches) == 0:
            missing.append(name)
            continue
        if len(matches) > 1:
            return CaseAssignmentResult(
                candidate_id=candidate_id,
                target_activities=target_activities,
                skipped=True,
                skip_reason=(
                    f"target activity {name!r} has {len(matches)} matches "
                    "in BPMN; not unique"
                ),
            )
        target_task_ids.append(str(matches[0].get_id()))

    if len(target_task_ids) < 2:
        return CaseAssignmentResult(
            candidate_id=candidate_id,
            target_activities=target_activities,
            skipped=True,
            skip_reason=(
                f"only {len(target_task_ids)} of {len(target_activities)} target "
                f"activities remain in the BPMN (missing: {missing}); CA needs at "
                "least 2 to consolidate"
            ),
        )

    return CaseAssignmentResult(
        candidate_id=candidate_id,
        target_activities=target_activities,
        target_task_ids=target_task_ids,
        skipped=False,
        missing_target_activities=missing,
    )

def result_to_json(result: CaseAssignmentResult) -> dict:
    """Convert a CaseAssignmentResult to a plain dict for JSON export."""
    return asdict(result)

def save_result_json(result: CaseAssignmentResult, path: str | Path) -> None:
    """Persist a single CaseAssignmentResult to *path* as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_to_json(result), f, indent=2, ensure_ascii=False)
    print(f"Saved Phase 3 CA result to {path}")

_FIX_ZERO_DIST = {
    "distribution_name": "fix",
    "distribution_params": [{"value": 0.0}],
}

def _is_instantaneous_task(entry: dict) -> bool:
    """True if the activity is an instantaneous state activity — every existing"""

    resources = entry.get("resources", [])
    if not resources:
        return False
    for r in resources:
        if r.get("distribution_name") != "fix":
            return False
        vals = [p.get("value", 0.0) for p in r.get("distribution_params", [])]
        if any(abs(v) > 1e-9 for v in vals):
            return False
    return True

CA_STRATEGY = os.environ.get("MABPR_CA_STRATEGY", "intersection")

MIN_CA_POOL = 2

def _intersection_pool(params: dict, target_task_ids: set) -> tuple:
    """Resources that should own the whole target set, and the index of all."""

    index: dict = {}
    coverage: dict = {}
    for prof in params.get("resource_profiles", []):
        for res in prof.get("resource_list", []):
            rid = str(res.get("id"))
            index[rid] = res
            coverage[rid] = len(set(res.get("assignedTasks", [])) & target_task_ids)

    full = {rid for rid, n in coverage.items() if n == len(target_task_ids)}
    if len(full) >= MIN_CA_POOL:
        return full, index, len(full)

    ranked = sorted((rid for rid, n in coverage.items() if n > 0),
                    key=lambda r: -coverage[r])
    pool = set(full)
    for rid in ranked:
        if len(pool) >= MIN_CA_POOL:
            break
        pool.add(rid)
    return pool, index, len(full)

def _apply_intersection(params: dict, result: CaseAssignmentResult) -> dict:
    """Concentrate the target activities on the smallest viable shared pool."""
    target_task_ids = set(result.target_task_ids)
    pool, index, n_already_full = _intersection_pool(params, target_task_ids)

    assigned_added = assigned_removed = 0
    for rid, res in index.items():
        assigned = list(res.get("assignedTasks", []))
        if rid in pool:
            for tid in target_task_ids:
                if tid not in assigned:
                    assigned.append(tid)
                    assigned_added += 1
        else:
            before = len(assigned)
            assigned = [t for t in assigned if t not in target_task_ids]
            assigned_removed += before - len(assigned)
        res["assignedTasks"] = assigned

    _original_pool_sizes = {
        str(e.get("task_id")): len(e.get("resources") or [])
        for e in params.get("task_resource_distribution", [])
        if str(e.get("task_id")) in target_task_ids}
    _widest_original_pool = max(_original_pool_sizes.values(), default=0)

    resources_removed = 0
    resources_added = 0
    pool_sizes = {}
    for entry in params.get("task_resource_distribution", []):
        tid = str(entry.get("task_id"))
        if tid not in target_task_ids:
            continue
        original = entry.get("resources", [])
        kept = [r for r in original if str(r.get("resource_id")) in pool]

        if not kept:
            donor = next((r for r in original), None)
            if donor is not None:
                kept = [donor]
                pool.add(str(donor.get("resource_id")))

        present = {str(r.get("resource_id")) for r in kept}
        template = kept[0] if kept else None
        for rid in sorted(pool - present):
            if template is None:
                break
            new_pair = {k: v for k, v in template.items()}
            new_pair["resource_id"] = rid
            kept.append(new_pair)
            resources_added += 1
        resources_removed += len(original) - len([r for r in original
                                                  if str(r.get("resource_id")) in pool])
        entry["resources"] = kept
        pool_sizes[tid] = len(kept)

    return {
        "applied": True,
        "strategy": "intersection",
        "ca_pool_resource_count": len(pool),
        "expected_continuity_after": round(1.0 / max(1, len(pool)), 4),
        "expected_continuity_before": round(1.0 / max(1, _widest_original_pool), 4),
        "pool_before_per_target": _original_pool_sizes,
        "concentration": (round(len(pool) / _widest_original_pool, 3)
                          if _widest_original_pool else None),
        "delivery_note": (
            f"with {len(pool)} resources eligible for every target, two "
            f"consecutive steps land on the same person about "
            f"{100.0 / max(1, len(pool)):.1f}% of the time, against "
            f"{100.0 / max(1, _widest_original_pool):.1f}% before. Case "
            f"Assignment has no other way to deliver continuity here: Prosimos "
            f"picks by earliest availability and does not remember who handled "
            f"the previous step."),
        "resources_already_covering_all_targets": n_already_full,
        "resources_removed": resources_removed,
        "task_resources_added": resources_added,
        "assigned_tasks_added": assigned_added,
        "assigned_tasks_removed": assigned_removed,
        "pool_per_target": pool_sizes,
        "note": ("the target activities are now performed by one shared pool of "
                 f"{len(pool)}; with k eligible resources, consecutive steps land "
                 "on the same person with probability ~1/k, which is how this "
                 "simulator can express 'the same worker continues the case'"),
    }

def adapt_params_after_ca(
    params: dict,
    result: CaseAssignmentResult,
) -> dict:
    """Apply CA (resource-level union) to the Prosimos params JSON."""

    if result.skipped or not result.target_task_ids:
        return {"applied": False, "reason": result.skip_reason}

    if CA_STRATEGY == "intersection":
        return _apply_intersection(params, result)

    target_task_ids: set[str] = set(result.target_task_ids)

    resource_profiles = params.get("resource_profiles", [])
    ca_pool_ids: set[str] = set()

    resource_id_to_entry: dict[str, dict] = {}
    for prof in resource_profiles:
        for res in prof.get("resource_list", []):
            rid = str(res.get("id"))
            resource_id_to_entry[rid] = res
            assigned = set(res.get("assignedTasks", []))
            if assigned & target_task_ids:
                ca_pool_ids.add(rid)

    assigned_tasks_added = 0
    for rid in ca_pool_ids:
        res = resource_id_to_entry[rid]
        assigned = list(res.get("assignedTasks", []))
        existing = set(assigned)
        for tid in target_task_ids:
            if tid not in existing:
                assigned.append(tid)
                existing.add(tid)
                assigned_tasks_added += 1
        res["assignedTasks"] = assigned

    trd_list = params.get("task_resource_distribution", [])
    trd_by_task: dict[str, dict] = {
        str(e.get("task_id")): e for e in trd_list
    }

    res_dist_per_target: dict[str, dict[str, dict]] = {
        rid: {} for rid in ca_pool_ids
    }
    for tid in target_task_ids:
        entry = trd_by_task.get(tid)
        if entry is None:
            continue
        for r in entry.get("resources", []):
            rid = str(r.get("resource_id"))
            if rid in res_dist_per_target:
                res_dist_per_target[rid][tid] = r

    task_resources_added = 0
    fallback_zero_dist = 0
    instantaneous_kept_zero = 0
    for tid in target_task_ids:
        entry = trd_by_task.get(tid)
        if entry is None:

            continue

        instantaneous = _is_instantaneous_task(entry)
        present_rids = {str(r.get("resource_id")) for r in entry["resources"]}
        for rid in ca_pool_ids:
            if rid in present_rids:
                continue
            donor = None
            if not instantaneous:

                for other_tid, other_dist in res_dist_per_target.get(rid, {}).items():
                    if other_tid != tid:
                        donor = other_dist
                        break
            if donor is None:
                new_entry = {"resource_id": rid, **_FIX_ZERO_DIST}
                if instantaneous:
                    instantaneous_kept_zero += 1
                else:
                    fallback_zero_dist += 1
            else:
                new_entry = {
                    "resource_id": rid,
                    "distribution_name": donor.get("distribution_name", "fix"),
                    "distribution_params": list(donor.get("distribution_params", [{"value": 0.0}])),
                }
            entry["resources"].append(new_entry)
            task_resources_added += 1

    return {
        "applied": True,
        "ca_pool_resource_count": len(ca_pool_ids),
        "assigned_tasks_added": assigned_tasks_added,
        "task_resources_added": task_resources_added,
        "fallback_zero_dist": fallback_zero_dist,
        "instantaneous_kept_zero": instantaneous_kept_zero,

        "missing_target_activities": list(result.missing_target_activities),
        "targets_applied": len(result.target_task_ids),
        "targets_declared": len(result.target_activities),
    }
