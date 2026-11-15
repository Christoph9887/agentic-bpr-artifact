"""FOOTPRINT — what each candidate directly touched, for the Critic's attribution."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pm4py
from pm4py.objects.bpmn.obj import BPMN

@dataclass(frozen=True)
class Footprint:
    """What one candidate directly touched. `activities` are NAMES (the key the"""

    candidate_id: str
    heuristic: str
    activities: list[str]
    applied: bool
    skip_reason: str = ""
    pool_merged: int | None = None

    routing_change: dict = field(default_factory=dict)
    gateways_dropped: int = 0
    paths_renormalized: int = 0
    paths_dropped: int = 0
    deleted: bool = False

    created: list = field(default_factory=list)

    removes_activities: bool = False

    structural_noop: bool = False

_ROUTING_CHANGE_FIELDS = (
    "gateways_conditioned", "branch_rules_added", "loop_probability_reduced",
    "timers_removed", "batch_rules_dropped", "rules_removed",
    "batch_component_h_removed",
    "paths_remapped", "paths_dropped", "gateways_dropped",
)

_CHANGE_FIELDS = (
    "deleted_node_id", "subprocess_uuid", "subprocess_registered",
    "ca_pool_resource_count", "task_resources_added", "assigned_tasks_added",
    "rules_removed", "gateways_dropped", "paths_renormalized", "paths_dropped",
    "paths_remapped",
    "task_resource_distribution_dropped", "assigned_tasks_removed",
    "event_attributes_dropped", "targets_applied",

    "gateways_conditioned", "branch_rules_added",

    "resources_grown", "extra_per_resource",

    "resources_removed",

    "loop_probability_reduced",

    "timers_removed",

    "batch_rules_dropped",

    "batch_component_h_removed",
)

def _changed_anything(ar: dict) -> bool:
    return any(bool(ar.get(f)) for f in _CHANGE_FIELDS)

def _task_id_to_name(baseline_bpmn_path: Path) -> dict[str, str]:
    """{node_id: name} for every Task in the baseline BPMN. Tasks only —"""

    bpmn = pm4py.read_bpmn(str(baseline_bpmn_path))
    out: dict[str, str] = {}
    for n in bpmn.get_nodes():
        if isinstance(n, BPMN.Task):
            name = n.get_name()
            if name:
                out[str(n.get_id())] = name
    return out

def _iter_strings(obj: object) -> Iterator[str]:
    """Every string value in a nested adapt_report (scalars + list items). Lets"""

    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)

def _declared_targets(det: dict) -> set[str]:
    """Activity NAMES a candidate declares as its target."""

    out: set[str] = set()

    t = det.get("target_activities")
    if isinstance(t, list):
        out |= {str(x) for x in t if x}

    t = det.get("target_activity")
    if isinstance(t, str) and t:
        out.add(t)

    dp = det.get("divergence_point")
    if isinstance(dp, str) and dp:
        out.add(dp)

    excl = (det.get("variant_diff") or {}).get("exception_exclusive_activities")
    if isinstance(excl, list):
        out |= {str(x) for x in excl if x}

    for node in (det.get("process_fragment") or {}).get("nodes") or []:
        if isinstance(node, dict) and node.get("is_target") and node.get("name"):
            out.add(str(node["name"]))

    return out

def rename_records(footprints) -> list:
    """The ENCAPSULATIONS in a round, as `compare_case_endings` wants them."""

    out = []
    for cid, f in sorted((footprints or {}).items()):
        applied = f.applied if hasattr(f, "applied") else f.get("applied")
        deleted = f.deleted if hasattr(f, "deleted") else f.get("deleted")
        removes = (f.removes_activities if hasattr(f, "removes_activities")
                   else f.get("removes_activities"))
        created = list((f.created if hasattr(f, "created") else f.get("created")) or [])
        acts = list((f.activities if hasattr(f, "activities") else f.get("activities")) or [])
        if not (applied and removes and created) or deleted:
            continue
        for composite in created:
            parts = [a for a in acts if a not in created]
            if parts:
                out.append({"composite": composite, "parts": parts,
                            "by": cid})
    return out

def read_footprints(
    phase3_dir: str | Path,
    detector_results_path: str | Path,
    baseline_bpmn_path: str | Path | None = None,
    redesigned_bpmn_path: str | Path | None = None,
) -> dict[str, Footprint]:
    """Build {candidate_id: Footprint} from a Phase-3 output directory."""

    phase3_dir = Path(phase3_dir)
    if baseline_bpmn_path is None:
        baseline_bpmn_path = phase3_dir / "baseline.bpmn"
    if redesigned_bpmn_path is None:
        redesigned_bpmn_path = phase3_dir / "redesigned.bpmn"

    log = json.loads((phase3_dir / "phase3_log.json").read_text(encoding="utf-8"))
    detectors = json.loads(Path(detector_results_path).read_text(encoding="utf-8"))
    det_by_id = {c["candidate_id"]: c for c in detectors}
    id2name = _task_id_to_name(Path(baseline_bpmn_path))

    new2name: dict[str, str] = {}
    rp = Path(redesigned_bpmn_path)
    if rp.exists():
        try:
            new2name = {k: v for k, v in _task_id_to_name(rp).items() if k not in id2name}
        except Exception:
            new2name = {}

    out: dict[str, Footprint] = {}
    for entry in log.get("entries", []):
        cid = entry["candidate_id"]
        ar = entry.get("adapt_report") or {}
        det = det_by_id.get(cid, {})

        acts: set[str] = _declared_targets(det)
        created: set[str] = set()
        for s in _iter_strings(ar):
            if s in id2name:
                acts.add(id2name[s])
            elif s in new2name:
                created.add(new2name[s])

        acts |= created

        out[cid] = Footprint(
            candidate_id=cid,
            heuristic=entry.get("heuristic", det.get("heuristic", "")),
            activities=sorted(acts),
            applied=bool(ar.get("applied", False)),
            skip_reason=str(ar.get("reason", "")),
            pool_merged=ar.get("ca_pool_resource_count"),
            routing_change={k: ar[k] for k in _ROUTING_CHANGE_FIELDS
                            if ar.get(k)},
            gateways_dropped=int(ar.get("gateways_dropped", 0) or 0),
            paths_renormalized=int(ar.get("paths_renormalized", 0) or 0),
            paths_dropped=int(ar.get("paths_dropped", 0) or 0),
            deleted="deleted_node_id" in ar,
            created=sorted(created),

            removes_activities=bool(
                "deleted_node_id" in ar
                or ar.get("subprocess_uuid") or ar.get("subprocess_registered")),
            structural_noop=(bool(ar.get("applied")) and not _changed_anything(ar)),
        )
    return out
