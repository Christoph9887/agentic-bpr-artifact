"""Phase 3 Hybrid Transformation — orchestrator."""

from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import re

import pm4py

import pm4py.objects.bpmn.exporter.variants.etree as _bpmn_etree

_orig_get_xml_string = _bpmn_etree.get_xml_string

def _patched_get_xml_string(bpmn_graph, parameters=None):

    from collections import Counter
    pcounts = Counter()
    nodes = list(bpmn_graph.get_nodes())
    flows = list(bpmn_graph.get_flows())
    for n in nodes:
        pcounts[n.get_process()] += 1
    for f in flows:
        pcounts[f.get_process()] += 1
    if pcounts:
        canonical = pcounts.most_common(1)[0][0]
        for n in nodes:
            if n.get_process() != canonical:
                n.set_process(canonical)
        for f in flows:
            if f.get_process() != canonical:
                f.set_process(canonical)

    all_processes = set()
    for n in nodes:
        all_processes.add(n.get_process())
    for f in flows:
        all_processes.add(f.get_process())
    if len(all_processes) > 1:
        from pm4py.objects.bpmn.obj import BPMN
        if not any(isinstance(n, BPMN.Collaboration) for n in bpmn_graph.get_nodes()):
            collab = BPMN.Collaboration(id="id" + list(all_processes)[0], name="")
            bpmn_graph.add_node(collab)
    return _orig_get_xml_string(bpmn_graph, parameters=parameters)

_bpmn_etree.get_xml_string = _patched_get_xml_string

def _normalize_double_id_prefix(bpmn_path) -> int:
    """Strip the 'id' prefix pm4py's exporter prepends to every sequenceFlow id."""

    path = Path(bpmn_path)
    text = path.read_text(encoding="utf-8")
    flow_ids = set(re.findall(
        r'<(?:bpmn:)?sequenceFlow\b[^>]*?\bid="([^"]+)"', text))
    n = 0
    for fid in sorted(flow_ids, key=len, reverse=True):
        if not fid.startswith("id") or len(fid) <= 2:
            continue

        text, k = re.subn(r'(?<=["\>])' + re.escape(fid) + r'(?=["\<])',
                          fid[2:], text)
        n += k
    if n:
        path.write_text(text, encoding="utf-8")
    return n

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mabpr.foundation.detectors.feature_extraction import prepare_dataframe   # noqa: E402

from mabpr.foundation.transform import (                                       # noqa: E402
    ae_redesign, pa_redesign, re_redesign, ko_redesign, exc_redesign,
    ca_redesign, cbw_redesign,
)

def _stale_gateway_paths(params: dict, bpmn) -> list:
    """Params paths that name a flow the gateway no longer has."""

    outgoing: dict = {}
    for f in bpmn.get_flows():
        outgoing.setdefault(str(f.get_source().get_id()), set()).add(str(f.get_id()))
    nodes = {str(n.get_id()) for n in bpmn.get_nodes()}
    bad = []
    for entry in params.get("gateway_branching_probabilities") or []:
        gid = str(entry.get("gateway_id"))
        if gid not in nodes:
            continue
        have = outgoing.get(gid, set())
        for p in entry.get("probabilities") or []:
            if str(p.get("path_id")) not in have:
                bad.append((gid, str(p.get("path_id"))))
    return bad

DISPATCH: dict[str, tuple[Any, Any, bool]] = {
    "Activity Elimination": (ae_redesign.apply_activity_elimination,
                             ae_redesign.adapt_params_after_ae, True),
    "Parallelism":          (pa_redesign.apply_parallelism,
                             pa_redesign.adapt_params_after_par,  True),
    "Resequencing":         (re_redesign.apply_resequencing,
                             re_redesign.adapt_params_after_rsq,  True),
    "Knock-out":            (ko_redesign.apply_knockout,
                             ko_redesign.adapt_params_after_ko,   True),
    "Exception":            (exc_redesign.apply_exception,
                             exc_redesign.adapt_params_after_exc, True),
    "Case Assignment":      (ca_redesign.apply_case_assignment,
                             ca_redesign.adapt_params_after_ca,   False),

    "Case-based Work":      (cbw_redesign.apply_case_based_work,
                             cbw_redesign.adapt_params_after_cbw, True),
}

DEFAULT_PHASE2_PLAN = REPO_ROOT / "inputs" / "2012" / "phase2_result" / "phase2_plan_unfiltered_balanced_nop1.json"
DEFAULT_NORMALIZED  = REPO_ROOT / "inputs" / "2012" / "phase2_result" / "unfiltered_normalized_candidates.json"
DEFAULT_DETECTOR    = REPO_ROOT / "inputs" / "2012" / "detected_candidates" / "detector_results.json"
DEFAULT_BPMN        = REPO_ROOT / "simulation" / "output" / "resource_model" / "BPI_Challenge_2012_test_large.bpmn"
DEFAULT_PARAMS      = REPO_ROOT / "simulation" / "output" / "resource_model" / "BPI_Challenge_2012_test_large.json"
DEFAULT_LOG         = REPO_ROOT / "data" / "BPI_Challenge_2012_test_large.xes"
DEFAULT_OUTPUT_DIR  = REPO_ROOT / "simulation" / "redesigned" / "unfiltered_balanced_nop1"

def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_phase3_transformation(
    phase2_plan_path:           str | Path,
    normalized_candidates_path: str | Path,
    detector_results_path:      str | Path,
    baseline_bpmn_path:         str | Path,
    baseline_params_path:       str | Path,
    log_path:                   str | Path,
    output_dir:                 str | Path,
) -> dict:
    """Execute Phase 3 Hybrid Transformation end-to-end."""

    phase2_plan_path           = Path(phase2_plan_path)
    normalized_candidates_path = Path(normalized_candidates_path)
    detector_results_path      = Path(detector_results_path)
    baseline_bpmn_path         = Path(baseline_bpmn_path)
    baseline_params_path       = Path(baseline_params_path)
    log_path                   = Path(log_path)
    output_dir                 = Path(output_dir)

    print(f"\n=== Phase 3 Orchestrator ===")
    print(f"  phase2_plan : {phase2_plan_path.relative_to(REPO_ROOT) if REPO_ROOT in phase2_plan_path.parents else phase2_plan_path}")
    print(f"  baseline BPMN: {baseline_bpmn_path.name}")
    print(f"  baseline params: {baseline_params_path.name}")
    print(f"  output_dir  : {output_dir}")

    plan       = _load_json(phase2_plan_path)
    normalized = _load_json(normalized_candidates_path)  # noqa: F841 — held for future use
    detector_results = _load_json(detector_results_path)

    detector_by_id = {c["candidate_id"]: c for c in detector_results}

    print(f"\n[Step 0] Inputs loaded:")
    print(f"  detector candidates : {len(detector_results)}")
    print(f"  normalized candidates: {len(normalized)}")
    print(f"  plan phases          : {len(plan.get('phases', []))}")
    print(f"  plan retained        : {sum(len(p.get('candidates', [])) for p in plan.get('phases', []))}")

    print(f"\n[Step 0] Building DataFrame from {log_path.name}...")
    df = prepare_dataframe(str(log_path))

    print(f"\n[Step 0] Reading BPMN + params...")
    bpmn = pm4py.read_bpmn(str(baseline_bpmn_path))
    baseline_params_loaded = _load_json(baseline_params_path)

    redesigned_params = deepcopy(baseline_params_loaded)

    log_entries: list[dict] = []

    for phase in sorted(plan.get("phases", []), key=lambda p: p["phase_id"]):
        pid = phase["phase_id"]
        cids = phase.get("candidates", [])
        print(f"\n[Phase {pid}] {len(cids)} candidate(s): {cids}")

        for cid in cids:
            if cid not in detector_by_id:
                raise KeyError(
                    f"Candidate {cid!r} from phase {pid} not found in "
                    f"detector_results ({detector_results_path.name})"
                )
            candidate = detector_by_id[cid]
            heuristic = candidate["heuristic"]

            if heuristic not in DISPATCH:
                raise KeyError(
                    f"Heuristic {heuristic!r} (candidate {cid!r}) has no "
                    f"handler registered in DISPATCH. Known: {sorted(DISPATCH)}"
                )

            apply_fn, adapt_fn, is_structural = DISPATCH[heuristic]
            print(f"  [{cid}] {heuristic} → apply_fn={apply_fn.__name__}, "
                  f"adapt_fn={adapt_fn.__name__}, structural={is_structural}")

            result = apply_fn(candidate, df, bpmn)

            if is_structural:
                adapt_report = adapt_fn(redesigned_params, result, bpmn)

                stale = _stale_gateway_paths(redesigned_params, bpmn)
                if stale:
                    adapt_report = dict(adapt_report or {})
                    adapt_report["gateway_paths_left_stale"] = stale
                    print(f"    [phase3] {heuristic}: {len(stale)} gateway path(s) "
                          f"named in params no longer exist in the BPMN; "
                          f"reconciling on its behalf")
                    from mabpr.foundation.transform.bpmn_primitives import (
                        lazy_reconcile_gateway_probabilities)
                    adapt_report["gateway_reconciliation_by_orchestrator"] = (
                        lazy_reconcile_gateway_probabilities(
                            redesigned_params, bpmn))
            else:
                adapt_report = adapt_fn(redesigned_params, result)

            log_entries.append({
                "phase_id":     pid,
                "candidate_id": cid,
                "heuristic":    heuristic,
                "status":       "success",
                "adapt_report": adapt_report,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    bpmn_out          = output_dir / "redesigned.bpmn"
    baseline_bpmn_out = output_dir / "baseline.bpmn"
    redesigned_out    = output_dir / "redesigned_params.json"
    log_out           = output_dir / "phase3_log.json"

    print(f"\n[Step Final] Writing outputs to {output_dir}")
    pm4py.write_bpmn(bpmn, str(bpmn_out))
    n_fixed = _normalize_double_id_prefix(bpmn_out)
    print(f"  → {bpmn_out.name}  (normalized {n_fixed} double-id flow refs)")

    shutil.copy(baseline_bpmn_path, baseline_bpmn_out)
    print(f"  → {baseline_bpmn_out.name}  (unmutated copy)")

    with open(redesigned_out, "w", encoding="utf-8") as f:
        json.dump(redesigned_params, f, indent=2, ensure_ascii=False)
    print(f"  → {redesigned_out.name}")

    summary = {
        "phase2_plan":     str(phase2_plan_path),
        "baseline_bpmn":   str(baseline_bpmn_path),
        "baseline_params": str(baseline_params_path),
        "output_dir":      str(output_dir),
        "n_phases":        len(plan.get("phases", [])),
        "n_candidates":    sum(len(p.get("candidates", [])) for p in plan.get("phases", [])),
        "entries":         log_entries,
    }
    with open(log_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"  → {log_out.name}  ({len(log_entries)} entries)")

    print(f"\n✓ Phase 3 complete. {len(log_entries)} candidates dispatched successfully.")
    return summary

if __name__ == "__main__":
    run_phase3_transformation(
        DEFAULT_PHASE2_PLAN,
        DEFAULT_NORMALIZED,
        DEFAULT_DETECTOR,
        DEFAULT_BPMN,
        DEFAULT_PARAMS,
        DEFAULT_LOG,
        DEFAULT_OUTPUT_DIR,
    )
