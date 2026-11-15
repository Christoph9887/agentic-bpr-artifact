"""Run all seven redesign-heuristic detectors and merge their outputs."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mabpr.foundation.detectors import (
    activity_elimination_detector as ae,
    case_assignment_detector as ca,
    case_based_work_detector as cbw,
    exception_detector as exc,
    knockout_detector as ko,
    parallelism_detector as par,
    resequencing_detector as rsq,
)
from mabpr.detectors import (
    activity_composition_detector as ac,
    case_types_detector as ct,
    contact_reduction_detector as cr,
    extra_resources_detector as xr,
    interfacing_detector as ifc,
    numerical_involvement_detector as ni,
    specialize_detector as spc,
    triage_detector as tri,
)
from mabpr.foundation.detectors.feature_extraction import (
    FeatureStore,
    extract_features,
    prepare_dataframe,
    print_summary,
)

from mabpr import datasets  # noqa: E402

DEFAULT_LOG_PATH   = datasets.get().log_xes
DEFAULT_MODEL_PATH = datasets.get().detector_model
DEFAULT_OUTPUT_DIR = datasets.get().detector_results.parent

@dataclass(frozen=True)
class DetectorSpec:
    """Static description of how to invoke and persist one heuristic detector."""

    name: str
    detect: Callable[..., list]
    print_fn: Callable[[list], None]
    save_fn: Callable[[list, str], None]
    output_filename: str
    requires_df: bool

    reports: bool = False

    requires_context: bool = False

DETECTORS: list[DetectorSpec] = [
    DetectorSpec(
        "Activity Elimination",
        ae.detect_activity_elimination, ae.print_candidates, ae.save_candidates_json,
        "activity_elimination_candidates.json", requires_df=False,
    ),
    DetectorSpec(
        "Parallelism",
        par.detect_parallelism, par.print_candidates, par.save_candidates_json,
        "parallelism_candidates.json", requires_df=True,
    ),
    DetectorSpec(
        "Case-based Work",
        cbw.detect_case_based_work, cbw.print_candidates, cbw.save_candidates_json,
        "case_based_work_candidates.json", requires_df=True,
    ),
    DetectorSpec(
        "Case Assignment",
        ca.detect_case_assignment, ca.print_candidates, ca.save_candidates_json,
        "case_assignment_candidates.json", requires_df=True,
    ),
    DetectorSpec(
        "Exception",
        exc.detect_exception, exc.print_candidates, exc.save_candidates_json,
        "exception_candidates.json", requires_df=False, reports=True,
    ),
    DetectorSpec(
        "Knock-out",
        ko.detect_knockout, ko.print_candidates, ko.save_candidates_json,
        "knockout_candidates.json", requires_df=True,
    ),
    DetectorSpec(
        "Resequencing",
        rsq.detect_resequencing, rsq.print_candidates, rsq.save_candidates_json,
        "resequencing_candidates.json", requires_df=True,
    ),
    DetectorSpec(
        "Extra Resources",
        xr.detect_extra_resources_from_store, xr.print_candidates,
        xr.save_candidates_json,
        "extra_resources_candidates.json", requires_df=True, requires_context=True,
    ),
    DetectorSpec(
        "Activity Composition",
        ac.detect_activity_composition, ac.print_candidates, ac.save_candidates_json,
        "activity_composition_candidates.json", requires_df=True, requires_context=True,
    ),
    DetectorSpec(
        "Specialize",
        spc.detect_specialize, spc.print_candidates, spc.save_candidates_json,
        "specialize_candidates.json", requires_df=True, requires_context=True,
    ),
    DetectorSpec(
        "Case Types",
        ct.detect_case_types, ct.print_candidates, ct.save_candidates_json,
        "case_types_candidates.json", requires_df=True, requires_context=True,
    ),
    DetectorSpec(
        "Contact Reduction",
        cr.detect_contact_reduction, cr.print_candidates, cr.save_candidates_json,
        "contact_reduction_candidates.json", requires_df=True, requires_context=True,
    ),
    DetectorSpec(
        "Interfacing",
        ifc.detect_interfacing, ifc.print_candidates, ifc.save_candidates_json,
        "interfacing_candidates.json", requires_df=True, requires_context=True,
    ),
    DetectorSpec(
        "Triage",
        tri.detect_triage, tri.print_candidates, tri.save_candidates_json,
        "triage_candidates.json", requires_df=True, requires_context=True,
    ),
    DetectorSpec(
        "Numerical Involvement",
        ni.detect_numerical_involvement, ni.print_candidates, ni.save_candidates_json,
        "numerical_involvement_candidates.json",
        requires_df=True, requires_context=True,
    ),
]

def run_detector(
    spec: DetectorSpec,
    store: FeatureStore,
    df: pd.DataFrame,
    output_dir: Path,
    context: dict | None = None,
) -> None:
    """Execute one detector and persist its candidates to JSON."""
    print(f"\nRunning {spec.name} detector...")
    if spec.requires_context:
        report: list = []
        candidates = spec.detect(store, df, context=context or {}, report=report)
    elif spec.reports:
        report = {}
        candidates = spec.detect(store, report=report)
    else:
        report = None
        candidates = spec.detect(store, df) if spec.requires_df else spec.detect(store)
    spec.print_fn(candidates)
    if report is None:
        spec.save_fn(candidates, str(output_dir / spec.output_filename))
    else:
        spec.save_fn(candidates, str(output_dir / spec.output_filename), report=report)

def merge_detector_outputs(input_dir: Path, output_file: Path) -> None:
    """Merge the known per-detector JSONs in *input_dir* into a single flat list."""

    expected = {spec.output_filename for spec in DETECTORS}
    merged: list[Any] = []

    for filename in sorted(expected):
        file_path = input_dir / filename
        if not file_path.is_file():
            continue

        print(f"Reading: {filename}")
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as err:
            print(f"Skipping {filename}, error: {err}")
            continue

        if isinstance(data, list):
            merged.extend(data)
        elif isinstance(data, dict) and "candidates" in data:
            merged.extend(data["candidates"])
        else:
            merged.append(data)

    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, ensure_ascii=False)

    print(f"\nMerged file saved to: {output_file}")

def _attach_delay_split(output_dir: Path, merged_output: Path,
                        params_path: Path, baseline_bpmn_path: Path) -> None:
    """Apportion each CBW target's extraneous-delay timer between batching and the rest."""

    from mabpr.evidence.batch_delay import split as batch_split
    try:
        per_activity = batch_split(params_path, baseline_bpmn_path)
    except Exception as exc:                                        # noqa: BLE001
        print(f"\n[batch split] NOT attached: {type(exc).__name__}: {exc}")
        print("[batch split] Case-based Work will refuse until this succeeds; run "
              f"`python -m mabpr.evidence.batch_delay --params {params_path} "
              f"--bpmn {baseline_bpmn_path}` to see why")
        return
    touched: list[str] = []
    for path in (merged_output, output_dir / "case_based_work_candidates.json"):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        cands = payload if isinstance(payload, list) else (payload.get("candidates") or [])
        for cand in cands:
            if cand.get("heuristic") != "Case-based Work":
                continue
            rec = per_activity.get(cand.get("target_activity"))
            if rec is None:
                continue
            cand.setdefault("evidence", {})["extraneous_delay_split"] = rec
            touched.append(f"{cand.get('candidate_id')}@{path.name}")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[batch split] attached to {len(touched)} candidate record(s): {touched}")
    ranked = sorted(per_activity.items(),
                    key=lambda kv: -(kv[1].get("share_of_timer") or 0.0))
    for activity, rec in ranked[:8]:
        print("  %-32s %5.1f%% of its %8.2f h timer  (z=%s)"
              % (activity[:32], 100 * (rec.get("share_of_timer") or 0.0),
                 rec.get("timer_h") or 0.0, rec.get("z")))

def run_detectors(
    log_path:   str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    params_path: str | Path | None = None,
    baseline_bpmn_path: str | Path | None = None,
) -> list[Any]:
    """Run all 7 redesign-heuristic detectors on a single (log, model) pair."""

    log_path   = Path(log_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_output = output_dir / "detector_results.json"

    df = prepare_dataframe(str(log_path))
    store = extract_features(str(log_path), str(model_path))
    print_summary(store)

    context: dict = {}
    if params_path and Path(params_path).exists():
        with open(params_path, "r", encoding="utf-8") as fh:
            context["params"] = json.load(fh)
    if baseline_bpmn_path and Path(baseline_bpmn_path).exists():
        context["baseline_bpmn"] = str(baseline_bpmn_path)

    for spec in DETECTORS:
        run_detector(spec, store, df, output_dir, context=context)

    merge_detector_outputs(output_dir, merged_output)

    if params_path and baseline_bpmn_path \
            and Path(params_path).exists() and Path(baseline_bpmn_path).exists():
        _attach_delay_split(output_dir, merged_output,
                            Path(params_path), Path(baseline_bpmn_path))

    with open(merged_output, "r", encoding="utf-8") as fh:
        return json.load(fh)

if __name__ == "__main__":
    ds = datasets.from_argv(sys.argv)
    print(f"dataset {ds.tag}: {ds.log_xes.name} + {ds.detector_model.name}")
    run_detectors(ds.log_xes, ds.detector_model, ds.detector_results.parent,
                  params_path=ds.baseline_params,
                  baseline_bpmn_path=ds.baseline_bpmn)
