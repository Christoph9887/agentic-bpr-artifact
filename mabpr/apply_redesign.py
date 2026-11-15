"""Apply a set of redesign candidates to the baseline model → redesigned model."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from mabpr.foundation.transform.phase3_orchestrator import run_phase3_transformation

from mabpr.transform import register as _register_mabpr_handlers  # noqa: E402

_register_mabpr_handlers()

from mabpr import datasets  # noqa: E402

DETECTOR_RESULTS = datasets.get().detector_results
BASELINE_BPMN    = datasets.get().baseline_bpmn
BASELINE_PARAMS  = datasets.get().baseline_params
LOG_XES          = datasets.get().log_xes
DEFAULT_OUT_DIR  = datasets.get().all_candidates_phase3.parent

def all_candidate_ids(detector_results_path: Path = DETECTOR_RESULTS) -> list[str]:
    """Every candidate id in the merged detector output (all 8 on BPI 2012)."""
    with open(detector_results_path, "r", encoding="utf-8") as f:
        return [c["candidate_id"] for c in json.load(f)]

def apply_redesign(
    candidate_ids: list[str],
    out_dir: str | Path = DEFAULT_OUT_DIR,
    detector_results_path: str | Path = DETECTOR_RESULTS,
    baseline_bpmn_path:    str | Path = BASELINE_BPMN,
    baseline_params_path:  str | Path = BASELINE_PARAMS,
    log_path:              str | Path = LOG_XES,
) -> tuple[Path, Path]:
    """Apply `candidate_ids` to the baseline → (redesigned_bpmn, redesigned_params)."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase3_dir = out_dir / "phase3"

    plan = {"goal": "apply_all",
            "phases": [{"phase_id": 1, "candidates": list(candidate_ids),
                        "rationale": "apply selected candidates to baseline"}]}
    plan_path = out_dir / "plan.json"
    norm_path = out_dir / "norm_stub.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump([], f)

    run_phase3_transformation(
        phase2_plan_path=plan_path,
        normalized_candidates_path=norm_path,
        detector_results_path=Path(detector_results_path),
        baseline_bpmn_path=Path(baseline_bpmn_path),
        baseline_params_path=Path(baseline_params_path),
        log_path=Path(log_path),
        output_dir=phase3_dir,
    )
    return phase3_dir / "redesigned.bpmn", phase3_dir / "redesigned_params.json"

_APPLY_WORKER = """
import json, sys
sys.path.insert(0, sys.argv[1])
from mabpr.apply_redesign import apply_redesign
args = json.loads(sys.argv[2])
bpmn, params = apply_redesign(
    args["candidate_ids"], out_dir=args["out_dir"],
    detector_results_path=args["detector_results_path"],
    baseline_bpmn_path=args["baseline_bpmn_path"],
    baseline_params_path=args["baseline_params_path"],
    log_path=args["log_path"])
print("__RESULT__" + json.dumps([str(bpmn), str(params)]))
"""

def apply_redesign_isolated(
    candidate_ids: list[str],
    out_dir: str | Path = DEFAULT_OUT_DIR,
    detector_results_path: str | Path = DETECTOR_RESULTS,
    baseline_bpmn_path:    str | Path = BASELINE_BPMN,
    baseline_params_path:  str | Path = BASELINE_PARAMS,
    log_path:              str | Path = LOG_XES,
    timeout_s: int = 1800,
    max_attempts: int = 3,
) -> tuple[Path, Path]:
    """Same as :func:`apply_redesign`, in a fresh subprocess, with retries."""

    import subprocess

    payload = json.dumps({
        "candidate_ids": list(candidate_ids), "out_dir": str(out_dir),
        "detector_results_path": str(detector_results_path),
        "baseline_bpmn_path": str(baseline_bpmn_path),
        "baseline_params_path": str(baseline_params_path),
        "log_path": str(log_path),
    })
    cmd = [sys.executable, "-c", _APPLY_WORKER, str(REPO_ROOT), payload]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    last = ""
    for attempt in range(1, max_attempts + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace",
                                    env=env, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            last = f"timed out after {timeout_s}s"
        else:
            marker = next((ln for ln in result.stdout.splitlines()
                           if ln.startswith("__RESULT__")), None)
            if result.returncode == 0 and marker:
                if attempt > 1:
                    print(f"      ✓ apply succeeded on attempt {attempt}/{max_attempts}",
                          flush=True)
                bpmn, params = json.loads(marker[len("__RESULT__"):])
                return Path(bpmn), Path(params)
            last = (f"exit {result.returncode}\n--- stdout ---\n{result.stdout[-1500:]}"
                    f"\n--- stderr ---\n{result.stderr[-1500:]}")
        if attempt < max_attempts:
            print(f"      ! apply attempt {attempt}/{max_attempts} failed, retrying",
                  flush=True)

    raise RuntimeError(f"isolated apply_redesign failed after {max_attempts} attempts "
                       f"for {list(candidate_ids)}: {last}")

if __name__ == "__main__":
    ds = datasets.from_argv(sys.argv)

    flags = {"--dataset", ds.tag}
    ids = [a for a in sys.argv[1:]
           if not a.startswith("--") and a not in flags] or all_candidate_ids(ds.detector_results)
    print(f"dataset {ds.tag}: applying {len(ids)} candidate(s): {ids}")

    bpmn, params = apply_redesign_isolated(
        ids, out_dir=ds.all_candidates_phase3.parent, detector_results_path=ds.detector_results,
        baseline_bpmn_path=ds.baseline_bpmn, baseline_params_path=ds.baseline_params,
        log_path=ds.log_xes)
    print("\n=== Phase-3 done ===")
    print(f"  redesigned BPMN   : {bpmn}   exists={bpmn.exists()}")
    print(f"  redesigned params : {params}   exists={params.exists()}")

def design_fingerprint(phase3_dir) -> str:
    """A hash of the MODEL Phase-3 built — not of the candidate list that asked for it."""

    import hashlib
    import json
    import xml.etree.ElementTree as ET

    d = Path(phase3_dir)
    bpmn, params = d / "redesigned.bpmn", d / "redesigned_params.json"
    if not bpmn.exists() or not params.exists():
        return ""
    tree = ET.parse(bpmn)
    id2name = {el.get("id"): (el.get("name") or el.tag.split("}")[-1])
               for el in tree.iter() if el.get("id")}
    nodes, flows = [], []
    for el in tree.iter():
        tag = el.tag.split("}")[-1]
        if tag in ("task", "userTask", "serviceTask", "subProcess", "exclusiveGateway",
                   "parallelGateway", "inclusiveGateway", "startEvent", "endEvent"):
            nodes.append(f"{tag}:{el.get('name') or ''}")
        elif tag == "sequenceFlow":
            flows.append(f"{id2name.get(el.get('sourceRef'), '?')}>"
                         f"{id2name.get(el.get('targetRef'), '?')}")

    def canon(x):
        """Replace every node id with its name, and sort every list, so two"""

        if isinstance(x, dict):
            return {k: canon(v) for k, v in sorted(x.items())}
        if isinstance(x, list):
            return sorted((canon(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
        if isinstance(x, str):
            return id2name.get(x, x)
        return x

    h = hashlib.sha256()
    h.update("|".join(sorted(nodes)).encode())
    h.update(b"||")
    h.update("|".join(sorted(flows)).encode())
    h.update(b"||")
    h.update(json.dumps(canon(json.loads(params.read_text(encoding="utf-8"))),
                        sort_keys=True).encode())
    return h.hexdigest()[:24]
