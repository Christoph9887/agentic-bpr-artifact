"""The datasets the pipeline runs on, in one place."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class Dataset:
    tag: str
    log_xes: Path
    detector_model: Path
    baseline_bpmn: Path
    baseline_params: Path
    detector_results: Path
    starting_at: str
    total_cases: int
    out_dir: Path
    compare_dir: Path

    all_candidates_phase3: Path
    overlay: Path

    domain: str = ""

    def paths(self) -> dict:
        """The dict the graph carries in state."""

        return {
            "dataset": self.tag,
            "compare_dir": str(self.compare_dir),
            "detector_results": str(self.detector_results),
            "baseline_bpmn": str(self.baseline_bpmn),
            "baseline_params": str(self.baseline_params),
            "log_xes": str(self.log_xes),
            "out_dir": str(self.out_dir),
            "overlay": str(self.overlay),
            "domain": self.domain,
        }

def _log_path(stem: str) -> Path:
    """The event log: `<stem>.xes`, or the gzipped form when only that is there."""
    plain = REPO_ROOT / "data" / (stem + ".xes")
    if not plain.exists():
        gz = REPO_ROOT / "data" / (stem + ".xes.gz")
        if gz.exists():
            return gz
    return plain

def _bpi(tag: str, *, starting_at: str, out_dir: Path,
         total_cases: int = 3000, year: str | None = None,
         subset: str = "first_3000_cases") -> Dataset:

    stem = f"BPI_Challenge_{year or tag}_{subset}"
    return Dataset(
        tag=tag,
        log_xes=_log_path(stem),
        detector_model=REPO_ROOT / "inputs" / tag / f"{stem}.bpmn",
        baseline_bpmn=REPO_ROOT / "inputs" / tag / f"{stem}.bpmn",
        baseline_params=REPO_ROOT / "inputs" / tag / f"{stem}.json",
        detector_results=REPO_ROOT / "inputs" / tag / "detected_candidates" / "detector_results.json",
        starting_at=starting_at,
        total_cases=total_cases,
        out_dir=out_dir,
        compare_dir=out_dir / "compare_all",
        all_candidates_phase3=out_dir / "all_candidates_fixture" / "phase3",
        overlay=out_dir / "knowledge_overlay.json",

        domain="a Dutch bank loan process",
    )

def _bpo(tag: str = "bpo",
         stem: str = "BPO_ServiceProcess_first_3000_cases") -> Dataset:
    """A synthetic BPO service process — a THIRD log, and deliberately unlike the"""

    out_dir = REPO_ROOT / "mabpr_out" / tag
    inputs = REPO_ROOT / "inputs" / tag
    return Dataset(
        tag=tag,
        log_xes=_log_path(stem),
        detector_model=inputs / f"{stem}.bpmn",
        baseline_bpmn=inputs / f"{stem}.bpmn",
        baseline_params=inputs / f"{stem}.json",
        detector_results=inputs / "detected_candidates" / "detector_results.json",
        starting_at="2020-01-01T00:00:00Z",
        total_cases=3000,
        out_dir=out_dir,
        compare_dir=out_dir / "compare_all",
        all_candidates_phase3=out_dir / "all_candidates_fixture" / "phase3",
        overlay=out_dir / "knowledge_overlay.json",
    )

DATASETS: dict[str, Dataset] = {
    "2012": _bpi("2012", starting_at="2011-10-01T00:00:00Z",
                 out_dir=REPO_ROOT / "mabpr_out" / "2012"),

    "2017": _bpi("2017", starting_at="2016-01-01T00:00:00Z",
                 out_dir=REPO_ROOT / "mabpr_out" / "2017"),

    "2012_full": _bpi("2012_full", year="2012", subset="full",
                      starting_at="2011-10-01T00:00:00Z",
                      out_dir=REPO_ROOT / "mabpr_out" / "2012_full"),
    "2017_full": _bpi("2017_full", year="2017", subset="full",
                      starting_at="2016-01-01T00:00:00Z",
                      out_dir=REPO_ROOT / "mabpr_out" / "2017_full"),

    "bpo": _bpo(),

    "bpo_full": _bpo("bpo_full", "BPO_ServiceProcess_full"),
}

DEFAULT_DATASET = "2012"

def get(tag: str | None = None) -> Dataset:
    tag = tag or DEFAULT_DATASET
    if tag not in DATASETS:
        raise KeyError(f"unknown dataset {tag!r}; have {sorted(DATASETS)}")
    return DATASETS[tag]

def from_argv(argv: list[str]) -> Dataset:
    """``--dataset 2017`` / ``--dataset=2017``, defaulting to 2012."""
    for i, a in enumerate(argv):
        if a == "--dataset" and i + 1 < len(argv):
            return get(argv[i + 1])
        if a.startswith("--dataset="):
            return get(a.split("=", 1)[1])
    return get()
