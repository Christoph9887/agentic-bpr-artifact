"""Measure THE BASELINE: N Prosimos runs of the raw uncalibrated Simod model,"""

from __future__ import annotations

import json
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

from mabpr import datasets
from mabpr.simulate import run_batch

N_RUNS = 20

def _simulate_n(name: str, bpmn: Path, params: Path, out_dir: Path,
                total_cases: int, starting_at: str) -> dict:
    """Run Prosimos N_RUNS times, return the cross-run aggregate. Keeps every"""

    print(f"  [{name}]", flush=True)
    paths = run_batch(bpmn, params, out_dir, n_runs=N_RUNS, total_cases=total_cases,
                      starting_at=starting_at,
                      log_pattern=f"{name}_sim_{{i}}.csv",
                      stat_pattern=f"{name}_stats_{{i}}.csv")

    from mabpr.measure import aggregate_across_runs, kpi_from_csv
    kpis = [kpi_from_csv(log_csv) for log_csv, _ in paths]
    return aggregate_across_runs(kpis)

def refresh(dataset: str | None = None) -> None:
    """Recompute the stored baseline aggregate from the runs already on disk."""

    from mabpr.measure import aggregate_across_runs, kpi_from_csv

    ds = datasets.get(dataset)
    out_json = ds.compare_dir / "comparison.json"
    if not out_json.exists():
        raise SystemExit(f"{out_json} does not exist; run `run_compare` first")
    sims = sorted(ds.compare_dir.glob("baseline_sim_*.csv"))
    if not sims:
        raise SystemExit(
            f"no baseline_sim_*.csv under {ds.compare_dir} — they were compacted "
            f"away, so the aggregate cannot be recomputed. Re-run "
            f"`python -m mabpr.run_compare --dataset {ds.tag}`")
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    before = sorted(payload.get("baseline") or {})
    payload["baseline"] = aggregate_across_runs([kpi_from_csv(f) for f in sims])
    payload.setdefault("config", {})["n_runs"] = len(sims)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
    added = sorted(set(payload["baseline"]) - set(before))
    print(f"recomputed from {len(sims)} runs on disk -> {out_json}")
    print(f"  keys added: {added or 'none'}")

def main(dataset: str | None = None) -> None:
    ds = datasets.get(dataset)
    BASELINE_BPMN, BASELINE_PARAMS = ds.baseline_bpmn, ds.baseline_params
    OUT_DIR = ds.compare_dir
    TOTAL_CASES = ds.total_cases

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"dataset  = {ds.tag}  (starting_at={ds.starting_at}, cases={TOTAL_CASES})")
    print(f"baseline = {BASELINE_BPMN.name} + {BASELINE_PARAMS.name} (RAW, uncalibrated)\n")

    base_agg = _simulate_n("baseline", BASELINE_BPMN, BASELINE_PARAMS, OUT_DIR,
                           TOTAL_CASES, ds.starting_at)

    print("\n" + "=" * 70)
    print(f"BASELINE (mean ± std across {N_RUNS} runs) — every later z divides by this:")
    for key, label in [("throughput_h_median", "throughput median /case"),
                       ("throughput_h_mean",   "throughput mean   /case"),
                       ("utilization_mean",    "resource utilization")]:
        print(f"  {label:26} {base_agg[f'{key}_mean']:9.4f}  ± {base_agg[f'{key}_std']:.4f}")

    per_act = base_agg.get("per_activity") or {}
    print(f"\n  per-activity rows: {len(per_act)}  (waiting + duration, each mean ± std)")

    payload = {
        "config": {"dataset": ds.tag, "total_cases": TOTAL_CASES, "n_runs": N_RUNS,
                   "starting_at": ds.starting_at},
        "baseline": base_agg,
    }
    out_json = OUT_DIR / "comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ written → {out_json}")

if __name__ == "__main__":
    if "--refresh" in sys.argv:
        refresh(datasets.from_argv(sys.argv).tag)
    else:
        main(datasets.from_argv(sys.argv).tag)
