"""Summarise a finished run's simulation CSVs, then delete them."""

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
from mabpr.measure import (
    aggregate_across_runs, compare_activities, kpi_from_csv, scalar_z, significant,
)

KPI = "throughput_h_median"

def _mb(paths) -> float:
    return sum(p.stat().st_size for p in paths) / 1e6

def summarise(sim_dir: Path, base_agg: dict) -> dict | None:
    """Everything the round measured, small enough to keep forever."""

    csvs = (sorted(sim_dir.glob("sim_*.csv")) or sorted(sim_dir.glob("s_*.csv"))
            or sorted(sim_dir.glob("confirm_sim_*.csv")))
    if not csvs:
        return None
    agg = aggregate_across_runs([kpi_from_csv(f) for f in csvs])
    delta, z = scalar_z(base_agg, agg, KPI, len(csvs))
    out = {
        "n_runs": len(csvs),
        "kpi": {KPI: agg[f"{KPI}_mean"], f"{KPI}_std": agg[f"{KPI}_std"],
                "vs_baseline_pct": delta / base_agg[f"{KPI}_mean"] * 100,
                "vs_baseline_z": z},
        "aggregate": agg,
    }
    for metric in ("waiting_h", "duration_h"):
        sig = significant(compare_activities(base_agg, agg, len(csvs), metric=metric))
        out[f"significant_{metric}"] = [
            {"activity": d.activity, "base": d.base_mean, "redes": d.redes_mean,
             "z": d.z, "status": d.status} for d in sig]
    return out

def run_dirs(out_dir: Path) -> list[Path]:
    """Directories that hold a finished simulation."""

    found = [d for d in sorted((out_dir / "loop").glob("*/*")) if (d / "sim").is_dir()]
    found += [d for d in sorted((out_dir / "loop").glob("*")) if (d / "sim").is_dir()]
    found += [d for d in sorted(out_dir.glob("*")) if (d / "sim").is_dir()]
    return sorted(set(found))

def trim_compare(ds, apply: bool) -> float:
    """The baseline comparison keeps 40 raw CSVs; exactly one of them is ever read."""

    keep = {"_sim_0.csv"}
    doomed = [p for p in sorted(ds.compare_dir.glob("*_sim_*.csv"))
              if not any(p.name.endswith(k) for k in keep)]
    size = _mb(doomed)
    print(f"\ncompare_all: {len(doomed)} unread raw CSVs, {size:.0f} MB "
          f"(keeping *_sim_0.csv, every *_stats_N.csv and comparison.json)")
    if apply:
        for p in doomed:
            p.unlink()
        print("  deleted — regenerate with  python -m mabpr.run_compare "
              f"--dataset {ds.tag}  if case-level baseline detail is ever needed")
    return size

def main(dataset: str | None = None, apply: bool = False,
         compare: bool = False, only: str | None = None) -> None:
    """Compact finished runs. ``only`` restricts to one run id."""

    ds = datasets.get(dataset)
    comparison = ds.compare_dir / "comparison.json"
    if not comparison.exists():
        raise FileNotFoundError(f"{comparison} is missing — nothing to compare against.")
    base_agg = json.loads(comparison.read_text(encoding="utf-8"))["baseline"]

    print(f"dataset {ds.tag}  (baseline {KPI} = {base_agg[f'{KPI}_mean']:.4f} h)")
    print(f"{'MODE: dry run — pass --apply to delete' if not apply else 'MODE: deleting'}\n")

    freed = 0.0
    dirs = run_dirs(ds.out_dir)
    if only:
        dirs = [d for d in dirs if only in d.parts]
        if not dirs:
            raise FileNotFoundError(
                f"no run directory under {ds.out_dir / 'loop'} matches {only!r}; "
                f"have {sorted({p.parent.name for p in run_dirs(ds.out_dir)})}")
        print(f"restricted to run {only!r}: {len(dirs)} round(s)\n")
    for d in dirs:
        sim = d / "sim"

        csvs = [p for p in sorted(sim.glob("*.csv"))
                if not p.name.startswith("stats_") and p.name != "sim_0.csv"]
        if not csvs:
            continue
        size = _mb(csvs)

        label = str(d.relative_to(ds.out_dir))
        s = summarise(sim, base_agg)
        if s is None:
            print(f"  {label:24} {size:8.0f} MB  SKIPPED (no recognised sim CSVs)")
            continue
        (d / "summary.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"  {label:24} {size:8.0f} MB  n={s['n_runs']:2}  "
              f"{s['kpi']['vs_baseline_pct']:+6.1f}%  z={s['kpi']['vs_baseline_z']:+7.1f}  "
              f"-> summary.json ({len(s['significant_waiting_h'])} significant waiting rows)")
        if apply:
            for f in csvs:
                f.unlink()
            for leftover in sim.glob("*"):
                if leftover.name.startswith("stats_") or leftover.name == "sim_0.csv":
                    continue
                try:
                    leftover.unlink()
                except OSError:
                    pass

            if not any(sim.iterdir()):
                try:
                    sim.rmdir()
                except OSError:
                    pass
        freed += size

    if compare:
        freed += trim_compare(ds, apply)

    print(f"\n{'freed' if apply else 'would free'} {freed / 1000:.1f} GB")
    if not apply:
        print("re-run with --apply to delete the CSVs (summary.json files are already written)")

if __name__ == "__main__":
    _only = None
    for _i, _a in enumerate(sys.argv):
        if _a == "--only" and _i + 1 < len(sys.argv):
            _only = sys.argv[_i + 1]
        elif _a.startswith("--only="):
            _only = _a.split("=", 1)[1]
    main(datasets.from_argv(sys.argv).tag, apply="--apply" in sys.argv,
         compare="--compare" in sys.argv, only=_only)
