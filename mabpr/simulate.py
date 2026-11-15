"""Run a batch of Prosimos replications concurrently."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_WORKERS = min(16, max(1, (os.cpu_count() or 4) - 2))

def workers_from_env(default: int = DEFAULT_WORKERS) -> int:
    """``MABPR_SIM_WORKERS=1`` restores the old serial behaviour."""
    try:
        n = int(os.environ.get("MABPR_SIM_WORKERS", default))
    except ValueError:
        return default
    return max(1, n)

def run_batch(bpmn: Path, params: Path, out_dir: Path, *, n_runs: int,
              total_cases: int, starting_at: str,
              log_pattern: str = "sim_{i}.csv", stat_pattern: str = "stats_{i}.csv",
              workers: int | None = None, quiet: bool = False) -> list[tuple[Path, Path]]:
    """Run ``n_runs`` replications concurrently."""

    from mabpr.foundation.simulation_runner import _run_simulation_isolated

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or workers_from_env()
    paths = [(out_dir / log_pattern.format(i=i), out_dir / stat_pattern.format(i=i))
             for i in range(n_runs)]

    if not quiet:
        print(f"  [simulate] {n_runs} runs x {total_cases} cases, {workers} at a time",
              flush=True)

    def one(i: int) -> int:
        log_csv, stat_csv = paths[i]
        _run_simulation_isolated(bpmn, params, total_cases, log_csv, stat_csv,
                                 starting_at)
        return i

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, i): i for i in range(n_runs)}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if not quiet and (done % 5 == 0 or done == n_runs):
                print(f"  [simulate] {done}/{n_runs}", flush=True)
    return paths
