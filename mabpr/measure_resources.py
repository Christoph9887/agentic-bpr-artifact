"""MEASURE (resource level) — per-resource utilization, base vs redesigned, with z."""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from mabpr.measure import Z_THRESHOLD

_SECTION = "Resource Utilization"

_COLS = {
    "Utilization Ratio": "utilization",
    "Tasks Allocated": "tasks_allocated",
    "Worked Time (seconds)": "worked_time_s",
    "Available Time (seconds)": "available_time_s",
}

@dataclass(frozen=True)
class ResourceDelta:
    """One resource's change between baseline and redesigned."""
    resource: str
    pool: str
    metric: str
    base_mean: float | None
    redes_mean: float | None
    delta: float | None
    z: float | None
    status: str

    @property
    def significant(self) -> bool:
        return self.z is not None and abs(self.z) > Z_THRESHOLD

def parse_resource_stats(stats_csv: str | Path) -> dict[str, dict]:
    """Parse one stats CSV's Resource Utilization section."""

    rows = list(csv.reader(Path(stats_csv).read_text(encoding="utf-8").splitlines()))

    start = next((i for i, r in enumerate(rows) if r and r[0].strip() == _SECTION), None)
    if start is None:
        raise ValueError(f"{stats_csv}: no {_SECTION!r} section")

    header = [h.strip() for h in rows[start + 1]]
    idx = {name: j for j, name in enumerate(header)}
    missing = [c for c in _COLS if c not in idx]
    if missing:
        raise ValueError(f"{stats_csv}: missing columns {missing}; header={header}")

    pool_col = next((c for c in ("Pool name", "Pool ID") if c in idx),
                    next((c for c in header if "pool" in c.lower()), None))

    out: dict[str, dict] = {}
    for row in rows[start + 2:]:
        if not row or not row[0].strip():
            break
        rid = row[idx["Resource ID"]].strip()
        rec: dict = {"pool": row[idx[pool_col]].strip() if pool_col else ""}
        for col, metric in _COLS.items():
            try:
                rec[metric] = float(row[idx[col]])
            except (ValueError, IndexError):
                rec[metric] = float("nan")
        out[rid] = rec
    return out

def aggregate_resource_runs(per_run: list[dict[str, dict]]) -> dict[str, dict]:
    """N runs -> {resource_id: {pool, <metric>_mean, <metric>_std, n_runs_observed}}."""

    ids: set[str] = set()
    for run in per_run:
        ids |= set(run)

    out: dict[str, dict] = {}
    for rid in sorted(ids):
        seen = [run[rid] for run in per_run if rid in run]
        rec: dict = {"pool": seen[0].get("pool", ""), "n_runs_observed": len(seen)}
        for metric in _COLS.values():
            vals = [s[metric] for s in seen if not math.isnan(s.get(metric, float("nan")))]
            rec[f"{metric}_mean"] = statistics.fmean(vals) if vals else float("nan")
            rec[f"{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[rid] = rec
    return out

def load_resource_aggregate(compare_dir: str | Path, prefix: str) -> dict[str, dict]:
    """Aggregate every stats CSV for one side of a comparison."""

    d = Path(compare_dir)
    pattern = f"{prefix}_stats_*.csv" if prefix else "stats_*.csv"
    files = sorted(d.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {d}")
    return aggregate_resource_runs([parse_resource_stats(f) for f in files])

def compare_resources(base_agg: dict[str, dict], redes_agg: dict[str, dict],
                      n_runs: int, metric: str = "utilization") -> list[ResourceDelta]:
    """Per-resource deltas on `metric`, each with a z. Sorted by |z| desc;"""

    mkey, skey = f"{metric}_mean", f"{metric}_std"
    out: list[ResourceDelta] = []
    for rid in sorted(set(base_agg) | set(redes_agg)):
        b, r = base_agg.get(rid), redes_agg.get(rid)
        pool = (b or r).get("pool", "")
        if b is not None and r is not None:
            delta = r[mkey] - b[mkey]
            se = math.sqrt((b[skey] ** 2 + r[skey] ** 2) / n_runs)
            z = (math.inf if delta != 0.0 else 0.0) if se == 0.0 else delta / se
            out.append(ResourceDelta(rid, pool, metric, b[mkey], r[mkey], delta, z, "changed"))
        elif b is not None:
            out.append(ResourceDelta(rid, pool, metric, b[mkey], None, None, None, "REMOVED"))
        else:
            out.append(ResourceDelta(rid, pool, metric, None, r[mkey], None, None, "NEW"))

    return sorted(out, key=lambda d: (0, -abs(d.z)) if d.z is not None else (1, 0.0))
