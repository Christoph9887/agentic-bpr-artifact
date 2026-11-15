"""MEASURE — compare a baseline simulation log against a redesigned one."""

from __future__ import annotations

import math
from dataclasses import dataclass

from mabpr.foundation.simulation_runner import (          # noqa: F401  (re-exported)
    _compute_kpi_from_log as compute_kpi_from_log,
    _aggregate_across_runs as _foundation_aggregate,
)

_EXTRA_SCALARS = ("duration_h_per_case_mean",)

def aggregate_across_runs(kpi_dicts: list) -> dict:
    """foundation's aggregation, plus the scalars this layer defines."""
    from statistics import mean as _mean, pstdev as _pstdev

    agg = _foundation_aggregate(kpi_dicts)

    ends = [d.get("case_endings") or {} for d in kpi_dicts]
    if any(ends):
        keys = sorted({a for e in ends for a in e})
        agg["case_endings_mean"] = {
            a: _mean([float(e.get(a, 0)) for e in ends]) for a in keys}
        agg["case_endings_std"] = {
            a: (_pstdev([float(e.get(a, 0)) for e in ends]) if len(ends) > 1 else 0.0)
            for a in keys}
    for k in _EXTRA_SCALARS:
        vals = [d[k] for d in kpi_dicts if k in d]
        if not vals:
            continue
        agg[f"{k}_mean"] = _mean(vals)
        agg[f"{k}_std"] = _pstdev(vals) if len(vals) > 1 else 0.0
    return agg

Z_THRESHOLD = 3.0

KPI_COLUMNS = ["case_id", "activity", "enable_time", "start_time", "end_time", "resource"]

def kpi_from_csv(log_csv) -> dict:
    """One simulated log on disk -> its KPI dict, without holding the whole frame."""

    import gc

    import pandas as pd

    df = pd.read_csv(log_csv, usecols=KPI_COLUMNS)
    try:
        k = compute_kpi_from_log(df)

        n_cases = max(1, int(k.get("n_cases") or 1))
        k["duration_h_per_case_mean"] = (
            float(k.get("duration_h_mean") or 0.0) * float(k.get("n_events") or 0.0) / n_cases)
        k["case_endings"] = _case_endings(df)
        return k
    finally:
        del df
        gc.collect()

def _case_endings(df) -> dict:
    """{activity: how many cases END on it}."""

    last = (df.sort_values("end_time", kind="mergesort")
              .groupby("case_id")["activity"].last())
    return {str(a): int(n) for a, n in last.value_counts().items()}

def _fold_merged(b, r, merged):
    """Fold a RENAME back together before comparing. Two handlers rename."""

    if not merged:
        return b, r, []

    want: "dict[str, list]" = {}
    for item in merged:
        if isinstance(item, dict):
            composite = str(item.get("composite") or "")
            parts = [str(x) for x in (item.get("parts") or []) if str(x)]
        else:
            composite = str(item)
            parts = [x.strip() for x in composite.split(" + ") if x.strip()]
        if not composite or len(parts) < 1:
            continue
        seen = want.setdefault(composite, [])
        for x in parts:
            if x != composite and x not in seen:
                seen.append(x)
    b, r, folded = dict(b), dict(r), []
    for composite, parts in want.items():
        if composite not in r or composite in b:
            continue
        if any(r.get(x) for x in parts) or not any(x in b for x in parts):
            continue
        gone = [x for x in parts if x in b]
        b[composite] = sum(b.pop(x, 0.0) for x in parts)
        for x in parts:
            r.pop(x, None)
        folded.append({"composite": composite, "parts": gone})
    return b, r, folded

def compare_case_endings(base_agg: dict, redes_agg: dict, n_runs: int,
                         min_cases: float = 10.0, merged=None) -> dict:
    """Which endings changed, with a z each. Facts only -- nothing is ranked."""

    b = base_agg.get("case_endings_mean")
    r = redes_agg.get("case_endings_mean")
    if not b or not r:
        missing = [n for n, v in (("baseline", b), ("redesign", r)) if not v]
        return {"available": False,
                "reason": (f"no case-ending counts for the {' and '.join(missing)} "
                           f"aggregate, so nothing can be compared. Re-run "
                           f"`python -m mabpr.run_compare --dataset <tag>` (or "
                           f"`--refresh-endings`) to add them to comparison.json")}
    bs = base_agg.get("case_endings_std") or {}
    rs = redes_agg.get("case_endings_std") or {}
    b, r, folded = _fold_merged(b, r, merged)

    bs = dict(bs)
    for f in folded:
        bs[f["composite"]] = math.sqrt(sum(float(bs.pop(p, 0.0)) ** 2
                                           for p in f["parts"]))
    rows = []
    for a in sorted(set(b) | set(r)):
        bv, rv = float(b.get(a, 0.0)), float(r.get(a, 0.0))
        if max(bv, rv) < min_cases:
            continue
        se = math.sqrt(((bs.get(a, 0.0) ** 2) + (rs.get(a, 0.0) ** 2)) / max(1, n_runs))
        z = (rv - bv) / se if se else 0.0
        rows.append({"activity": a, "base": round(bv, 1), "redes": round(rv, 1),
                     "delta": round(rv - bv, 1),
                     "pct": round(100.0 * (rv - bv) / bv, 1) if bv else None,
                     "z": round(z, 2), "significant": abs(z) > Z_THRESHOLD})
    sig = [x for x in rows if x["significant"]]
    out = {"rows": rows, "significant": sig, "merges_folded": folded,
           "note": ("How many cases END on each activity. Every reported measure "
                    "is a per-case average, so none of them can see a redesign "
                    "that changes how many cases reach an outcome at all. Which "
                    "ending is desirable is a domain question this system does "
                    "not answer.")}
    if folded:
        out["note"] += (" Activity Composition merged "
                        + "; ".join(f"{f['parts']} into {f['composite']!r}"
                                    for f in folded)
                        + " — those endings are counted together on BOTH sides, "
                        "because a rename is not an outcome shift.")
    if sig:
        moved = sum(abs(x["delta"]) for x in sig) / 2.0
        out["cases_moved_between_endings"] = round(moved, 1)
        out["headline"] = (
            f"{moved:.0f} cases per run end somewhere different: "
            + ", ".join(f"{x['activity']} {x['delta']:+.0f}" for x in sig[:5]))
    return out

@dataclass(frozen=True)
class ActivityDelta:
    """One activity's change on one metric between baseline and redesigned."""

    activity: str
    metric: str
    base_mean: float | None
    redes_mean: float | None
    delta: float | None
    z: float | None
    status: str

    @property
    def significant(self) -> bool:
        return self.z is not None and abs(self.z) > Z_THRESHOLD

def _z(base_mean: float, base_std: float,
       redes_mean: float, redes_std: float, n_runs: int) -> tuple[float, float]:
    """(delta, z). Std-error of the difference pools the two per-run stds over"""

    delta = redes_mean - base_mean
    se = math.sqrt((base_std ** 2 + redes_std ** 2) / n_runs)
    if se == 0.0:
        return delta, (math.inf if delta != 0.0 else 0.0)
    return delta, delta / se

def scalar_z(base_agg: dict, redes_agg: dict, key: str, n_runs: int) -> tuple[float, float]:
    """(delta, z) for a case-level scalar such as ``throughput_h_median``."""

    return _z(base_agg[f"{key}_mean"], base_agg[f"{key}_std"],
              redes_agg[f"{key}_mean"], redes_agg[f"{key}_std"], n_runs)

def compare_activities(base_agg: dict, redes_agg: dict, n_runs: int,
                       metric: str = "waiting_h") -> list[ActivityDelta]:
    """Per-activity deltas on `metric` ('waiting_h' or 'duration_h'), each with a"""

    base = base_agg["per_activity"]
    redes = redes_agg["per_activity"]
    mkey, skey = f"{metric}_mean", f"{metric}_std"

    out: list[ActivityDelta] = []
    for act in sorted(set(base) | set(redes)):
        b, r = base.get(act), redes.get(act)
        if b is not None and r is not None:
            delta, z = _z(b[mkey], b[skey], r[mkey], r[skey], n_runs)
            out.append(ActivityDelta(act, metric, b[mkey], r[mkey], delta, z, "changed"))
        elif b is not None:
            out.append(ActivityDelta(act, metric, b[mkey], None, None, None, "REMOVED"))
        else:
            out.append(ActivityDelta(act, metric, None, r[mkey], None, None, "NEW"))

    return sorted(out, key=lambda d: (0, -abs(d.z)) if d.z is not None else (1, 0.0))

def significant(deltas: list[ActivityDelta]) -> list[ActivityDelta]:
    """Keep only rows worth showing the Critic: |z| > 3, plus every NEW/REMOVED"""

    return [d for d in deltas if d.status != "changed" or d.significant]
