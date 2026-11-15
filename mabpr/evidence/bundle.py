"""BUNDLE — the compact evidence packet the Critic reasons over (Option C, v0)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mabpr.evidence.footprint import Footprint
from mabpr.evidence.scorecard import build_scorecard
from mabpr.measure import compare_activities, significant

_SUMMARY_KEYS = [
    ("throughput_h_median",      "throughput_median_h_per_case"),
    ("throughput_h_mean",        "throughput_mean_h_per_case"),
    ("waiting_h_per_case_mean",  "waiting_h_per_case"),
    ("duration_h_per_case_mean", "service_h_per_case"),
    ("utilization_mean",         "resource_utilization"),
]

def _summary(base_agg: dict, redes_agg: dict, n_runs: int) -> dict:
    import math

    out: dict[str, Any] = {"n_runs": n_runs}
    for key, label in _SUMMARY_KEYS:
        b = base_agg.get(f"{key}_mean")
        r = redes_agg.get(f"{key}_mean")
        if b is None or r is None:
            continue
        pct = (r - b) / b * 100 if b else None

        sb = base_agg.get(f"{key}_std") or 0.0
        sr = redes_agg.get(f"{key}_std") or 0.0
        se = math.sqrt((sb ** 2 + sr ** 2) / max(1, n_runs))
        z = None if se == 0 else (r - b) / se
        out[label] = {"base": b, "redes": r, "pct_change": pct,
                      "z": None if z is None else round(z, 2),
                      "significant": bool(z is not None and abs(z) > 3.0)}

    out["noise_floor_h"] = base_agg.get("throughput_h_median_std")
    return out

def _deltas_joined(base_agg: dict, redes_agg: dict, n_runs: int) -> list[dict]:
    """Per-activity rows with BOTH waiting and duration z. Kept if significant in"""

    w = compare_activities(base_agg, redes_agg, n_runs, "waiting_h")
    d = compare_activities(base_agg, redes_agg, n_runs, "duration_h")
    d_by = {x.activity: x for x in d}
    keep = {x.activity for x in significant(w)} | {x.activity for x in significant(d)}

    base_pa = base_agg.get("per_activity", {})
    redes_pa = redes_agg.get("per_activity", {})
    rows: list[dict] = []
    for wr in w:
        if wr.activity not in keep:
            continue
        dr = d_by.get(wr.activity)
        rows.append({
            "activity": wr.activity,
            "status": wr.status,
            "waiting": {"base": wr.base_mean, "redes": wr.redes_mean, "z": wr.z},
            "duration": {"base": dr.base_mean if dr else None,
                         "redes": dr.redes_mean if dr else None,
                         "z": dr.z if dr else None},
            "n": {"base": (base_pa.get(wr.activity) or {}).get("n_mean"),
                  "redes": (redes_pa.get(wr.activity) or {}).get("n_mean")},
        })
    return rows

def build_bundle(base_agg: dict, redes_agg: dict, n_runs: int,
                 footprints: dict[str, Footprint],
                 base_sim_csvs=None, redes_sim_csvs=None,
                 resource_deltas=None, activity_ledger=None,
                 candidates=None, case_endings=None,
                 handler_reports=None) -> dict:
    """Assemble the Critic's evidence packet. Pure/deterministic."""

    fps = {cid: asdict(fp) for cid, fp in footprints.items()}
    deltas = _deltas_joined(base_agg, redes_agg, n_runs)
    return {
        "summary": _summary(base_agg, redes_agg, n_runs),

        "case_endings": case_endings,
        "significant_deltas": deltas,
        "footprints": fps,
        "scorecard": build_scorecard(fps, deltas, base_sim_csvs, redes_sim_csvs,
                                     resource_deltas, n_runs=n_runs,
                                     handler_reports=handler_reports,
                                     activity_ledger=activity_ledger,
                                     candidates=candidates,
                                     baseline_case_hours=base_agg.get(
                                         'throughput_h_mean_mean'),

                                     plan_case_hours=redes_agg.get(
                                         'throughput_h_mean_mean'),

                                     baseline_case_median=base_agg.get(
                                         'throughput_h_median_mean'),
                                     plan_case_median=redes_agg.get(
                                         'throughput_h_median_mean')),
    }
