"""Prosimos KPI comparison — original log vs baseline vs redesigned."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_PROSIMOS_WORKER = (
    "import sys; "
    "from prosimos.simulation_engine import run_simulation; "
    "run_simulation("
    "bpmn_path=sys.argv[1], json_path=sys.argv[2], "
    "total_cases=int(sys.argv[3]), log_out_path=sys.argv[4], "
    "stat_out_path=sys.argv[5], starting_at=sys.argv[6])"
)

def _run_simulation_isolated(
    bpmn_path:    Path,
    params_path:  Path,
    total_cases:  int,
    log_csv:      Path,
    stat_csv:     Path,
    starting_at:  str,
    timeout_s:    int = 3600,
    max_attempts: int = 3,
) -> None:
    """Run a single Prosimos simulation in a fresh subprocess, retrying"""

    cmd = [
        sys.executable, "-c", _PROSIMOS_WORKER,
        str(bpmn_path), str(params_path), str(total_cases),
        str(log_csv), str(stat_csv), starting_at,
    ]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    last_desc, last_out, last_err = "", "", ""
    for attempt in range(1, max_attempts + 1):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                env=env, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            last_desc, last_out, last_err = (
                f"timed out after {timeout_s}s", "", "")
        else:
            if result.returncode == 0:
                if attempt > 1:
                    print(f"      ✓ succeeded on attempt {attempt}/{max_attempts}",
                          flush=True)
                return
            last_desc = f"exit code {result.returncode}"
            last_out, last_err = result.stdout, result.stderr

        if attempt < max_attempts:
            print(f"      ⚠ attempt {attempt}/{max_attempts} failed "
                  f"({last_desc}) — retrying in a fresh subprocess...",
                  flush=True)

    raise RuntimeError(
        f"Prosimos subprocess failed after {max_attempts} attempts "
        f"({last_desc})\n"
        f"  bpmn={bpmn_path}\n  params={params_path}\n"
        f"--- last subprocess stdout ---\n{last_out}\n"
        f"--- last subprocess stderr ---\n{last_err}"
    )

def _compute_kpi_from_log(df: pd.DataFrame) -> dict:
    """Compute summary + per-activity KPIs from one simulated log."""

    df = df.copy()
    df["enable"] = pd.to_datetime(df["enable_time"], utc=True)
    df["start"]  = pd.to_datetime(df["start_time"],  utc=True)
    df["end"]    = pd.to_datetime(df["end_time"],    utc=True)
    df["waiting_h"]  = (df["start"] - df["enable"]).dt.total_seconds() / 3600.0
    df["duration_h"] = (df["end"]   - df["start"]).dt.total_seconds() / 3600.0

    case_thru = df.groupby("case_id").apply(
        lambda g: (g["end"].max() - g["enable"].min()).total_seconds() / 3600.0
    )

    case_wait_sum = df.groupby("case_id")["waiting_h"].sum()
    n_cases = int(df["case_id"].nunique())

    obs_h = (df["end"].max() - df["start"].min()).total_seconds() / 3600.0
    if obs_h <= 0:
        obs_h = float("nan")

    res_util = df.groupby("resource")["duration_h"].sum() / obs_h

    per_activity = df.groupby("activity").agg(
        waiting_h=("waiting_h", "mean"),
        duration_h=("duration_h", "mean"),
        n=("case_id", "count"),
    ).to_dict("index")

    return {
        "n_cases":                   n_cases,
        "n_events":                  int(len(df)),
        "throughput_h_mean":         float(case_thru.mean()),
        "throughput_h_median":       float(case_thru.median()),
        "throughput_h_p90":          float(case_thru.quantile(0.90)),
        "throughput_h_p99":          float(case_thru.quantile(0.99)),
        "throughput_h_max":          float(case_thru.max()),
        "waiting_h_per_event_mean":  float(df["waiting_h"].mean()),
        "waiting_h_per_case_mean":   float(case_wait_sum.mean()),
        "duration_h_mean":           float(df["duration_h"].mean()),
        "utilization_mean":          float(res_util.mean()),
        "obs_window_h":              float(obs_h),
        "per_activity":              per_activity,

        "waiting_h_mean":            float(df["waiting_h"].mean()),
    }

def _kpi_from_original_log(xes_path: Path) -> dict:
    """Compute one KPI dict from the *original* event log."""

    from mabpr.foundation.detectors.feature_extraction import prepare_dataframe

    df = prepare_dataframe(str(xes_path))
    if "org:resource" not in df.columns:
        df["org:resource"] = "UNKNOWN"
    df = df.rename(columns={
        "case:concept:name": "case_id",
        "concept:name":      "activity",
        "start_timestamp":   "start_time",
        "time:timestamp":    "end_time",
        "org:resource":      "resource",
    })[["case_id", "activity", "start_time", "end_time", "resource"]].copy()
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"]   = pd.to_datetime(df["end_time"],   utc=True)

    df = df.sort_values(["case_id", "start_time", "end_time"]).reset_index(drop=True)
    prev_end = df.groupby("case_id")["end_time"].shift(1)
    enable = prev_end.where(prev_end.notna(), df["start_time"])
    df["enable_time"] = enable.where(enable <= df["start_time"], df["start_time"])

    return _compute_kpi_from_log(df)

_SCALAR_KEYS = (
    "n_cases", "n_events",
    "throughput_h_mean", "throughput_h_median",
    "throughput_h_p90", "throughput_h_p99", "throughput_h_max",
    "waiting_h_per_event_mean", "waiting_h_per_case_mean",
    "duration_h_mean",
    "utilization_mean", "obs_window_h",

    "waiting_h_mean",
)

def _mean_std(vals: list[float]) -> tuple[float, float]:
    """Return (mean, std). std uses ddof=1; collapses to 0.0 for n<2."""
    if not vals:
        return float("nan"), 0.0
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s

def _aggregate_across_runs(kpi_dicts: list[dict]) -> dict:
    """Collapse a list of per-run KPI dicts into one aggregated dict"""

    agg: dict[str, Any] = {}
    for k in _SCALAR_KEYS:
        m, s = _mean_std([d[k] for d in kpi_dicts])
        agg[k + "_mean"] = m
        agg[k + "_std"]  = s

    all_acts: set[str] = set()
    for d in kpi_dicts:
        all_acts.update(d["per_activity"].keys())

    pa: dict[str, dict] = {}
    for act in sorted(all_acts):
        w_vals = [d["per_activity"][act]["waiting_h"]
                  for d in kpi_dicts if act in d["per_activity"]]
        d_vals = [d["per_activity"][act]["duration_h"]
                  for d in kpi_dicts if act in d["per_activity"]]
        n_vals = [d["per_activity"][act]["n"]
                  for d in kpi_dicts if act in d["per_activity"]]

        wm, ws = _mean_std(w_vals)
        dm, ds = _mean_std(d_vals)
        nm, _  = _mean_std(n_vals)
        pa[act] = {
            "waiting_h_mean":  wm,
            "waiting_h_std":   ws,
            "duration_h_mean": dm,
            "duration_h_std":  ds,
            "n_mean":          nm,
            "n_runs_observed": len(w_vals),
        }
    agg["per_activity"] = pa
    return agg

def _pct_delta(base: float, redesign: float) -> float | None:
    """Return (redesign − base) / base × 100, or None if base ≈ 0."""
    if base is None or redesign is None:
        return None
    if not np.isfinite(base) or not np.isfinite(redesign):
        return None
    if abs(base) < 1e-9:
        return None
    return float((redesign - base) / base * 100.0)

def _overall_delta(from_agg: dict, to_agg: dict) -> dict:
    """Scalar %-deltas (to − from)/from for each headline KPI."""
    def d(key: str) -> float | None:
        return _pct_delta(from_agg[key + "_mean"], to_agg[key + "_mean"])
    return {
        "throughput_h_pct":              d("throughput_h_mean"),
        "throughput_h_median_pct":       d("throughput_h_median"),
        "throughput_h_p90_pct":          d("throughput_h_p90"),
        "throughput_h_p99_pct":          d("throughput_h_p99"),
        "throughput_h_max_pct":          d("throughput_h_max"),
        "waiting_h_per_event_pct":       d("waiting_h_per_event_mean"),
        "waiting_h_per_case_pct":        d("waiting_h_per_case_mean"),
        "duration_h_pct":                d("duration_h_mean"),
        "utilization_pct":               d("utilization_mean"),

        "waiting_h_pct":                 d("waiting_h_mean"),
    }

_INSTANT_DUR_EPS_H = 1e-4

def _build_comparison(
    original_agg: dict, baseline_agg: dict, redesigned_agg: dict,
) -> dict:
    """Produce the comparison dict."""

    redesigned_vs_baseline = _overall_delta(baseline_agg, redesigned_agg)

    per_act_delta: dict[str, dict] = {}
    all_acts = (set(original_agg["per_activity"].keys())
                | set(baseline_agg["per_activity"].keys())
                | set(redesigned_agg["per_activity"].keys()))
    for act in sorted(all_acts):
        o = original_agg["per_activity"].get(act)
        b = baseline_agg["per_activity"].get(act)
        r = redesigned_agg["per_activity"].get(act)

        b_dur = b["duration_h_mean"] if b else None
        r_dur = r["duration_h_mean"] if r else None
        if (b_dur is not None and r_dur is not None
                and b_dur < _INSTANT_DUR_EPS_H and r_dur < _INSTANT_DUR_EPS_H):
            duration_h_pct = None
        else:
            duration_h_pct = _pct_delta(b_dur, r_dur)

        per_act_delta[act] = {
            "in_original":      o is not None,
            "in_baseline":      b is not None,
            "in_redesigned":    r is not None,
            "waiting_h_pct":    _pct_delta(b["waiting_h_mean"]  if b else None,
                                           r["waiting_h_mean"]  if r else None),
            "duration_h_pct":   duration_h_pct,
        }
    return {
        "redesigned_vs_baseline": redesigned_vs_baseline,
        "per_activity":           per_act_delta,
    }

def _fmt_mean_std(m: float, s: float, unit: str = "h", precision: int = 3) -> str:
    if m is None or not np.isfinite(m):
        return "—"
    return f"{m:.{precision}f}±{s:.{precision}f} {unit}".strip()

def _fmt_pct(p: float | None) -> str:
    if p is None:
        return "—"
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.2f}%"

def _write_markdown(
    comparison: dict,
    original_agg: dict,
    baseline_agg: dict,
    redesigned_agg: dict,
    md_path: Path,
    config: dict,
) -> None:
    """Write a paper-ready Markdown report (summary + per-activity tables)"""

    rvb = comparison["redesigned_vs_baseline"]

    lines: list[str] = []
    lines.append("# Prosimos KPI Comparison — Original vs Baseline vs Redesigned\n")
    lines.append("## Config\n")
    for k, v in config.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    lines.append("## Summary KPIs (mean ± std across runs)\n")
    lines.append(
        "> **Original log** is shown for reference only. It measures real "
        "calendar time (overnight / weekend / dormant gaps included), while "
        "the simulated columns use the Simod `resource_model`-stage baseline "
        "which omits extraneous-delay timers by design (see `plan.txt §8.0`). "
        "The two are not on the same measurement basis, so the only delta "
        "reported is the methodologically clean **Δ Redes/Base** (redesigned "
        "vs baseline — both simulated identically, differing only in the "
        "Phase 3 edits).\n"
    )
    lines.append("| Metric | Original log | Baseline (sim) | Redesigned (sim) | Δ Redes/Base |")
    lines.append("|---|---|---|---|---|")
    summary_specs = [
        ("Throughput mean / case",       "throughput_h_mean",        "throughput_h_pct",         "h", 3),
        ("Throughput median / case",     "throughput_h_median",      "throughput_h_median_pct",  "h", 3),
        ("Throughput p90 / case",        "throughput_h_p90",         "throughput_h_p90_pct",     "h", 3),
        ("Throughput p99 / case",        "throughput_h_p99",         "throughput_h_p99_pct",     "h", 3),
        ("Throughput max / case",        "throughput_h_max",         "throughput_h_max_pct",     "h", 3),
        ("Avg waiting / case",           "waiting_h_per_case_mean",  "waiting_h_per_case_pct",   "h", 3),
        ("Avg waiting / event",          "waiting_h_per_event_mean", "waiting_h_per_event_pct",  "h", 3),
        ("Avg duration / event",         "duration_h_mean",          "duration_h_pct",           "h", 3),
        ("Avg resource utilization",     "utilization_mean",         "utilization_pct",          "",  4),
    ]
    for label, agg_key, delta_key, unit, prec in summary_specs:
        lines.append("| " + " | ".join([
            label,
            _fmt_mean_std(original_agg[agg_key + "_mean"],
                          original_agg[agg_key + "_std"], unit, prec),
            _fmt_mean_std(baseline_agg[agg_key + "_mean"],
                          baseline_agg[agg_key + "_std"], unit, prec),
            _fmt_mean_std(redesigned_agg[agg_key + "_mean"],
                          redesigned_agg[agg_key + "_std"], unit, prec),
            _fmt_pct(rvb[delta_key]),
        ]) + " |")
    lines.append("| " + " | ".join([
        "Cases (observed / simulated)",
        f"{original_agg['n_cases_mean']:.0f}",
        f"{baseline_agg['n_cases_mean']:.0f}",
        f"{redesigned_agg['n_cases_mean']:.0f}",
        "—",
    ]) + " |")
    lines.append("")

    def _wait(x, missing):
        return _fmt_mean_std(x["waiting_h_mean"], x["waiting_h_std"]) if x else missing

    def _dur(x, missing):
        return _fmt_mean_std(x["duration_h_mean"], x["duration_h_std"]) if x else missing

    lines.append("## Per-Activity Breakdown\n")
    lines.append("Δ columns are redesigned-vs-baseline (the redesign effect).\n")
    lines.append("| Activity | Orig wait | Base wait | Redes wait | Δ wait | "
                 "Orig dur | Base dur | Redes dur | Δ dur |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for act, delta in comparison["per_activity"].items():
        o = original_agg["per_activity"].get(act)
        b = baseline_agg["per_activity"].get(act)
        r = redesigned_agg["per_activity"].get(act)
        lines.append("| " + " | ".join([
            act,
            _wait(o, "—"), _wait(b, "—"), _wait(r, "(removed)"),
            _fmt_pct(delta["waiting_h_pct"]),
            _dur(o, "—"), _dur(b, "—"), _dur(r, "(removed)"),
            _fmt_pct(delta["duration_h_pct"]),
        ]) + " |")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

def run_prosimos_comparison(
    baseline_bpmn_path:     str | Path,
    baseline_params_path:   str | Path,
    redesigned_bpmn_path:   str | Path,
    redesigned_params_path: str | Path,
    original_log_path:      str | Path,
    output_dir:             str | Path,
    total_cases:            int  = 1000,
    n_runs:                 int  = 5,
    starting_at:            str  = "2011-10-01T00:00:00Z",
) -> dict:
    """Evaluate three groups — original log, baseline sim, redesigned sim."""

    baseline_bpmn_path     = Path(baseline_bpmn_path)
    baseline_params_path   = Path(baseline_params_path)
    redesigned_bpmn_path   = Path(redesigned_bpmn_path)
    redesigned_params_path = Path(redesigned_params_path)
    original_log_path      = Path(original_log_path)
    output_dir             = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Prosimos KPI Comparison ===")
    print(f"  Original   : {original_log_path.name} (read once)")
    print(f"  Baseline   : {baseline_bpmn_path.name} + {baseline_params_path.name}")
    print(f"  Redesigned : {redesigned_bpmn_path.name} + {redesigned_params_path.name}")
    print(f"  Output     : {output_dir}")
    print(f"  total_cases={total_cases}, n_runs={n_runs}, starting_at={starting_at}")

    print(f"\n[original] reading {original_log_path.name} + deriving enable_time via pm4py...")
    original_kpi = _kpi_from_original_log(original_log_path)
    original_agg = _aggregate_across_runs([original_kpi])
    print(f"      n_cases={original_kpi['n_cases']}  "
          f"throughput_h={original_kpi['throughput_h_mean']:.2f}  "
          f"waiting_h={original_kpi['waiting_h_mean']:.3f}  "
          f"util={original_kpi['utilization_mean']:.3f}")

    configs = [
        ("baseline",   baseline_bpmn_path,   baseline_params_path),
        ("redesigned", redesigned_bpmn_path, redesigned_params_path),
    ]
    per_cfg_kpi: dict[str, list[dict]] = {"baseline": [], "redesigned": []}

    for cfg_name, bpmn, params in configs:
        print(f"\n[{cfg_name}] running Prosimos × {n_runs}...")
        for i in range(n_runs):
            log_csv  = output_dir / f"{cfg_name}_simulated_{i}.csv"
            stat_csv = output_dir / f"{cfg_name}_stats_{i}.csv"
            print(f"  [{i+1}/{n_runs}] → {log_csv.name}")
            _run_simulation_isolated(
                bpmn, params, total_cases, log_csv, stat_csv, starting_at,
            )

            df = pd.read_csv(log_csv)
            kpi = _compute_kpi_from_log(df)
            per_cfg_kpi[cfg_name].append(kpi)
            print(f"      n_cases={kpi['n_cases']}  "
                  f"throughput_h={kpi['throughput_h_mean']:.2f}  "
                  f"waiting_h={kpi['waiting_h_mean']:.3f}  "
                  f"util={kpi['utilization_mean']:.3f}")

    print(f"\n[aggregate] computing mean±std across {n_runs} runs per simulated config...")
    baseline_agg   = _aggregate_across_runs(per_cfg_kpi["baseline"])
    redesigned_agg = _aggregate_across_runs(per_cfg_kpi["redesigned"])

    comparison = _build_comparison(original_agg, baseline_agg, redesigned_agg)
    config_dict = {
        "total_cases":      total_cases,
        "n_runs":           n_runs,
        "starting_at":      starting_at,
        "original_log":     str(original_log_path),
        "baseline_bpmn":    str(baseline_bpmn_path),
        "baseline_params":  str(baseline_params_path),
        "redesigned_bpmn":  str(redesigned_bpmn_path),
        "redesigned_params":str(redesigned_params_path),
    }

    payload = {
        "config":                       config_dict,
        "original":                     original_agg,
        "baseline":                     baseline_agg,
        "redesigned":                   redesigned_agg,
        "delta_redesigned_vs_baseline": comparison["redesigned_vs_baseline"],
        "per_activity_delta":           comparison["per_activity"],
    }
    json_path = output_dir / "kpi_comparison.json"
    md_path   = output_dir / "kpi_comparison.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    _write_markdown(comparison, original_agg, baseline_agg, redesigned_agg,
                    md_path, config_dict)

    print(f"\n✓ Comparison complete.")
    print(f"  → {json_path.name}")
    print(f"  → {md_path.name}")
    print(f"\n(Original log shown for reference only — not on the same "
          f"measurement basis as the simulated groups; see the report.)")
    print(f"\nΔ redesigned vs baseline (redesign effect):")
    for k, v in comparison["redesigned_vs_baseline"].items():
        print(f"  {k:24s}: {_fmt_pct(v)}")

    return payload

ORIGINAL_LOG      = REPO_ROOT / "data" / "BPI_Challenge_2017_first_3000_cases.xes"
BASELINE_BPMN     = REPO_ROOT / "simulation" / "redesigned" / "filtered_balanced" / "baseline.bpmn"
BASELINE_PARAMS   = REPO_ROOT / "simulation" / "redesigned" / "filtered_balanced" / "baseline_params_for_cbw.json"
REDESIGNED_BPMN   = REPO_ROOT / "simulation" / "redesigned" / "filtered_balanced" / "redesigned.bpmn"
REDESIGNED_PARAMS = REPO_ROOT / "simulation" / "redesigned" / "filtered_balanced" / "redesigned_params.json"
OUTPUT_DIR        = REPO_ROOT / "simulation" / "redesigned" / "filtered_balanced" / "kpi"

TOTAL_CASES       = 3000
N_RUNS            = 20
STARTING_AT       = "2011-10-01T00:00:00Z"

if __name__ == "__main__":
    try:
        run_prosimos_comparison(
            BASELINE_BPMN, BASELINE_PARAMS,
            REDESIGNED_BPMN, REDESIGNED_PARAMS,
            ORIGINAL_LOG,
            OUTPUT_DIR,
            total_cases=TOTAL_CASES,
            n_runs=N_RUNS,
            starting_at=STARTING_AT,
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
