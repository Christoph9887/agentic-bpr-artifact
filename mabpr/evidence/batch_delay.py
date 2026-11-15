"""Split a fitted extraneous-delay timer into its batching part and the rest."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

_PARAM_SCALING = {
    "fix":     ["m"],
    "fixed":   ["m"],
    "expon":   ["m", "m", "m"],
    "norm":    ["m", "m", "m", "m"],
    "uniform": ["m", "m"],
    "lognorm": ["m", "v", "m", "m"],
    "gamma":   ["m", "v", "m", "m"],
}

MIN_Z = 3.0

def scale_distribution(entry: dict, factor: float) -> dict:
    """Return a copy of an ``event_distribution`` entry with the variable scaled."""

    name = str(entry.get("distribution_name", "")).lower()
    layout = _PARAM_SCALING.get(name)
    params = entry.get("distribution_params") or []

    if factor <= 0:
        return {**{k: v for k, v in entry.items()
                   if k not in ("distribution_name", "distribution_params")},
                "distribution_name": "fix",
                "distribution_params": [{"value": 0.0}]}
    if layout is None or len(layout) != len(params):
        raise ValueError(
            f"cannot scale a {name!r} distribution with {len(params)} parameters — "
            f"the layout is unknown, and guessing would silently change what the "
            f"timer means")
    out = dict(entry)
    out["distribution_params"] = [
        {"value": float(p.get("value", 0.0)) * (factor if k == "m" else factor ** 2)}
        for p, k in zip(params, layout)
    ]
    return out

def candidates_without_a_split(catalogue: list) -> list:
    """Case-based Work candidates whose batching share was never computed."""

    return [c.get("candidate_id") for c in (catalogue or [])
            if c.get("heuristic") == "Case-based Work"
            and not (c.get("evidence") or {}).get("extraneous_delay_split")]

def timer_by_activity(bpmn_path: Path, params: dict) -> dict:
    """{activity name: (event_id, fitted mean in hours)} for timers feeding a task."""
    text = Path(bpmn_path).read_text(encoding="utf-8")
    names = dict(re.findall(
        r'<(?:bpmn:)?task[^>]*?id="([^"]+)"[^>]*?name="([^"]*)"', text))
    succ: dict = {}
    for src, tgt in re.findall(r'sourceRef="([^"]+)" targetRef="([^"]+)"', text):
        succ.setdefault(src, []).append(tgt)
    out: dict = {}
    for e in params.get("event_distribution", []) or []:
        prm = e.get("distribution_params") or []
        mean_h = (float(prm[0].get("value", 0.0)) / 3600.0) if prm else 0.0
        for tgt in succ.get(str(e.get("event_id")), []):
            if tgt in names:
                out[names[tgt]] = (str(e.get("event_id")), mean_h)
    return out

def split(baseline_params: str | Path, baseline_bpmn: str | Path) -> dict:
    """Per-activity decomposition of the fitted extraneous delay, FROM THE FILES."""

    import pandas as pd
    from pix_framework.discovery.batch_processing.batch_characteristics import (
        discover_batches)

    params = json.loads(Path(baseline_params).read_text(encoding="utf-8"))
    timers = timer_by_activity(Path(baseline_bpmn), params)

    log = _prepared_log(Path(baseline_params))
    ids = log.attrs["log_ids"]
    batched = discover_batches(log, ids, batch_min_size=3,
                               max_sequential_gap=pd.Timedelta("10m"))
    batched["wait_h"] = (batched[ids.start_time]
                         - batched[ids.enabled_time]).dt.total_seconds() / 3600.0
    batched["is_batched"] = batched[ids.batch_id].notna()

    out: dict = {}
    for activity, (event_id, timer_h) in sorted(timers.items()):
        rows = batched[batched[ids.activity] == activity]
        yes = rows[rows.is_batched]["wait_h"]
        no = rows[~rows.is_batched]["wait_h"]
        rec = {"event_id": event_id, "timer_h": round(timer_h, 4),
               "n": int(len(rows)), "n_batched": int(len(yes))}
        if len(yes) < 3 or len(no) < 3:
            rec.update(batch_component_h=0.0, share_of_timer=0.0,
                       reason="fewer than 3 executions on one side of the split")
            out[activity] = rec
            continue
        delta = float(yes.mean() - no.mean())
        se = math.sqrt(yes.var(ddof=1) / len(yes) + no.var(ddof=1) / len(no))
        z = delta / se if se > 0 else 0.0
        share = len(yes) / len(rows)
        rec.update(batch_share=round(share, 4),
                   wait_batched_h=round(float(yes.mean()), 4),
                   wait_unbatched_h=round(float(no.mean()), 4),
                   delta_h=round(delta, 4), z=round(z, 2))
        if z <= MIN_Z:
            why = ("negative — batches form when a resource frees up and "
                   "clears the queue, which is selection, not a batching cost"
                   if delta < 0 else "inside the noise")
            rec.update(batch_component_h=0.0, share_of_timer=0.0,
                       reason=(f"batched executions do not demonstrably wait "
                               f"longer (delta {delta:+.2f} h at z={z:.2f}); "
                               f"{why}"))
        else:
            comp = min(share * delta, timer_h)
            rec.update(batch_component_h=round(comp, 4),
                       share_of_timer=round(comp / timer_h, 4) if timer_h else 0.0)
        out[activity] = rec
    return out

def _prepared_log(baseline_params: Path):
    """The log with enabled times, grouped the way Simod grouped it when fitting."""

    import pandas as pd
    from simod.event_log.event_log import EventLog
    from simod.settings.simod_settings import SimodSettings

    cfg = baseline_params.parent / "mining_config.yml"
    if not cfg.exists():
        raise SystemExit(
            f"{cfg} is missing. It must be the Simod configuration that produced "
            f"{baseline_params.name}; without it the batch grouping used "
            f"here would not be the one behind the fitted timers.")
    settings = SimodSettings.from_path(cfg)
    ids = settings.common.log_ids
    ev = EventLog.from_path(log_ids=ids,
                            train_log_path=Path(settings.common.train_log_path),
                            test_log_path=None,
                            preprocessing_settings=settings.preprocessing,
                            need_test_partition=False)
    df = ev.train_validation_partition.copy()
    zero = df[ids.start_time] == df[ids.end_time]
    df.loc[zero, ids.start_time] = df.loc[zero, ids.end_time] - pd.Timedelta(milliseconds=1)
    df.attrs["log_ids"] = ids
    return df

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Show how much of each activity's fitted extraneous delay the "
                    "log attributes to batching. Read-only: `run_detectors` is the "
                    "one writer of `extraneous_delay_split`.")
    ap.add_argument("--params", help="baseline params json (mining_config.yml must "
                                     "sit beside it)")
    ap.add_argument("--bpmn", help="the baseline BPMN those params belong to")
    ap.add_argument("--dataset", help="shorthand for a dataset registered in "
                                      "mabpr.datasets, instead of --params/--bpmn")
    args = ap.parse_args()

    if args.params and args.bpmn:
        params_path, bpmn_path = Path(args.params), Path(args.bpmn)
    else:
        from mabpr import datasets
        ds = datasets.get(args.dataset)
        params_path, bpmn_path = Path(ds.baseline_params), Path(ds.baseline_bpmn)
    print(f"{params_path.name} + {bpmn_path.name}")
    result = split(params_path, bpmn_path)

    print("%-32s %6s %8s %8s %9s %10s %8s"
          % ("activity", "n", "batched", "delta h", "z", "timer h", "of timer"))
    for a, r in sorted(result.items(), key=lambda kv: -kv[1]["share_of_timer"]):
        print("%-32s %6d %7.1f%% %8s %9s %10.2f %7.1f%%"
              % (a[:32], r["n"], 100 * r.get("batch_share", 0.0),
                 f"{r['delta_h']:+.2f}" if "delta_h" in r else "-",
                 f"{r['z']:.2f}" if "z" in r else "-",
                 r["timer_h"], 100 * r["share_of_timer"]))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
