# -*- coding: utf-8 -*-
"""Reproduce everything the paper reports from the run records in `results/`.

One command, no model call and no simulation:

    python reproduce_results.py

It prints the nine runs the paper reports with all five key performance
indicators, and the three result tables exactly as the paper carries them.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import gen_tables as gt

RULE = {"better": "satisfied",
        "mixed": "NOT satisfied (significantly worse on one measure)",
        "worse": "NOT satisfied", "unchanged": "not established"}


def runs() -> None:
    """The nine runs, each with the five indicators and the improvement rule."""

    print("\nTHE NINE RUNS")
    print("  Every percentage is against that dataset's own simulated baseline; "
          "`*` marks\n  |z| > 3. The improvement rule is the pre-specified "
          "stopping criterion:\n  significantly better on at least one "
          "directional measure and significantly\n  worse on none. Mean "
          "resource utilisation has no direction and can neither\n  create nor "
          "block the verdict.\n")
    for tag in gt.MAIN_ORDER:
        print("  %s" % gt.TAG[tag])
        medians = []
        for run in gt.RUNS[tag]:
            r = gt.load_run(tag, run)
            b = r["best"]
            rule = gt._rule(b["all_measures"])["verdict"]
            cells = []
            for key, _ in gt.KPI:
                m = b["all_measures"].get(key) or {}
                pct = m.get("pct")
                cells.append("     --  " if pct is None else
                             "%+7.2f%s" % (pct, "*" if m.get("significant") else " "))
            med = (b["all_measures"].get("throughput_h_median") or {}).get("pct")
            medians.append(None if med is None else round(med, 2))
            print("    %-3s %s  | rule %-14s | gain %s"
                  % (run, " ".join(cells), RULE.get(rule, rule),
                     (b.get("gain_source") or {}).get("verdict", "?")))
            print("        plan %2d candidate(s), %2d of them changed the model: %s"
                  % (len(b.get("plan") or []), len(b.get("built") or b.get("plan") or []),
                     ", ".join(b.get("built") or b.get("plan") or [])))
        got = [m for m in medians if m is not None]
        if len(got) > 1:
            print("    median spread across the three runs: %.2f percentage points\n"
                  % (max(got) - min(got)))
    print("  columns: " + " | ".join(label for _, label in gt.KPI))


def tables() -> None:
    """The three table bodies, as `04-evaluation.tex` carries them."""

    nine = {t: {run: gt.load_run(t, run) for run in gt.ORDER and gt.RUNS[t]}
            for t in gt.ORDER}
    for t in gt.ORDER:
        if gt.RUN_TAG.get(t) is None:
            gt.RUN_TAG[t] = gt.best_run(t)
    rs = {t: gt.load(t) for t in gt.ORDER}
    print("\nTHE PAPER'S TABLES")
    print("  Table 1 reports all nine runs; Table 2 walks through every "
          "measurement of\n  %s; Table 3 gives the per-candidate account of "
          "the best-median run of each\n  dataset (%s)."
          % (gt.TRAJ_RUN.get("bpo", "?"),
             ", ".join("%s=%s" % (t, gt.RUN_TAG[t]) for t in gt.ORDER)))
    for name, body in (("main", gt.main_table(nine)),
                       ("trajectory", gt.trajectory_table(nine)),
                       ("per_candidate", gt.per_candidate_table(rs))):
        print("\n%% <GEN:%s>" % name)
        print(body)
        print("%% </GEN:%s>" % name)


def main() -> None:
    runs()
    tables()
    print("\nWhere every other figure in the paper comes from:")
    print("  results/<dataset>/loop/<run>/result.json   the deliverable, its five "
          "indicators,\n      the per-candidate account, the trajectory and the "
          "selector's reading")
    print("  results/<dataset>/loop/<run>/round<N>/     per round: the plan, the "
          "compliance\n      ruling, the explanation, the full LLM trace, and "
          "the measurement summary")


if __name__ == "__main__":
    main()
