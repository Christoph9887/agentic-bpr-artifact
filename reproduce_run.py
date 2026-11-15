# -*- coding: utf-8 -*-
"""Run the whole thing from scratch on one dataset: detect, measure, redesign.

One command:

    python reproduce_run.py --dataset bpo

The inputs are three files and nothing else -- an event log, a process model and
a simulation model fitted to that log:

    data/<name>.xes            the event log
    data/<name>.bpmn           the process model the detectors read
    inputs/<tag>/<name>.bpmn   the same model, as the simulator reads it
    inputs/<tag>/<name>.json   the simulation model (resources, calendars,
                               branching, durations) fitted to that log

From those it runs the three stages the paper describes, in order:

    1. detection      every heuristic is detected from the log and the model,
                      and the result is written to
                      inputs/<tag>/detected_candidates/
    2. the baseline   the unchanged simulation model is simulated 20 times. That
                      is the one fixed basis every later measurement is scored
                      against, so it is measured before any redesign exists
    3. the closed loop the compliance gate, the three agents, apply, simulate and
                      measure, round after round until the pre-specified
                      stopping rule ends the run

Stage 3 needs an OpenAI-compatible key in OPENAI_API_KEY. Stages 2 and 3 run
Prosimos many times: expect roughly half an hour for the synthetic log and
several hours for a BPI log.

Options:
    --dataset bpo|2012_full|2017_full   which dataset (default bpo)
    --tag NAME                          name the run directory (default: a date)
    --max-rounds N                      the round cap (default 12)
    --workers N                         parallel Prosimos processes (default 8)
    --skip-detection, --skip-baseline   reuse what stage 1 or 2 left on disk
"""

from __future__ import annotations

import os
import subprocess
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

PACKAGES = ["pm4py", "prosimos", "simod", "pix_framework", "pandas", "numpy",
            "networkx", "langgraph", "openai", "dotenv"]


def _arg(name, default=None):
    """The value after a flag on the command line, or a default."""
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def preflight(tag, need_key):
    """Refuse to start unless the packages, the key and the three inputs are there."""

    from mabpr import datasets
    ds = datasets.get(tag)
    problems = []

    missing = []
    for mod in PACKAGES:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        problems.append("missing Python packages: " + ", ".join(missing)
                        + " -- install the pinned versions with "
                          "`pip install -r requirements.txt`")

    roles = [("event log", ds.log_xes),
             ("process model", ds.detector_model),
             ("model, as simulated", ds.baseline_bpmn),
             ("simulation model", ds.baseline_params)]
    inputs = []
    for label, p in roles:
        for seen in inputs:
            if seen[1] == p:
                seen[0].append(label)
                break
        else:
            inputs.append(([label], p))
    print("INPUTS")
    for labels, p in inputs:
        there = p.exists()
        if not there:
            size = "NOT FOUND"
        elif p.stat().st_size >= 1e6:
            size = "%.1f MB" % (p.stat().st_size / 1e6)
        else:
            size = "%.0f KB" % (p.stat().st_size / 1e3)
        print("  %-26s %-50s %s"
              % (" / ".join(labels), p.relative_to(ROOT).as_posix(), size))
        if not there:
            problems.append("%s not found: %s"
                            % (labels[0], p.relative_to(ROOT).as_posix()))
    if need_key and not os.environ.get("OPENAI_API_KEY"):
        problems.append(
            "OPENAI_API_KEY is not set, and the closed loop calls the model "
            "8 to 18 times per round. Put your own key in the environment, or "
            "in a .env file at the root of this artifact (see .env.example).")

    if problems:
        print("\nCANNOT START")
        for p in problems:
            print("  - %s" % p)
        raise SystemExit(1)
    print("  environment ok: " + ", ".join(PACKAGES))
    return ds


def stage(n, what, args, env):
    """Run one stage as its own process, and stop everything if it fails."""

    print("\n" + "=" * 78)
    print("STAGE %d  %s" % (n, what))
    print("  python " + " ".join(args))
    print("=" * 78, flush=True)
    rc = subprocess.call([sys.executable] + args, cwd=str(ROOT), env=env)
    if rc != 0:
        raise SystemExit("stage %d failed (exit %d): %s" % (n, rc, what))


def main():
    tag = _arg("--dataset", "bpo")
    run_tag = _arg("--tag")
    rounds = _arg("--max-rounds", "12")
    workers = _arg("--workers", "8")

    print("ONE RUN FROM SCRATCH, DATASET %s\n" % tag)
    ds = preflight(tag, need_key=True)

    env = dict(os.environ)
    env["MABPR_SIM_WORKERS"] = workers
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print("  %s Prosimos processes in parallel (--workers). Each one imports "
          "pandas and\n  numpy, so lower it on a machine with little memory: "
          "what binds is committed\n  memory, not cores." % workers)

    rel = lambda p: p.relative_to(ROOT).as_posix()

    if "--skip-detection" in sys.argv:
        print("\n[stage 1 skipped, reusing %s]" % rel(ds.detector_results))
    else:
        stage(1, "detect the candidates from the log and the model  ->  %s"
                 % rel(ds.detector_results.parent),
              ["-m", "mabpr.run_detectors", "--dataset", tag], env)

    if "--skip-baseline" in sys.argv:
        print("\n[stage 2 skipped, reusing %s]"
              % rel(ds.compare_dir / "comparison.json"))
    else:
        stage(2, "simulate the unchanged model 20 times  ->  %s"
                 % rel(ds.compare_dir),
              ["-m", "mabpr.run_compare", "--dataset", tag], env)

    loop = ["-m", "mabpr.graph", "--run", "--dataset", tag, "--max-rounds", rounds]
    if run_tag:
        loop += ["--tag", run_tag]
    stage(3, "the closed loop: gate, agents, apply, simulate, measure", loop, env)

    print("\n" + "=" * 78)
    print("DONE")
    print("  the run wrote its own records under %s" % rel(ds.out_dir / "loop"))
    print("  `result.json` there carries the delivered plan, all five key "
          "performance\n  indicators with their significance, the per-candidate "
          "account and the\n  trajectory, in the same shape as the nine runs "
          "under results/")
    print("\n  to put it beside what the paper reports:")
    print("    results/comparison_%s.json      the baseline the paper measured "
          "against, so\n      stage 2 can be checked against it" % tag)
    print("    results/%s/loop/*/result.json   the three runs the paper reports "
          "on this\n      dataset" % tag)
    print("  Three runs of this command will not agree exactly. The paper "
          "reports the\n  spread of three (15.06, 0.41 and 0.71 percentage "
          "points of the median on the\n  synthetic, the 2012 and the 2017 log) "
          "and names the three steps whose\n  outcome depends on a model call: "
          "the compliance ruling, which hypothesis the\n  Critic spends a "
          "confirmation on, and the choice among the measurements.")


if __name__ == "__main__":
    main()
