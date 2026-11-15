# -*- coding: utf-8 -*-
"""Generate the evaluation section's three LaTeX table bodies from `result.json`."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"

TAG = {"2012_full": "BPI 2012", "2017_full": "BPI 2017", "bpo": "BPO (synthetic)"}
ORDER = ("2012_full", "2017_full", "bpo")

TRAJ_ORDER = ("bpo",)

MAIN_ORDER = ("bpo", "2012_full", "2017_full")

KPI = [("throughput_h_median", "Median case cycle time"),
       ("throughput_h_mean", "Mean case cycle time"),
       ("waiting_h_per_case_mean", "Mean waiting time per case"),
       ("duration_h_per_case_mean", "Mean service time per case"),
       ("utilization_mean", "Mean resource utilisation")]
VERDICT = {"helped": "helped", "hurt": "hurt", "covered": "not established",
           "contended": "contended", "neutral": "neutral",
           "exogenous_only": "log only", "no_sim_signal": "never ran"}

DIMS = [("Activity hours", "throughput"), ("Waiting", "waiting"),
        ("Service", "duration"), ("Utilisation", "resources")]
BOLD = r"\textbf{%s}"
ARROW = {"better": r"$\downarrow$", "worse": r"$\uparrow$", "flat": "$=$",
         "shared": "shared", "n/a": "--", "": "--", None: "--"}

RUN_TAG = {"2012_full": "p3", "2017_full": "q2", "bpo": "n3"}

TRAJ_RUN = {"bpo": "n1"}

RUNS = {"2012_full": ("p1", "p2", "p3"),
        "2017_full": ("q1", "q2", "q3"),
        "bpo": ("n1", "n2", "n3")}

def load_run(tag: str, run: str) -> dict:
    return json.loads((RESULTS / f"{tag}/loop/{run}/result.json")
                      .read_text(encoding="utf-8"))

def best_run(tag: str) -> str:
    """The run with the best median — the one tables 2 and 3 illustrate."""

    def med(run):
        m = (load_run(tag, run)["best"]["all_measures"]
             .get("throughput_h_median") or {})
        return m.get("pct", 0.0)
    return min(RUNS[tag], key=med)

def traj_run(tag: str) -> str:
    """Which run the trajectory table walks through; `RUN_TAG` when unnamed."""
    return TRAJ_RUN.get(tag) or RUN_TAG[tag]

def load(tag: str) -> dict:
    run = RUN_TAG[tag]
    return json.loads((RESULTS / f"{tag}/loop/{run}/result.json")
                      .read_text(encoding="utf-8"))

KIND = {"confirmation": "confirmation", "probe": "subset check"}
RULE = {"better": "satisfied", "mixed": "not satisfied (worse on one measure)",
        "worse": "not satisfied", "unchanged": "not established"}

def _rule(all_measures: dict) -> dict:
    """The deliverable's improvement verdict, from `explain.improvement_rule`."""

    from mabpr.explain import improvement_rule
    return improvement_rule(all_measures or {})

def main_table(rs: dict) -> str:
    """One row per RUN, nine of them, with the candidates each one delivered."""

    B = chr(92)
    L = []
    for i, t in enumerate(MAIN_ORDER):
        if i:
            L.append("    " + B + "midrule")
        runs = [rs[t][r] for r in RUNS[t]]

        group = (B + "multirow{%d}{*}{" % len(RUNS[t]) + B + "textbf{" + TAG[t] + "}}")
        for j, (run, r) in enumerate(zip(RUNS[t], runs), start=1):
            b = r["best"]

            ok = _rule(b["all_measures"])["verdict"] == "better"
            mark = "" if ok else "$^{" + B + "ddagger}$"

            cells = ["   " + (group if j == 1 else ""), str(j) + mark]
            for key, _ in KPI:
                m = (b["all_measures"].get(key) or {})
                pct = m.get("pct")
                cells.append("--" if pct is None else
                             "$%+.1f%s$" % (pct, "^*" if m.get("significant") else ""))
            L.append(" & ".join(cells) + " " + B * 2)
    return chr(10).join(L)

def trajectory_table(nine: dict) -> str:
    """Every measurement one run took, in order, with the delivered one in bold."""

    L = []
    for i, t in enumerate(TRAJ_ORDER):
        r = nine[t][traj_run(t)]
        if i:
            L.append(r"    \midrule")
        rows = list(r["trajectory"])
        names = []
        for x in rows:
            names.append("round %s" % x["round"] if x["mode"] != "confirming"
                         else "round %s %s" % (x["round"],
                                               KIND.get(x.get("kind"),
                                                        "extra measurement")))

        seen = {}
        for n in names:
            seen[n] = seen.get(n, 0) + 1
        used = {}
        for k, n in enumerate(names):
            if seen[n] > 1:
                used[n] = used.get(n, 0) + 1
                names[k] = "%s %d" % (n, used[n])
        for k, (x, name) in enumerate(zip(rows, names)):

            win = bool(x.get("is_best"))
            cells = []
            for key, _ in KPI:
                m = (x.get("measures") or {}).get(key) or {}
                if m.get("pct") is None:
                    cells.append("--")
                    continue
                body = "%+.1f%s" % (m["pct"], "^*" if m.get("significant") else "")
                cells.append(("$\\mathbf{%s}$" % body) if win else ("$%s$" % body))
            shown = (BOLD % name) if win else name

            run_no = "(run %d)" % (RUNS[t].index(traj_run(t)) + 1)
            if len(rows) == 1:
                label = "%s %s" % (TAG[t], run_no)
            else:
                label = TAG[t] if k == 0 else (run_no if k == 1 else "")
            L.append("    %-22s & %s & %d & %s \\\\"
                     % (label, shown, len(x["plan"]), " & ".join(cells)))
    return chr(10).join(L)

def per_candidate_table(rs: dict) -> str:

    L = []
    for i, t in enumerate(MAIN_ORDER):
        r = rs[t]

        delivered = set(r["best"].get("built") or r["best"]["plan"])
        rows = [c for c in (r.get("per_candidate") or [])
                if (c.get("candidate_id") or c.get("candidate")) in delivered]
        rows.sort(key=lambda c: -abs((c.get("contribution") or {}).get("throughput_h_per_case") or 0.0))
        if i:
            L.append(r"    \midrule")
        for j, c in enumerate(rows):
            cid = c.get("candidate_id") or c.get("candidate")
            dims = c.get("critic_dimensions") or {}
            exo = bool((c.get("contribution") or {}).get("exogenous_saving"))
            raw = c.get("critic_verdict")

            cells = [ARROW.get(dims.get(key), "--") for _, key in DIMS]
            tag = f"\\texttt{{{cid}}}" + (r"$^\dagger$" if exo else "")
            L.append(f"    {(TAG[t] if j == 0 else ''):<16} & {tag} & "
                     f"{c.get('heuristic') or '--'} & " + " & ".join(cells) + r" \\")
    return "\n".join(L)

def main() -> None:

    nine = {t: {run: load_run(t, run) for run in RUNS[t]} for t in ORDER}
    for t in ORDER:
        if RUN_TAG.get(t) is None:
            RUN_TAG[t] = best_run(t)
    rs = {t: load(t) for t in ORDER}
    blocks = {"main": main_table(nine), "trajectory": trajectory_table(nine),
              "per_candidate": per_candidate_table(rs)}
    for name, body in blocks.items():
        print()
        print("%% <GEN:" + name + ">")
        print(body)
        print("%% </GEN:" + name + ">")
def _parse_run_overrides(argv) -> None:
    for i, a in enumerate(argv):
        if a == "--run" and i + 1 < len(argv) and "=" in argv[i + 1]:
            ds, run = argv[i + 1].split("=", 1)
            RUN_TAG[ds] = run
        if a == "--traj-run" and i + 1 < len(argv) and "=" in argv[i + 1]:
            ds, run = argv[i + 1].split("=", 1)
            TRAJ_RUN[ds] = run

if __name__ == "__main__":
    _parse_run_overrides(sys.argv)
    print("runs: " + ", ".join(f"{k}={v}" for k, v in RUN_TAG.items())
          + "  |  trajectory: "
          + ", ".join(f"{k}={v}" for k, v in TRAJ_RUN.items()))
    main()
