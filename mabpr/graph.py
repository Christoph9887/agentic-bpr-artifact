"""The compiled LangGraph — and therefore the architecture figure."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from mabpr import datasets
from mabpr.agents.arbiter import ArbiterBox, calibrate, gate, target_survival
from mabpr.agents.critic import (conclude, confirmation_gate, investigate,
                                 removal_fact_lines,
                                 normalize_directive, orient)
from mabpr.agents.proposer import propose, unbuildable
from mabpr.evidence.batch_delay import candidates_without_a_split
from mabpr.evidence.bundle import build_bundle
from mabpr.evidence.footprint import read_footprints, rename_records
from mabpr.evidence.scorecard import (MAX_REMOVAL_FACTS, _isolation,
                                      ledger_rows, measured_removals,
                                      unmeasured_removals)
from mabpr.evidence.tools import ToolBox
from mabpr.llm.client import set_trace_sink
from mabpr.explain import (explain_round, improvement_rule, render_final,
                           write_final_report, write_round_explanation)
from mabpr.knowledge.overlay import KnowledgeOverlay
from mabpr.knowledge.tools import KnowledgeBox
from mabpr.measure import (aggregate_across_runs, compare_case_endings,
                           kpi_from_csv, scalar_z)
from mabpr.measure_resources import compare_resources, load_resource_aggregate
from mabpr.state import MABPRState

KPI = "throughput_h_median"

REPORT_KPIS = [
    ("throughput_h_median", "median h/case", "case span, the convergence KPI"),
    ("throughput_h_mean", "mean h/case", "case span, additive"),
    ("waiting_h_per_case_mean", "waiting h/case", "queueing a case sits through"),
    ("duration_h_per_case_mean", "service h/case", "work actually performed"),
    ("utilization_mean", "utilisation", "share of the window resources are busy"),
]
IMPROVE_Z = -3.0
PATIENCE = 2

HARMFUL_VERDICTS = ("hurt",)

SimulateFn = Callable[[Path, Path, Path, dict], list[dict]]

def prosimos_simulate(bpmn: Path, params: Path, out_dir: Path, cfg: dict) -> list[dict]:
    """The real thing: N isolated Prosimos runs -> per-run KPIs."""

    from mabpr.simulate import run_batch

    paths = run_batch(bpmn, params, out_dir,
                      n_runs=int(cfg.get("n_runs", 20)),
                      total_cases=int(cfg.get("total_cases", 3000)),
                      starting_at=cfg.get("starting_at"))

    return [kpi_from_csv(log_csv) for log_csv, _ in paths]

def round_dir(state: MABPRState) -> "Path | None":
    """This ROUND's own directory — never the confirmation's."""

    p = state.get("paths") or {}
    if not p.get("out_dir"):
        return None
    return (Path(p["out_dir"]) / "loop" / p.get("run_id", "run")
            / f"round{state.get('round', 1)}")

def evidence_sim_dir(state: MABPRState) -> "Path | None":
    """Where the RAW SIMULATION LOGS behind the current evidence live."""

    p = state.get("paths") or {}
    return Path(p["active_sim"]) if p.get("active_sim") else None

def _toolbox(state: MABPRState) -> ToolBox:
    p = state["paths"]
    sim = evidence_sim_dir(state)
    return ToolBox(compare_dir=p["compare_dir"], phase3_dir=_active_phase3(state),
                   detector_results_path=p["detector_results"],
                   n_runs=int(state.get("sim_config", {}).get("n_runs", 20)),

                   sim_dir=sim,

                   ledger=state.get("candidate_ledger"))

def _round_phase3(state: MABPRState) -> str | None:
    """What THIS ROUND's plan built — never the confirmation's."""

    p = state["paths"]
    return p.get("round_phase3") or p.get("active_phase3")

def _active_phase3(state: MABPRState) -> str | None:
    """The Phase-3 output the current evidence belongs to, or None before the"""

    return state["paths"].get("active_phase3")

def apply_node(state: MABPRState) -> dict:
    """APPLY REDESIGN (Phase-3). Constructive ground truth: building the plan is"""

    from mabpr.apply_redesign import apply_redesign_isolated as apply_redesign

    p = state["paths"]
    confirming = state.get("mode") == "confirming"
    plan = state["confirmation_plan"] if confirming else state["current_plan"]
    tag = f"confirm_round{state.get('round', 1)}" if confirming else f"round{state.get('round', 1)}"
    run_root = Path(p["out_dir"]) / "loop" / p.get("run_id", "run")
    if confirming:

        base, k = tag, 1
        while (run_root / tag).exists():
            k += 1
            tag = f"{base}_{k}"
    out = run_root / tag

    bpmn, params = apply_redesign(
        plan, out_dir=out, detector_results_path=p["detector_results"],
        baseline_bpmn_path=p["baseline_bpmn"], baseline_params_path=p["baseline_params"],
        log_path=p["log_xes"])

    history = {k: list(v) for k, v in (state.get("build_history") or {}).items()}
    refused = []
    for e in _phase3_entries(out / "phase3"):
        ar = e.get("adapt_report") or {}
        applied = bool(ar.get("applied"))
        reason = "" if applied else str(ar.get("reason") or "")

        handler = {k: ar[k] for k in _HANDLER_SELF_REPORT if ar.get(k) is not None}
        cid = e["candidate_id"]
        rec = {"round": state.get("round", 1), "applied": applied, "reason": reason,
               "handler": handler,

               "order_dependent": (False if applied else
                                   _order_dependent(reason, cid, plan, p)),
               "plan": list(plan)}

        prior = history.get(cid, [])
        if not applied and not rec["order_dependent"] and not any(r.get("applied") for r in prior):
            cached = next((r["probe"] for r in prior
                           if (r.get("probe") or {}).get("builds_alone") is not None), None)
            if cached is not None:
                alone, note = cached["builds_alone"], "cached: " + cached["note"]
            else:
                alone, note = _builds_alone(cid, p, out.parent)
            rec["probe"] = {"builds_alone": alone, "note": note}
            if alone is True:
                rec["order_dependent"] = True
                rec["reason"] = (reason + " — but it BUILDS ALONE on the baseline, so "
                                 "this refusal is about the plan's earlier candidates, "
                                 "not about the candidate")
            elif alone is None:

                rec["order_dependent"] = True
                rec["reason"] = reason + " — probe inconclusive, not banning on an unknown"
        history.setdefault(cid, []).append(rec)
        if not applied:
            refused.append((cid, rec["order_dependent"]))

    built = [e["candidate_id"] for e in _phase3_entries(out / "phase3")
             if (e.get("adapt_report") or {}).get("applied")]
    from mabpr.apply_redesign import design_fingerprint
    design_key = design_fingerprint(out / "phase3") or "|".join(built)
    measured = state.get("measured_designs") or {}
    seen = measured[design_key] if design_key in measured else None

    return {
        "paths": {**p, "active_dir": str(out), "active_phase3": str(out / "phase3"),
                  "active_sim": (seen or {}).get("sim_dir") or str(out / "sim"),

                  **({} if confirming else {"round_phase3": str(out / "phase3")})},
        "build_history": history,
        "design_key": design_key,

        "apply_dir": str(out),

        **({"last_agg": seen["agg"]} if seen else {}),
        "trace": [{"node": "APPLY", "mode": state.get("mode"), "plan": plan,
                   "bpmn": str(bpmn),
                   **({"reused": f"design already measured in round {seen['round']}"}
                      if seen else {}),
                   **({"refused": [c for c, _ in refused],
                       "structural": [c for c, od in refused if not od]} if refused else {})}],
    }

def _phase3_entries(phase3_dir: Path) -> list[dict]:
    log = Path(phase3_dir) / "phase3_log.json"
    if not log.exists():
        return []
    return json.loads(log.read_text(encoding="utf-8")).get("entries", [])

def _order_dependent(reason: str, cid: str, plan: list[str], paths: dict) -> bool:
    """Could a different ORDER have let this candidate build?"""

    from mabpr.agents.arbiter import target_survival

    all_c = json.loads(Path(paths["detector_results"]).read_text(encoding="utf-8"))
    by_id = {c["candidate_id"]: c for c in all_c}
    if cid not in by_id:
        return False
    s = target_survival(by_id[cid], list(plan), by_id)
    return any(a in reason for a in s["removed_by_earlier_candidate"])

def _builds_alone(cid: str, paths: dict, run_dir: Path) -> "tuple[bool | None, str]":
    """Does this candidate build when it is the ONLY one applied to the pristine"""

    from mabpr.apply_redesign import apply_redesign_isolated
    out = Path(run_dir) / "probe" / cid
    try:
        apply_redesign_isolated(
            [cid], out_dir=out, detector_results_path=paths["detector_results"],
            baseline_bpmn_path=paths["baseline_bpmn"],
            baseline_params_path=paths["baseline_params"], log_path=paths["log_xes"])
    except Exception as exc:                                    # noqa: BLE001
        return None, f"probe could not run: {type(exc).__name__}: {str(exc)[:160]}"
    e = next((x for x in _phase3_entries(out / "phase3") if x.get("candidate_id") == cid), None)
    if e is None:
        return None, "probe wrote no phase3 entry for the candidate"
    ar = e.get("adapt_report") or {}
    if ar.get("applied"):
        return True, "builds alone on the pristine baseline"
    return False, f"refuses alone too: {str(ar.get('reason') or '')[:200]}"

def route_after_apply(state: MABPRState) -> str:
    """Skip the measurement when Phase-3 just rebuilt a design already simulated."""

    return ("MEASURE" if state.get("design_key", "") in (state.get("measured_designs") or {})
            else "SIMULATE")

def make_simulate_node(simulate_fn: SimulateFn):
    """SIMULATE — its own node so the checkpointer caches it."""

    def simulate_node(state: MABPRState) -> dict:
        p = state["paths"]
        phase3 = Path(_active_phase3(state))
        confirming = state.get("mode") == "confirming"
        plan = state["confirmation_plan"] if confirming else state["current_plan"]
        cfg = {**state.get("sim_config", {}), "plan": plan}

        kpis = simulate_fn(phase3 / "redesigned.bpmn", phase3 / "redesigned_params.json",
                           Path(p["active_dir"]) / "sim", cfg)
        agg = aggregate_across_runs(kpis)
        key = state.get("design_key", "")
        out: dict = {
            "last_agg": agg,

            "measured_designs": {**(state.get("measured_designs") or {}),
                                 key: {"agg": agg, "round": state.get("round", 1),
                                       "plan": list(plan),

                                       "sim_dir": str(Path(p["active_dir"]) / "sim")}},
            "sim_budget": int(state.get("sim_budget", 20)) - 1,
            "trace": [{"node": "SIMULATE", "mode": state.get("mode"), "runs": len(kpis),
                       "sim_budget_left": int(state.get("sim_budget", 20)) - 1}],
        }
        if confirming:

            if (state.get("confirmation_request") or {}).get("hypothesis_id") != "clean_subset":
                out["confirm_budget"] = int(state.get("confirm_budget", 2)) - 1
                out["confirm_per_round"] = int(state.get("confirm_per_round", 1)) - 1
        return out

    return simulate_node

def measure_node(state: MABPRState) -> dict:
    """MEASURE (J5) — two aggregates in, per-activity z + who-touched-what out."""

    p = state["paths"]
    n_runs = int(state.get("sim_config", {}).get("n_runs", 20))

    agg = state["last_agg"]
    fps = read_footprints(_active_phase3(state), p["detector_results"])

    sim = evidence_sim_dir(state)
    base_csvs = sorted((Path(p["compare_dir"])).glob("baseline_sim_*.csv"))
    res_deltas = None
    if sim and sorted(sim.glob("stats_*.csv")):
        try:
            res_deltas = compare_resources(
                load_resource_aggregate(p["compare_dir"], "baseline"),
                load_resource_aggregate(sim, ""), n_runs)
        except FileNotFoundError:
            res_deltas = None

    case_endings = compare_case_endings(
        state["base_agg"], agg, n_runs,
        merged=_merged_names(state) + rename_records(fps))

    from mabpr.explain import _handler_self_report
    _handlers = {cid: (_handler_self_report(state.get("build_history"), cid) or {})
                 for cid in fps}
    bundle = build_bundle(state["base_agg"], agg, n_runs, fps,
                          handler_reports=_handlers,
                          case_endings=case_endings,
                          base_sim_csvs=base_csvs,
                          redes_sim_csvs=sorted(sim.glob("sim_*.csv")) if sim else None,
                          resource_deltas=res_deltas,

                          activity_ledger=state.get("activity_ledger"),

                          candidates=json.loads(Path(p["detector_results"])
                                                .read_text(encoding="utf-8")))

    act_rows = [{"round": state.get("round", 1), "mode": state.get("mode", "normal"),
                 **r} for r in (bundle["scorecard"].get("activity_rows") or [])]

    _ordering_now = list((state.get("confirmation_plan")
                          if state.get("mode") == "confirming"
                          else state.get("current_plan")) or [])
    _built, _nothing = [], []
    for _cid in _ordering_now:
        _fp = fps.get(_cid)
        if _fp is None:
            _nothing.append({"candidate_id": _cid,
                             "why": "no footprint: the build step has no record of it"})
        elif not getattr(_fp, "applied", False):
            _nothing.append({"candidate_id": _cid,
                             "why": "the handler refused: %s"
                                    % (getattr(_fp, "skip_reason", "")
                                       or "no reason recorded")})
        elif getattr(_fp, "structural_noop", False):

            _nothing.append({"candidate_id": _cid,
                             "why": "applied but left no structural mark"})
        else:
            _built.append(_cid)

    if state.get("mode") == "confirming":

        delta, z = scalar_z(state["base_agg"], agg, KPI, n_runs)
        _rq = state.get("confirmation_request") or {}
        return {"confirmation_bundle": bundle,
                "activity_ledger": act_rows,
                "tried_plans": [{"round": state.get("round", 1), "mode": "confirming",
                                 "artifacts": state.get("apply_dir", ""),

                                 "kind": ("probe" if _rq.get("hypothesis_id")
                                          == "clean_subset" else "confirmation"),
                                 "probe_reason": (_rq.get("reason")
                                                  if _rq.get("hypothesis_id")
                                                  == "clean_subset" else None),
                                 "ordering": list(state.get("confirmation_plan") or []),
                                 "built": _built, "changed_nothing": _nothing,
                                 "design": state.get("design_key", ""),
                                 "median_h": agg[f"{KPI}_mean"],
                                 "median_std": agg[f"{KPI}_std"],
                                 "kpis": multi_kpi(state["base_agg"], agg, n_runs),

                                 "case_endings": case_endings,
                                 "vs_baseline_z": z,
                                 "pct_vs_baseline":
                                     delta / state["base_agg"][f"{KPI}_mean"] * 100}],
                "trace": [{"node": "MEASURE", "mode": "confirming",
                           "significant_rows": len(bundle["significant_deltas"])}]}

    delta, z = scalar_z(state["base_agg"], agg, KPI, n_runs)
    median, std = agg[f"{KPI}_mean"], agg[f"{KPI}_std"]

    verdict, no_gain, new_best_agg = convergence(
        agg, state.get("best_agg"), n_runs,
        int(state.get("rounds_without_improvement", 0)), baseline_z=z)
    new_best = new_best_agg.get(f"{KPI}_mean", median)
    new_best_std = new_best_agg.get(f"{KPI}_std", std)

    return {
        "redes_agg": agg,
        "footprints": {cid: asdict(f) for cid, f in fps.items()},
        "bundle": bundle,
        "activity_ledger": act_rows,
        "best_agg": new_best_agg,

        "best_median_h": new_best,
        "best_median_std": new_best_std,
        "rounds_without_improvement": no_gain,
        "improvement": verdict,
        "improvement_z": (verdict.get("per_measure", {})
                          .get(KPI, {}).get("z") if not verdict.get("first_round")
                          else verdict.get("baseline_z")),
        "tried_plans": [{"round": state.get("round", 1), "mode": "normal",
                                 "artifacts": state.get("apply_dir", ""),
                         "ordering": list(state.get("current_plan") or []),
                         "built": _built, "changed_nothing": _nothing,
                         "design": state.get("design_key", ""),
                         "median_h": median, "median_std": std, "vs_baseline_z": z,
                         "pct_vs_baseline": (delta / state["base_agg"][f"{KPI}_mean"] * 100),

                         "kpis": multi_kpi(state["base_agg"], agg, n_runs),

                         "case_endings": case_endings}],
        "trace": [{"node": "MEASURE", "mode": "normal", "round": state.get("round", 1),
                   "median_h": round(median, 4), "vs_baseline_z": round(z, 1),
                   "significant_rows": len(bundle["significant_deltas"]),
                   "rounds_without_improvement": no_gain}],
    }

def accept_anti_combos(existing: list | None, directive: dict | None,
                       conflicts: list | None) -> tuple[list, list]:
    """(all anti-combos, the ones added now). Code decides; the Critic proposes."""

    combos = [list(c) for c in (existing or [])]
    known = {tuple(sorted(c)) for c in combos}
    measured = {tuple(sorted(str(x) for x in (c.get("between") or [])))
                for c in (conflicts or [])
                if c.get("confidence") == "measured" and len(c.get("between") or []) == 2}
    added: list = []
    for pair in ((directive or {}).get("avoid") or []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        key = tuple(sorted(str(x) for x in pair))
        if key in known or key not in measured:
            continue
        known.add(key)
        combos.append(list(key))
        added.append(list(key))
    return combos, added

_HANDLER_SELF_REPORT = (
    "channel", "strategy",
    "timers_removed", "timer_h_removed_per_occurrence", "batch_component_h_removed",
    "share_of_timer_removed", "rules_removed", "batch_rules_dropped",
    "paths_remapped", "gateways_conditioned", "branch_rules_added",
    "loop_probability_reduced", "resources_removed", "task_resources_added",
    "ca_pool_resource_count", "resources_grown",

    "composite_name", "distributions_folded", "distributions_carried_over",
    "assigned_tasks_removed", "assigned_tasks_added", "setup_saving_assumed_h",
    "subprocess_registered", "subprocess_resource_count", "subprocess_duration_s",
    "subprocess_assigned_added", "encapsulation", "divergence_point",
    "expected_continuity_before", "expected_continuity_after", "concentration",
    "delivery_note", "note",
)

def _merged_names(state: dict) -> list:
    """Composite names any Activity Composition handler built this round."""

    out = []
    for runs in (state.get("build_history") or {}).values():
        for rec in runs or []:
            name = (rec.get("handler") or {}).get("composite_name")
            if name and name not in out:
                out.append(name)
    return out

def confirmation_counterfactual(round_plan: list, conf_plan: list,
                                round_agg: dict, conf_agg: dict, base_agg: dict,
                                n_runs: int) -> dict:
    """The confirmation read as a plan-level counterfactual, in the stopping"""

    removed = [c for c in round_plan if c not in conf_plan]
    added = [c for c in conf_plan if c not in round_plan]
    v = improvement_verdict(conf_agg, round_agg, n_runs)
    full, conf = multi_kpi(base_agg, round_agg, n_runs), multi_kpi(base_agg, conf_agg, n_runs)
    rows = []
    for key in CONVERGENCE_DIRECTION:
        if key in full and key in conf and full[key].get("pct") is not None and conf[key].get("pct") is not None:
            rows.append({"measure": key, "full_pct": full[key]["pct"], "conf_pct": conf[key]["pct"],
                         "vs_full_z": (v.get("per_measure") or {}).get(key, {}).get("z")})
    return {"removed": removed, "added": added,
            "verdict": {"improved": v.get("improved"), "better_on": v.get("better_on"),
                        "worse_on": v.get("worse_on")},
            "per_measure": rows}

def multi_kpi(base_agg: dict, redes_agg: dict, n_runs: int) -> dict:
    """Every reported KPI with its own z — the plan-level counterpart of the"""

    out = {}
    for key, label, what in REPORT_KPIS:
        b, r = base_agg.get(f"{key}_mean"), redes_agg.get(f"{key}_mean")
        if b is None or r is None:
            continue
        sb = base_agg.get(f"{key}_std") or 0.0
        sr = redes_agg.get(f"{key}_std") or 0.0
        se = math.sqrt((sb ** 2 + sr ** 2) / max(1, n_runs))
        z = None if se == 0 else (r - b) / se
        out[key] = {"label": label, "measures": what, "base": b, "redes": r,
                    "delta": r - b, "pct": ((r - b) / b * 100) if b else None,
                    "z": None if z is None else round(z, 2),
                    "significant": bool(z is not None and abs(z) > 3.0)}
    return out

CONVERGENCE_DIRECTION = {
    "throughput_h_median": "lower",
    "throughput_h_mean": "lower",
    "waiting_h_per_case_mean": "lower",
    "duration_h_per_case_mean": "lower",
    "utilization_mean": None,
}

def improvement_verdict(agg: dict, incumbent: dict, n_runs: int) -> dict:
    """Did this round significantly improve on the incumbent, across ALL measures?"""

    per, better, worse = {}, [], []
    for key, direction in CONVERGENCE_DIRECTION.items():
        b, r = incumbent.get(f"{key}_mean"), agg.get(f"{key}_mean")
        if b is None or r is None:
            continue
        sb = incumbent.get(f"{key}_std") or 0.0
        sr = agg.get(f"{key}_std") or 0.0
        se = math.sqrt((sb ** 2 + sr ** 2) / max(1, n_runs))
        if se == 0:
            z = 0.0 if r == b else (math.inf if r > b else -math.inf)
        else:
            z = (r - b) / se
        sig = abs(z) > abs(IMPROVE_Z)
        per[key] = {"z": None if z in (math.inf, -math.inf) else round(z, 2),
                    "significant": bool(sig), "directional": direction is not None}
        if direction is None or not sig:
            continue
        (better if (z < 0) == (direction == "lower") else worse).append(key)
    return {"improved": bool(better) and not worse,
            "better_on": better, "worse_on": worse, "per_measure": per}

def convergence(agg: dict, incumbent: dict | None, n_runs: int,
                no_gain_so_far: int, *, baseline_z: float) -> tuple:
    """(verdict, rounds-without-improvement, the new incumbent aggregate)."""

    if not incumbent:
        return ({"improved": True, "better_on": [], "worse_on": [],
                 "per_measure": {}, "first_round": True, "baseline_z": baseline_z},
                0, dict(agg))
    v = improvement_verdict(agg, incumbent, n_runs)
    return (v, 0 if v["improved"] else no_gain_so_far + 1,
            dict(agg) if v["improved"] else incumbent)

def route_after_measure(state: MABPRState) -> str:
    """The spine serves both loops; the mode says which one asked for it."""
    return "CRITIC_CONCLUDE" if state.get("mode") == "confirming" else "CRITIC_INVESTIGATE"

def critic_investigate_node(state: MABPRState) -> dict:
    """ORIENT + INVESTIGATE + DECIDE. The tool loop stays inside the node: reads"""

    bundle = state["bundle"]
    box = _toolbox(state)
    confirm_left = int(state.get("confirm_budget", 2))
    per_round = int(state.get("confirm_per_round", 1))

    hyps = orient(state.get("round", 1), bundle, state.get("tried_plans"))

    _rm = measured_removals(state.get("tried_plans"))[:MAX_REMOVAL_FACTS]
    _un = unmeasured_removals(state.get("current_plan"),
                              measured_removals(state.get("tried_plans")))
    hyps, req, loop_meta = investigate(hyps, bundle, box, confirm_left=confirm_left,
                                       per_round_left=per_round,
                                       removal_lines=removal_fact_lines(_rm, _un))
    applied = [cid for cid, fp in (bundle.get("footprints") or {}).items() if fp.get("applied")]
    granted, reason = confirmation_gate(req, box, confirm_left=confirm_left,
                                        per_round_left=per_round, applied=applied,
                                        hypotheses=hyps)
    if granted and int(state.get("sim_budget", 20)) <= 0:
        granted, reason = False, "simulation budget exhausted"

    return {
        "hypotheses": [asdict(h) for h in hyps],
        "read_log": box.audit(),
        "confirmation_request": req,
        "confirmation_granted": granted,
        "confirmation_reason": reason,
        "confirmation_plan": [str(c) for c in (req or {}).get("plan", [])] if granted else [],
        "mode": "confirming" if granted else "normal",
        "trace": [{"node": "CRITIC_INVESTIGATE", "round": state.get("round", 1),
                   "hypotheses": len(hyps), "reads": box.audit()["total_reads"],
                   **loop_meta, "confirmation": {"granted": granted, "reason": reason}}],
    }

def route_after_investigate(state: MABPRState) -> str:
    """J2: spend the earned confirmation, or go straight to the write-up. Code"""

    return "APPLY" if state.get("confirmation_granted") else "CRITIC_CONCLUDE"

def _attach_isolation(bundle: dict, ledger: "list | None") -> None:
    """Re-check the shared activities against the ledger AS IT NOW STANDS."""

    for row in ((bundle or {}).get("scorecard", {}).get("candidates") or {}).values():
        shared = row.get("unattributable_activities") or []
        if not shared:
            continue
        owners = set(row.get("shared_with") or []) | {
            c for c in (row.get("claimants") or {})}
        iso = {a: _isolation(ledger, a, sorted(owners)) for a in shared}
        iso = {a: v for a, v in iso.items() if v}
        if iso:
            row["isolated_in_earlier_rounds"] = iso

def critic_conclude_node(state: MABPRState) -> dict:
    """CONCLUDE — hypotheses + evidence -> the §9.1 cards. A hypothesis the"""

    from mabpr.agents.critic import Hypothesis

    hyps = [Hypothesis.from_dict(h) for h in state.get("hypotheses", [])]
    confirmation = None
    if state.get("confirmation_bundle"):
        confirmation = {**(state.get("confirmation_request") or {}),
                        "plan": state.get("confirmation_plan"),
                        "bundle": state["confirmation_bundle"]}

    _attach_isolation(state.get("bundle"), state.get("activity_ledger"))

    plan_fact = None
    if state.get("redes_agg") and state.get("base_agg"):
        plan_fact = improvement_verdict(state["redes_agg"], state["base_agg"],
                                        int((state.get("sim_config") or {}).get("n_runs", 20)))

    conf_fact = None
    if confirmation and state.get("last_agg") and state.get("redes_agg") and state.get("base_agg"):
        conf_fact = confirmation_counterfactual(
            state["current_plan"], state.get("confirmation_plan") or [],
            state["redes_agg"], state["last_agg"], state["base_agg"],
            int((state.get("sim_config") or {}).get("n_runs", 20)))

    _dk = state.get("design_key") or ""
    _seen = (state.get("measured_designs") or {}).get(_dk) or {}
    _repeat = _seen.get("round")
    if _repeat is not None and int(_repeat) == int(state.get("round", 1)):
        _repeat = None
    round_fact = {
        "round_no": int(state.get("round", 1)),
        "no_gain": int(state.get("rounds_without_improvement", 0)),
        "max_rounds": int(state.get("max_rounds", 12)),
        "patience": PATIENCE,
        "design_repeat_of": _repeat,
    }

    baseline_hours = None
    if state.get("redes_agg") and state.get("base_agg"):
        baseline_hours = {
            "base_median_h": (state["base_agg"] or {}).get("throughput_h_median_mean"),
            "plan_median_h": (state["redes_agg"] or {}).get("throughput_h_median_mean"),
        }
    judged = conclude(state.get("round", 1), state["bundle"], hyps, confirmation,
                      confirm_left=int(state.get("confirm_budget", 0)),
                      plan_fact=plan_fact, conf_fact=conf_fact,
                      round_fact=round_fact, baseline_hours=baseline_hours,

                      follow_fact=state.get("directive_followthrough"),

                      blind_fact=unreachable_candidates(
                          state.get("scorecard"), (state.get("bundle") or {}).get(
                              "footprints"), state.get("current_plan")))

    bundle = state["bundle"]
    grounding = ["summary"]
    grounding += [f"activity.{r['activity']}" for r in bundle.get("significant_deltas", [])]
    grounding += [f"footprint.{cid}" for cid in bundle.get("footprints", {})]
    grounding += [f"tool.{r['tool']}->{r['hypothesis']}"
                  for r in state.get("read_log", {}).get("log", []) if not r.get("error")]
    if state.get("confirmation_bundle"):
        grounding.append(f"confirmation_sim.{state.get('confirmation_plan')}")

    _stopping = (int(state.get("sim_budget", 0)) <= 0
                 or int(state.get("rounds_without_improvement", 0)) >= PATIENCE
                 or int(state.get("round", 1)) >= int(state.get("max_rounds", 5)))
    _probe = state.get("clean_probe") or {}
    if _probe.get("state") not in ("pending", "queued", "finished"):

        _probe = plan_clean_probe(
            state, (state.get("bundle") or {}).get("scorecard") or {},
            judged.get("per_candidate"), _stopping)
    if _probe.get("state") in ("queued", "pending"):
        _probe = advance_probe_queue(_probe, state.get("tried_plans"),
                                     int(state.get("sim_budget", 0)))
        if _probe.get("current"):
            print("      probe %d/%d: measuring %s (without %s)"
                  % (len(_probe.get("done") or []) + 1,
                     len(_probe.get("done") or []) + 1 + len(_probe.get("queue") or []),
                     _probe["current"]["plan"], _probe["current"]["drops"]))

    if _probe.get("state") == "finished":
        _probe = reopen_after_clean_subset(
            _probe, state, (state.get("bundle") or {}).get("scorecard") or {})
        if _probe.get("state") == "queued":
            _probe = advance_probe_queue(_probe, state.get("tried_plans"),
                                         int(state.get("sim_budget", 0)))
            if _probe.get("current"):
                print("      probe (re-judged): measuring %s (without %s)"
                      % (_probe["current"]["plan"], _probe["current"]["drops"]))

    _malformed = [{"index": i,
                   "has_candidate_id": bool(e.get("candidate_id")),
                   "has_verdict": bool(e.get("verdict")),
                   "keys": sorted(e)[:8]}
                  for i, e in enumerate(judged.get("per_candidate") or [])
                  if not e.get("candidate_id") or not e.get("verdict")]
    if _malformed:
        print("      ! %d per_candidate row(s) missing candidate_id/verdict; "
              "recorded, not guessed" % len(_malformed))

    assessment = {
        "round": state.get("round", 1),
        "plan": list(state.get("current_plan") or []),
        "assessment": {
            "per_candidate": judged.get("per_candidate", []),
            "conflicts": judged.get("conflicts", []),
            "landed_as_expected": judged.get("landed_as_expected"),
            "surprises": judged.get("surprises", ""),

            "per_candidate_missing": judged.get("per_candidate_missing") or [],

            "plan_reading": judged.get("plan_reading"),
            "plan_vs_baseline": judged.get("plan_vs_baseline"),

            "confirmation_counterfactual": conf_fact,

            "per_candidate_malformed": _malformed,

            "schema_retry": judged.get("schema_retry"),

            "clean_probe": _probe,
        },
        "hypotheses": state.get("hypotheses", []),
        "recommendation": judged.get("recommendation"),

        "directive": normalize_directive(
            judged.get("directive"), judged.get("per_candidate"),
            scorecard=(state.get("bundle") or {}).get("scorecard") or {}),
        "confirmation": {"granted": bool(state.get("confirmation_granted")),
                         "reason": state.get("confirmation_reason"),
                         "plan": state.get("confirmation_plan") or None},
        "grounding": grounding,
    }

    harmful = {k: list(v) for k, v in (state.get("harmful_history") or {}).items()}
    for e in judged.get("per_candidate", []):
        if e.get("verdict") in HARMFUL_VERDICTS and e.get("candidate_id"):
            harmful.setdefault(e["candidate_id"], []).append(
                {"round": state.get("round", 1), "verdict": e.get("verdict"),
                 "confidence": e.get("confidence"), "note": str(e.get("note") or "")[:200]})

    expl = explain_round(
        state.get("round", 1),
        previous_plan=list(state.get("current_plan") or []),
        new_plan=list(state.get("current_plan") or []),
        scorecard=(state.get("bundle") or {}).get("scorecard") or {},
        assessment=assessment, proposal={},
        gate_ruling=state.get("gate_ruling"),
        build_history=state.get("build_history"),
        stage="measured")
    rd = round_dir(state)
    if rd is not None:
        write_round_explanation(rd, expl)

    combos, new_combos = accept_anti_combos(
        state.get("anti_combos"), judged.get("directive"), judged.get("conflicts"))

    rows = ledger_rows(state.get("round", 1), list(state.get("current_plan") or []),
                       (state.get("bundle") or {}).get("scorecard") or {}, assessment)

    _pending = bool(_probe.get("state") == "pending" and _probe.get("current"))
    return {"assessment": assessment,
            "candidate_ledger": rows,
            "explanation": expl,
            "anti_combos": combos,
            "clean_probe": _probe,
            "mode": "confirming" if _pending else "normal",
            **({"confirmation_plan": list(_probe["current"]["plan"]),
                "confirmation_request": {
                    "hypothesis_id": "clean_subset",
                    "plan": list(_probe["current"]["plan"]),
                    "reason": _probe["current"]["why"]},
                "confirmation_granted": True} if _pending else {}),
            "confirmation_bundle": None,
            "harmful_history": harmful,
            "trace": [{"node": "CRITIC_CONCLUDE", "round": state.get("round", 1),

                       "verdicts": {e.get("candidate_id"): e.get("verdict")
                                    for e in (judged.get("per_candidate") or [])
                                    if e.get("candidate_id")},
                       **({"per_candidate_malformed": _malformed} if _malformed else {}),
                       **({"anti_combos_added": new_combos} if new_combos else {}),
                       "directive": judged.get("directive")}]}

def clean_exclusions(scorecard, plan) -> list:
    """Which candidates the clean subset drops, and why — the ONE definition."""

    rows = (scorecard or {}).get("candidates") or {}
    out = []
    for cid in (plan or []):
        net = (rows.get(cid) or {}).get("net") or {}
        h = net.get("throughput_h_per_case")
        if h is not None and float(h) > 0 and bool(net.get("throughput_significant")):
            out.append({"candidate_id": cid,
                        "throughput_h_per_case": h,
                        "z": net.get("throughput_z")})
    return out

def unreachable_candidates(scorecard, footprints, plan) -> list:
    """Applied candidates that NO automatic mechanism in this loop can remove."""

    rows = (scorecard or {}).get("candidates") or {}
    reachable_by_clean = {e["candidate_id"] for e in clean_exclusions(scorecard, plan)}
    out = []
    for cid in (plan or []):
        row = rows.get(cid) or {}
        fp = (footprints or {}).get(cid) or {}
        if not row:
            continue
        if cid in reachable_by_clean:
            continue
        if (fp.get("routing_change") if isinstance(fp, dict)
                else getattr(fp, "routing_change", None)):
            continue
        net = row.get("net") or {}
        pool = (fp.get("pool_merged") if isinstance(fp, dict)
                else getattr(fp, "pool_merged", None))
        out.append({
            "candidate_id": cid,
            "heuristic": row.get("heuristic"),
            "own_throughput_h_per_case": net.get("throughput_h_per_case"),
            "own_z": net.get("throughput_z"),
            "own_significant": bool(net.get("throughput_significant")),
            "pool_resources_behind_its_targets": pool,
            "activities": sorted((fp.get("activities") if isinstance(fp, dict)
                                  else getattr(fp, "activities", None)) or []),
        })
    return out

def plan_weight(row) -> int:
    """How many candidates of this measurement actually changed the model."""

    if isinstance(row, dict) and row.get("built") is not None:
        return len(row.get("built") or [])
    return len((row or {}).get("ordering") or [])

def best_measured_so_far(tried_plans, n_runs: int):
    """The median-best measurement so far, by the SAME rule the final report uses."""

    best = None
    for t in (tried_plans or []):
        if t.get("median_h") is None:
            continue
        if best is None or _sig_better(t, best, n_runs):
            best = t
            continue

        if not _sig_better(best, t, n_runs):
            inf = float("inf")
            key_t = (plan_weight(t),
                     t.get("median_h") if t.get("median_h") is not None else inf)
            key_b = (plan_weight(best),
                     best.get("median_h") if best.get("median_h") is not None else inf)
            if key_t < key_b:
                best = t

    if best is not None:
        eligible = [t for t in (tried_plans or [])
                    if t.get("median_h") is not None
                    and not _sig_better(best, t, n_runs)]
        if eligible:
            inf = float("inf")
            best = min(eligible,
                       key=lambda t: (plan_weight(t),
                                      t.get("median_h") if t.get("median_h")
                                      is not None else inf))
    return best

def adverse_from_ledger(ledger, round_no) -> dict:
    """Candidates whose OWN attributable throughput was significantly worse in the"""

    out = {}
    for r_ in (ledger or []):
        if round_no is not None and r_.get("round") != round_no:
            continue
        v = r_.get("thr_h_per_case")
        if isinstance(v, (int, float)) and v > 0 and r_.get("thr_significant"):
            out[r_.get("candidate_id")] = {"candidate_id": r_.get("candidate_id"),
                                           "throughput_h_per_case": v,
                                           "z": r_.get("thr_z")}
    return out

MAX_LOO_PROBES = 4
MAX_COMPLIANCE_PROBES = 2

def routing_editors(footprints, plan) -> list:
    """Candidates in `plan` whose handler reported a ROUTING edit, in plan order."""

    out = []
    for cid in (plan or []):
        fp = (footprints or {}).get(cid)
        rc = ((fp.get("routing_change") if isinstance(fp, dict)
               else getattr(fp, "routing_change", None)) or {})
        if rc:
            out.append(cid)
    return out

def waiting_raisers(state, plan) -> list:
    """Candidates in `plan` whose transformation can only push WAITING UP."""

    from mabpr.explain import _handler_self_report
    from mabpr.evidence.scorecard import _WAIT_DIRECTION
    SHRINKS_A_POOL = ("resources_removed", "concentration")
    assert all(_WAIT_DIRECTION.get(k) == "up" for k in SHRINKS_A_POOL), (
        "a counter this predicate reads must be declared `up` in _WAIT_DIRECTION")
    out = []
    for cid in (plan or []):
        rep = _handler_self_report(state.get("build_history"), cid) or {}
        if any(rep.get(k) for k in SHRINKS_A_POOL):
            out.append(cid)
    return out

def compliance_exposed(footprints, plan, tried_plans) -> list:
    """Candidates that DELETE an activity and sit in EVERY measurement so far."""

    exposed = []
    for cid in (plan or []):
        fp = (footprints or {}).get(cid)
        if fp is None:
            continue
        deleted = (fp.get("deleted") if isinstance(fp, dict)
                   else getattr(fp, "deleted", None))
        if not deleted:
            continue
        if not tried_plans or all(cid in (t.get("ordering") or [])
                                  for t in tried_plans):
            exposed.append(cid)
    return exposed

def plan_clean_probe(state, scorecard, judged_per_candidate, stopping: bool) -> dict:
    """What extra measurements to take before the run ends, as an ordered QUEUE."""

    n_runs = int((state.get("sim_config") or {}).get("n_runs", 20))
    _best = best_measured_so_far(state.get("tried_plans"), n_runs)
    if _best and list(_best.get("ordering") or []):
        plan = list(_best["ordering"])
        basis = "r%s%s" % (_best.get("round"),
                           "c" if _best.get("mode") == "confirming" else "")
    else:
        plan, basis = list(state.get("current_plan") or []), "current round"

    adverse = adverse_from_ledger(state.get("candidate_ledger"),
                                  (_best or {}).get("round"))
    for x in clean_exclusions(scorecard, plan):
        adverse.setdefault(x["candidate_id"], x)
    clean = [c for c in plan if c not in adverse]
    dropped = [c for c in plan if c in adverse]
    exo = [e.get("candidate_id") or e.get("candidate")
           for e in (judged_per_candidate or [])
           if e.get("verdict") == "exogenous_only"]

    _net_rows = (scorecard or {}).get("candidates") or {}

    def _better_and_clean(t):
        _ord = list(t.get("ordering") or [])
        if not _ord or not t.get("kpis"):
            return False
        if improvement_rule(t["kpis"]).get("verdict") != "better":
            return False
        for _cid in _ord:
            _net = (_net_rows.get(_cid) or {}).get("net") or {}
            _h = _net.get("throughput_h_per_case")
            if _h is None:
                return False
            if float(_h) > 0 and bool(_net.get("throughput_significant")):
                return False
        return True

    _qualifying = [t for t in (state.get("tried_plans") or []) if _better_and_clean(t)]

    _rule_base = None
    _rows_rule = [t for t in (state.get("tried_plans") or [])
                  if t.get("kpis") and list(t.get("ordering") or [])
                  and improvement_rule(t["kpis"]).get("verdict") == "better"]

    _round_base = None
    _rows_round = [r for r in _rows_rule if r.get("mode") != "confirming"]
    if _rows_rule:
        _inf = float("inf")
        _rule_base = min(_rows_rule,
                         key=lambda t: (plan_weight(t),
                                        t.get("median_h") if t.get("median_h")
                                        is not None else _inf))
    if _rows_round:
        _inf2 = float("inf")
        _round_base = min(_rows_round,
                          key=lambda t: (plan_weight(t),
                                         t.get("median_h") if t.get("median_h")
                                         is not None else _inf2))
    rec = {"basis": basis, "deliverable": plan,
           "rule_basis": ("r%s%s" % (_rule_base.get("round"),
                                     "c" if _rule_base.get("mode") == "confirming"
                                     else "")) if _rule_base else None,
           "round_basis": ("r%s" % _round_base.get("round")) if _round_base else None,
           "qualifying_measurement": (
               "r%s%s" % (_qualifying[0].get("round"),
                          "c" if _qualifying[0].get("mode") == "confirming" else "")
               if _qualifying else None),
           "dropped_because": [adverse[c] for c in dropped],
           "excluded_exogenous_only": [c for c in exo if c in dropped],
           "queue": [], "done": [], "current": None}

    if not stopping:
        return {**rec, "state": "skipped", "reason": "the loop is not stopping yet"}
    if int(state.get("sim_budget", 0)) <= 0:
        return {**rec, "state": "skipped", "reason": "no simulation budget left"}

    verdict = (improvement_rule(_best["kpis"]).get("verdict")
               if (_best and _best.get("kpis")) else None)
    rec["deliverable_vs_baseline"] = verdict
    queue = []

    if clean and dropped:
        queue.append({"plan": clean, "drops": dropped,
                      "why": "the deliverable without the candidates whose own "
                             "attributable throughput measured significantly worse"})

    if _rule_base is not None and list(_rule_base["ordering"]) != plan:
        plan2 = list(_rule_base["ordering"])
        adverse2 = adverse_from_ledger(state.get("candidate_ledger"),
                                       _rule_base.get("round"))
        for x in clean_exclusions(scorecard, plan2):
            adverse2.setdefault(x["candidate_id"], x)
        clean2 = [c for c in plan2 if c not in adverse2]
        dropped2 = [c for c in plan2 if c in adverse2]
        rec["rule_basis_plan"] = plan2
        rec["rule_basis_dropped_because"] = [adverse2[c] for c in dropped2]
        if clean2 and dropped2 and clean2 != clean:
            queue.append({"plan": clean2, "drops": dropped2,
                          "why": "the leanest measurement that SATISFIES the "
                                 "improvement rule, without the candidates whose "
                                 "own attributable throughput measured "
                                 "significantly worse -- the median-best plan "
                                 "cleaned above is not necessarily the one that "
                                 "will be delivered"})

    if _round_base is not None:
        plan3 = list(_round_base["ordering"])
        adverse3 = adverse_from_ledger(state.get("candidate_ledger"),
                                       _round_base.get("round"))
        for x in clean_exclusions(scorecard, plan3):
            adverse3.setdefault(x["candidate_id"], x)
        clean3 = [c for c in plan3 if c not in adverse3]
        dropped3 = [c for c in plan3 if c in adverse3]
        rec["round_basis_plan"] = plan3
        rec["round_basis_dropped_because"] = [adverse3[c] for c in dropped3]
        if (clean3 and dropped3
                and all(clean3 != q["plan"] for q in queue)
                and clean3 != plan):
            queue.append({"plan": clean3, "drops": dropped3,
                          "why": "the leanest plan the loop itself proposed that "
                                 "satisfies the improvement rule, without the "
                                 "candidates whose own attributable throughput "
                                 "measured significantly worse -- both bases above "
                                 "can be confirmations, which are leaner than any "
                                 "round by construction"})

    _exposed = compliance_exposed((state.get("footprints") or {}), plan,
                                  state.get("tried_plans"))
    rec["compliance_exposed"] = _exposed
    for cid in _exposed[:MAX_COMPLIANCE_PROBES]:
        sub = [c for c in plan if c != cid]
        if sub:
            queue.append({"plan": sub, "drops": [cid],
                          "why": "the deliverable without %s, which deletes an "
                                 "activity and is in EVERY measurement -- so a "
                                 "final compliance ruling against it would leave "
                                 "the run with no legal deliverable at all" % cid})

    if verdict != "better":

        _worse = ((improvement_rule(_best["kpis"]).get("worse_on") or [])
                  if (_best and _best.get("kpis")) else [])
        queue.extend(loo_queue(state, plan, scorecard,
                               (_best or {}).get("round"), rec,
                               waiting_worse="waiting_h_per_case_mean" in _worse))

    if not queue and rec.get("loo_already_measured"):
        return {**rec, "state": "skipped",
                "reason": ("every routing editor in the deliverable has already "
                           "been isolated this run (%s), so a leave-one-out probe "
                           "would re-measure a design the run already has"
                           % ", ".join(rec["loo_already_measured"]))}
    if not queue:
        return {**rec, "state": "skipped",
                "reason": ("nothing in the deliverable measured significantly worse, "
                           "no candidate in it is exposed to a compliance ruling, "
                           "and it satisfies the improvement rule, so it is "
                           "delivered as measured" if verdict == "better"
                           else "nothing left to clean or isolate")}
    return {**rec, "queue": queue, "state": "queued",
            "reason": "%d extra measurement(s) queued before the run ends" % len(queue)}

def loo_queue(state, plan, scorecard, best_round, rec, waiting_worse=False) -> list:
    """The bounded leave-one-out queue over the plan's ROUTING editors."""

    editors = routing_editors((state.get("footprints") or {}), plan)

    _acct = {}
    for r_ in (state.get("candidate_ledger") or []):
        if r_.get("round") == best_round:
            v = r_.get("thr_h_per_case")
            if isinstance(v, (int, float)):
                _acct[r_.get("candidate_id")] = abs(v)
    _rows = (scorecard or {}).get("candidates") or {}
    for cid in editors:
        if cid not in _acct:
            v = ((_rows.get(cid) or {}).get("net") or {}).get("throughput_h_per_case")
            _acct[cid] = abs(v) if isinstance(v, (int, float)) else 0.0
    editors = sorted(editors, key=lambda c: (_acct.get(c, 0.0), c))
    rec["routing_editors"] = editors

    rec["waiting_raisers"] = raisers = waiting_raisers(state, plan) if waiting_worse else []
    editors = raisers + [c for c in editors if c not in raisers]

    rec["isolated_plan"] = list(plan or [])

    _known = {r["candidate"] for r in measured_removals(state.get("tried_plans"))}
    _fresh = [c for c in editors if c not in _known]
    rec["loo_already_measured"] = sorted(_known & set(editors))
    rec["loo_order"] = list(editors)
    rec["loo_queued"] = _fresh[:MAX_LOO_PROBES]
    out = []
    for cid in _fresh[:MAX_LOO_PROBES]:
        sub = [c for c in plan if c != cid]
        if sub:
            out.append({"plan": sub, "drops": [cid],
                        "why": "the deliverable without %s, whose handler "
                               "reported a routing edit -- a routing change "
                               "can move the case span while every "
                               "per-candidate column looks balanced, and "
                               "only an isolation can attribute it" % cid})
    return out

def reopen_after_clean_subset(probe: dict, state, scorecard) -> dict:
    """⚠️⚠️ RE-JUDGE ONCE THE CLEAN SUBSET HAS BEEN MEASURED."""

    if not probe or probe.get("reopened"):
        return probe
    if int(state.get("sim_budget", 0)) <= 0:
        return probe
    done = [d for d in (probe.get("done") or []) if len(d.get("drops") or []) > 1]
    if not done:
        return probe
    want = tuple(done[0].get("plan") or [])

    if tuple(probe.get("isolated_plan") or []) == want:
        return probe
    row = None
    for t in (state.get("tried_plans") or []):
        if tuple(t.get("ordering") or []) == want and t.get("kpis"):
            row = t
    if row is None:
        return probe
    p = dict(probe)
    p["reopened"] = True
    v = improvement_rule(row["kpis"]).get("verdict")
    p["clean_subset_verdict"] = v
    if v == "better":
        p["reopen_reason"] = ("the clean subset measured `better`, so the campaign "
                              "is finished -- a deliverable that satisfies the rule "
                              "is not probed further")
        return p

    q = loo_queue(state, list(done[0].get("plan") or []), scorecard,
                  (row or {}).get("round"), p,
                  waiting_worse=("waiting_h_per_case_mean"
                                 in (improvement_rule(row["kpis"]).get("worse_on")
                                     or [])))
    if not q:
        p["reopen_reason"] = ("the clean subset measured `%s`, but every routing "
                              "editor in it has already been isolated this run" % v)
        return p
    p["queue"] = list(p.get("queue") or []) + q
    p["state"] = "queued"
    p["reopen_reason"] = ("the clean subset measured `%s`, not `better`, and the "
                          "isolation campaign had not run because the BASE "
                          "satisfied the rule -- so the plan actually heading for "
                          "delivery was never isolated" % v)
    return p

def advance_probe_queue(probe: dict, tried_plans, sim_budget: int) -> dict:
    """Move the queue on by one: record what just landed, pick what is next."""

    p = {k: v for k, v in (probe or {}).items()}
    if p.get("current"):
        p["done"] = list(p.get("done") or []) + [p["current"]]
        p["current"] = None
    tried = {tuple(t.get("ordering") or []) for t in (tried_plans or [])}
    q = list(p.get("queue") or [])
    while q:
        if int(sim_budget) <= 0:
            p["queue"] = q
            p["state"] = "finished"
            p["reason"] = "simulation budget ran out with %d probe(s) unspent" % len(q)
            return p
        nxt = q.pop(0)
        if tuple(nxt["plan"]) in tried:
            p["already_measured"] = list(p.get("already_measured") or []) + [nxt["drops"]]
            continue
        p["queue"] = q
        p["current"] = nxt
        p["state"] = "pending"
        return p
    p["queue"] = []
    p["state"] = "finished"
    return p

def route_after_conclude(state: MABPRState) -> str:
    """Stopping is CODE's call, never the Critic's: a hard"""

    _p = state.get("clean_probe") or {}
    if _p.get("state") == "pending" and _p.get("current"):
        return "APPLY"
    if int(state.get("sim_budget", 0)) <= 0:
        return END
    if int(state.get("rounds_without_improvement", 0)) >= PATIENCE:
        return END
    if int(state.get("round", 1)) >= int(state.get("max_rounds", 5)):
        return END
    return "ARBITER_CALIBRATE"

def arbiter_calibrate_node(state: MABPRState) -> dict:
    """ⓑ CALIBRATE — reconcile the priors against what was measured and write back"""

    p = state["paths"]
    overlay = KnowledgeOverlay(p["overlay"])
    box = ArbiterBox(p["detector_results"], overlay_path=p["overlay"],
                     phase3_dir=_round_phase3(state))

    last = [t for t in (state.get("tried_plans") or []) if t.get("kpis")]
    sc_now = ((state.get("bundle") or {}).get("scorecard") or {}).get("candidates") or {}
    exo_now = {cid: row["exogenous_saving"] for cid, row in sc_now.items()
               if row.get("exogenous_saving")}
    r = calibrate(state.get("round", 1), state.get("assessment"),
                  state.get("proposal"), box, overlay,
                  plan_measures=(last[-1]["kpis"] if last else None),
                  exogenous=exo_now)
    return {
        "calibration": asdict(r),
        "contradictions": r.contradictions,
        "trace": [{"node": "ARBITER_CALIBRATE", "round": state.get("round", 1),
                   "lookups": box.audit()["total_reads"],
                   "accepted": len(r.accepted), "refused": len(r.refused),
                   "summary": r.summary[:120]}],
    }

def arbiter_gate_node(state: MABPRState) -> dict:
    """ⓐ GATE — may this plan run? Compliance (prior knowledge only,"""

    p = state["paths"]
    all_candidates = json.loads(Path(p["detector_results"]).read_text(encoding="utf-8"))
    box = ArbiterBox(p["detector_results"], overlay_path=p["overlay"],
                     phase3_dir=_round_phase3(state))

    never_builds = set(unbuildable(state.get("build_history")))

    _plan_in = list(state.get("current_plan") or [])
    _by_id = {c["candidate_id"]: c for c in all_candidates}
    _nullifiers: set = set()
    for _cid in _plan_in:
        if _cid not in _by_id:
            continue
        _fact = target_survival(_by_id[_cid], _plan_in, _by_id) or {}
        if _fact.get("signal") == "all_targets_removed":
            _nullifiers |= set(_fact.get("removed_by_earlier")
                               if "removed_by_earlier" in _fact
                               else (_fact.get("removed_by") or []))
    _probes = dict(state.get("deleter_probes") or {})
    for _d in sorted(_nullifiers - never_builds):
        if _d not in _probes:
            _ok, _note = _builds_alone(
                _d, p, round_dir(state) or Path(p["out_dir"]))
            _probes[_d] = {"builds_alone": _ok, "note": _note,
                           "why": ("probed because another candidate's "
                                   "nullification depends on it")}
            print(f"      deleter probe {_d}: builds_alone={_ok} -- {_note[:90]}")
        if _probes[_d].get("builds_alone") is False:
            never_builds.add(_d)

    r = gate(state.get("round", 1), _plan_in,
             all_candidates, box, never_builds=never_builds,
             domain=p.get("domain", ""))

    enforced = state.get("gate_enforced", True)

    enforced_blocked = list(r.blocked) if enforced else []
    plan_out = list(r.plan_out) if enforced else list(r.plan_in)
    blocked = enforced_blocked

    ruling = {"round": state.get("round", 1), "enforced": enforced, **asdict(r)}

    ruling["plan_out"] = list(plan_out)
    ruling["blocked"] = list(enforced_blocked)
    ruling["pruned"] = list(r.pruned) if enforced else []

    ruling["deleter_probes"] = {k: dict(v) for k, v in _probes.items()}

    rd = round_dir(state)
    if rd is not None:
        try:
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "gate_ruling.json").write_text(
                json.dumps(ruling, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")
        except Exception as e:
            print(f"      ! could not persist the gate ruling: {type(e).__name__}: {e}")

    return {
        "current_plan": plan_out,
        "gate_ruling": ruling,
        "deleter_probes": _probes,
        "blocked_candidates": blocked,

        "compliance_advisory": [{"round": state.get("round", 1),
                                 "blocked": list(enforced_blocked)}],
        "trace": [{"node": "ARBITER_GATE", "round": state.get("round", 1),
                   "lookups": box.audit()["total_reads"],
                   "plan_in": r.plan_in, "plan_out": plan_out,
                   "pruned": r.pruned if enforced else [],
                   "compliance_blocked": list(enforced_blocked),
                   **({} if enforced else
                      {"NOT_ENFORCED": {"would_prune": r.pruned,
                                        "would_block": r.blocked}}),
                   **({"halted": True} if r.halted and enforced else {}),
                   **({"rollback": r.rollback_note} if r.rollback_note and enforced else {})}],
    }

def route_after_gate(state: MABPRState) -> str:
    """An empty plan means compliance blocked every candidate. makes"""

    return "APPLY" if (state.get("current_plan") or []) else END

def proposer_node(state: MABPRState) -> dict:
    """Turn the Critic's directive into the next candidate set — selection and"""

    all_candidates = json.loads(
        Path(state["paths"]["detector_results"]).read_text(encoding="utf-8"))

    kb = KnowledgeBox(state["paths"]["detector_results"],
                      overlay_path=state["paths"].get("overlay"))
    proposal = propose(
        state.get("round", 1) + 1,
        all_candidates,
        state.get("footprints", {}),
        state.get("assessment"),
        knowledge=kb,
        current_plan=list(state.get("current_plan") or []),
        tried_plans=state.get("tried_plans"),
        anti_combos=state.get("anti_combos"),

        contested=((state.get("bundle") or {}).get("scorecard")
                   or {}).get("accounting", {}).get("contested_activities"),
        blocked=state.get("blocked_candidates"),
        build_history=state.get("build_history"),
        harmful_history=state.get("harmful_history"),
        ledger=state.get("candidate_ledger"),
        gate_ruling=state.get("gate_ruling"),
    )

    prev_plan = list(state.get("current_plan") or [])
    expl = explain_round(
        state.get("round", 1), previous_plan=prev_plan,
        new_plan=list(proposal.ordering),
        scorecard=(state.get("bundle") or {}).get("scorecard") or {},
        assessment=state.get("assessment") or {},
        proposal=asdict(proposal), gate_ruling=state.get("gate_ruling"),
        build_history=state.get("build_history"))
    rd = round_dir(state)
    if rd is not None:
        write_round_explanation(rd, expl)

    return {
        "round": int(state.get("round", 1)) + 1,
        "explanation": expl,

        "directive_followthrough": (expl or {}).get("directive_followthrough") or {},
        "current_plan": list(proposal.ordering),
        "proposal": asdict(proposal),
        "knowledge_log": kb.audit(),
        "mode": "normal",
        "confirm_per_round": 1,
        "trace": [{"node": "PROPOSER", "round": int(state.get("round", 1)) + 1,
                   "plan": proposal.ordering, "responds_to": proposal.responds_to,
                   "priors_read": kb.audit()["total_reads"],
                   "prior_vs_measured": proposal.prior_vs_measured or None,
                   "fallback": proposal.fallback or None}],
    }

def _traced(name: str, fn):
    """Point the LLM trace sink at this round's directory before the node runs."""

    def _node(state: MABPRState) -> dict:
        set_trace_sink(round_dir(state), f"{name}@round{state.get('round', 1)}")
        try:
            return fn(state)
        finally:
            set_trace_sink(None)
    _node.__name__ = getattr(fn, "__name__", name)
    return _node

def build_graph(simulate_fn: SimulateFn | None = None, *, checkpointer=None):
    """Wire and compile. Every node really runs; nothing is a stub."""
    g = StateGraph(MABPRState)

    for name, fn in [
        ("APPLY", apply_node),
        ("SIMULATE", make_simulate_node(simulate_fn or prosimos_simulate)),
        ("MEASURE", measure_node),
        ("CRITIC_INVESTIGATE", critic_investigate_node),
        ("CRITIC_CONCLUDE", critic_conclude_node),
        ("ARBITER_CALIBRATE", arbiter_calibrate_node),
        ("PROPOSER", proposer_node),
        ("ARBITER_GATE", arbiter_gate_node),
    ]:
        g.add_node(name, _traced(name, fn))

    g.add_edge(START, "ARBITER_GATE")
    g.add_conditional_edges("MEASURE", route_after_measure,
                            {"CRITIC_INVESTIGATE": "CRITIC_INVESTIGATE",
                             "CRITIC_CONCLUDE": "CRITIC_CONCLUDE"})

    g.add_conditional_edges("CRITIC_INVESTIGATE", route_after_investigate,
                            {"APPLY": "APPLY", "CRITIC_CONCLUDE": "CRITIC_CONCLUDE"})

    g.add_conditional_edges("CRITIC_CONCLUDE", route_after_conclude,
                            {"ARBITER_CALIBRATE": "ARBITER_CALIBRATE",
                             "APPLY": "APPLY", END: END})
    g.add_edge("ARBITER_CALIBRATE", "PROPOSER")
    g.add_edge("PROPOSER", "ARBITER_GATE")

    g.add_conditional_edges("ARBITER_GATE", route_after_gate,
                            {"APPLY": "APPLY", END: END})

    g.add_conditional_edges("APPLY", route_after_apply,
                            {"SIMULATE": "SIMULATE", "MEASURE": "MEASURE"})
    g.add_edge("SIMULATE", "MEASURE")

    return g.compile(checkpointer=checkpointer or MemorySaver())

def edge_count() -> dict:
    """The paper's measurable property: coordination cost is bounded by the graph,"""

    gr = build_graph().get_graph()
    return {"nodes": len(gr.nodes), "edges": len(gr.edges)}

def initial_state(round_no: int = 1, *, max_rounds: int = 12,
                  n_runs: int | None = None, dataset: str | None = None,
                  gate_enforced: bool = True, run_id: str | None = None) -> MABPRState:
    """Round 1 starts from the plan run_compare already measured."""

    ds = datasets.get(dataset)
    paths = ds.paths()

    if run_id is None:
        from datetime import datetime
        run_id = (f"{'gated' if gate_enforced else 'ungated'}-"
                  f"{datetime.now().strftime('%m%d-%H%M')}")
    paths["run_id"] = run_id

    paths["overlay"] = str(Path(paths["out_dir"]) / "loop" / run_id
                           / "knowledge_overlay.json")
    overlay = Path(paths["overlay"])
    if overlay.exists():
        overlay.unlink()

    comparison = Path(paths["compare_dir"]) / "comparison.json"
    if not comparison.exists():
        raise FileNotFoundError(
            f"{comparison} is missing — the loop needs the fixed uncalibrated BASELINE "
            f"to measure against (M2). Run:  python -m mabpr.run_detectors --dataset "
            f"{ds.tag}  ->  python -m mabpr.apply_redesign --dataset {ds.tag}"
            f"  ->  python -m mabpr.run_compare --dataset {ds.tag}")
    payload = json.loads(comparison.read_text(encoding="utf-8"))

    _recomputed = None
    if any(f"{k}_mean" not in payload.get("baseline", {}) for k, _, _ in REPORT_KPIS):
        csvs = sorted(Path(paths["compare_dir"]).glob("baseline_sim_*.csv"))
        if csvs:
            payload["baseline"] = aggregate_across_runs([kpi_from_csv(c) for c in csvs])
            payload.setdefault("config", {})["n_runs"] = len(csvs)
            _recomputed = len(csvs)
            print(f"[baseline] recomputed in memory from {len(csvs)} runs on disk "
                  f"(the stored aggregate predates a KPI). The shared "
                  f"comparison.json is NOT rewritten — use "
                  f"`run_compare --refresh` for that.")

    _bl = Path(paths["out_dir"]) / "loop" / run_id / "baseline_used.json"
    _bl.parent.mkdir(parents=True, exist_ok=True)
    _bl.write_text(json.dumps(
        {"source": str(comparison),
         "recomputed_in_memory_from_n_csvs": _recomputed,
         "shared_file_rewritten": False,
         "baseline": payload.get("baseline"),
         "config": payload.get("config")},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    _catalogue = json.loads(Path(paths["detector_results"]).read_text(encoding="utf-8"))

    _nosplit = candidates_without_a_split(_catalogue)
    if _nosplit:
        raise FileNotFoundError(
            f"{_nosplit} carry no `extraneous_delay_split`, so Case-based Work could "
            f"only refuse in this run: the detector output for {ds.tag} predates that "
            f"step. Run:  python -m mabpr.run_detectors --dataset {ds.tag}")
    plan = [c["candidate_id"] for c in _catalogue]
    return {
        "round": round_no,
        "paths": paths,
        "sim_config": {"n_runs": n_runs or payload["config"]["n_runs"],
                       "total_cases": payload["config"]["total_cases"],
                       "starting_at": ds.starting_at,
                       "dataset": ds.tag},
        "base_agg": payload["baseline"],
        "mode": "normal",
        "current_plan": plan,
        "anti_combos": [],
        "build_history": {},
        "harmful_history": {},
        "rounds_without_improvement": 0,

        "best_agg": {},

        "sim_budget": 20, "confirm_budget": 4, "confirm_per_round": 1,
        "max_rounds": max_rounds,
        "gate_enforced": gate_enforced,
        "trace": [],
    }

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if "--draw" in sys.argv or len(sys.argv) == 1:
        app = build_graph()
        print(app.get_graph().draw_mermaid())
        print(f"\n# {edge_count()}")
        out = REPO_ROOT / "docs" / "figures" / "mabpr_graph.mmd"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(app.get_graph().draw_mermaid(), encoding="utf-8")
        print(f"# written -> {out}")

    if "--run" in sys.argv:
        ds = datasets.from_argv(sys.argv)

        enforced = "--no-gate" not in sys.argv
        rounds = 12
        for i, a in enumerate(sys.argv):
            if a == "--max-rounds" and i + 1 < len(sys.argv):
                rounds = int(sys.argv[i + 1])
        print(f"dataset {ds.tag}: {ds.baseline_bpmn.name}, starting_at={ds.starting_at}, "
              f"max_rounds={rounds}, "
              f"gate={'ENFORCED' if enforced else 'RECORDED BUT NOT ENFORCED (ablation)'}")
        tag = None
        for i, a in enumerate(sys.argv):
            if a == "--tag" and i + 1 < len(sys.argv):
                tag = sys.argv[i + 1]
        state = initial_state(dataset=ds.tag, gate_enforced=enforced,
                              max_rounds=rounds, run_id=tag)
        rid = state["paths"]["run_id"]
        print(f"run_id  {rid}  ->  {Path(state['paths']['out_dir']) / 'loop' / rid}")
        app = build_graph()
        try:
            final = app.invoke(
                state,
                config={"configurable": {"thread_id": f"loop-{ds.tag}-{rid}"},

                        "recursion_limit": 150})
        except Exception as e:

            print(f"\n[loop FAILED: {type(e).__name__}: {e}]")
            try:
                from mabpr.show_run import rebuild_result
                rp = rebuild_result(ds.tag, rid)
                print(f"partial result rebuilt from the rounds on disk -> {rp}\n"
                      f"⚠️  PARTIAL: the loop did not converge, so this reports the "
                      f"best measurement among the rounds that COMPLETED, and it "
                      f"cannot know what the gate blocked (the ruling is not "
                      f"persisted).")
            except Exception as e2:
                print(f"[could not rebuild a partial result: {type(e2).__name__}: {e2}]")
            raise
        print_trajectory(final)

        try:
            final = {**final, **compliance_review(final)}
        except Exception as e:
            print(f"\n[final compliance review failed: {type(e).__name__}: {e}]")
        try:
            rp = write_final_report(final, sig_better=_sig_better,
                                    select_fn=_select_fn(final))
            print("\n" + render_final(json.loads(rp.read_text(encoding="utf-8"))))
            print(f"\nwritten -> {rp}")
        except Exception as e:
            print(f"\n[final report failed: {type(e).__name__}: {e}]")
        print(json.dumps(final["assessment"], ensure_ascii=False, indent=2, default=str))

def _sig_better(a: dict, b: dict, n_runs: int) -> bool:
    """Is measurement ``a`` SIGNIFICANTLY better (lower) than ``b``?"""

    if a is b:
        return False
    sa, sb = a.get("median_std", 0.0) or 0.0, b.get("median_std", 0.0) or 0.0
    se = math.sqrt((sa ** 2 + sb ** 2) / max(1, n_runs))
    if se == 0:
        return a["median_h"] < b["median_h"]
    return (a["median_h"] - b["median_h"]) / se < IMPROVE_Z

def _select_fn(final: dict):
    """Give `write_final_report` a Selector bound to this run's accounts."""

    from mabpr.agents.selector import build_options, select

    run = Path(final["paths"]["out_dir"]) / "loop" / final["paths"].get("run_id", "")
    accounts = {}
    for f in sorted(run.glob("round*/explanation.json")):
        try:
            accounts[f.parent.name] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass

    def _fn(pool: list) -> dict:
        opts = build_options(pool, accounts)
        sel = select(opts)
        print(f"\n  selection: {sel.chosen}  [{sel.gain_reading}]  {sel.why[:110]}")
        if sel.passengers:
            for p_ in sel.passengers:
                if p_.get("kind") not in (None, "", "none"):
                    print(f"      passenger {p_.get('candidate')}: {p_.get('kind')}")
        if sel.fallback:
            print(f"      ! {sel.fallback}")
        return asdict(sel)

    return _fn

def compliance_review(final: dict) -> dict:
    """THE FINAL compliance judgement — which POOL the deliverable comes from."""

    tried = [t for t in (final.get("tried_plans") or []) if t.get("median_h")]
    if not tried:
        return {}

    reviewed: list = []
    for t in tried:
        for cid in (t.get("ordering") or []):
            if cid not in reviewed:
                reviewed.append(cid)
    if not reviewed:
        return {}

    p = final["paths"]
    all_candidates = json.loads(Path(p["detector_results"]).read_text(encoding="utf-8"))
    box = ArbiterBox(p["detector_results"], overlay_path=p["overlay"])
    print(f"\n  final compliance review (on every candidate any measured plan "
          f"contained): {reviewed}")
    r = gate(0, reviewed, all_candidates, box, domain=p.get("domain", ""))
    barred = sorted(set(r.blocked) & set(reviewed))

    advisory = final.get("compliance_advisory") or []
    per_round = sorted({c for a in advisory for c in (a.get("blocked") or [])})
    eligible = [t for t in tried if not (set(t.get("ordering") or []) & set(barred))]
    print(f"      barred: {barred or 'nothing'}"
          + (f"   (the rounds themselves blocked: {per_round})" if per_round else ""))
    print(f"      the answer will be chosen from {len(eligible)} of "
          f"{len(tried)} measured plans")
    return {
        "blocked_candidates": barred,
        "compliance_final": {
            "candidates_reviewed": reviewed,
            "barred": barred,
            "rulings": r.compliance,
            "asked": ("every round during the loop, and once more here on the "
                      "union of everything measured"),
            "eligible_measurements": len(eligible),
            "measurements_total": len(tried),

            "per_round_blocks_during_loop": advisory,
            "note": ("DECISIONS B: judged on prior knowledge alone and never "
                     "overturnable by measurement. This ruling decides what the "
                     "deliverable may contain; measurements of plans it bars are "
                     "kept under `excluded_as_non_compliant` so the cost of the "
                     "block can still be read."),
        },
    }

_TRAJ_KEYS = ("throughput_h_median", "throughput_h_mean",
              "waiting_h_per_case_mean", "duration_h_per_case_mean",
              "utilization_mean")
_TRAJ_LABELS = ("median", "mean", "waiting", "work", "util")

def print_trajectory(final: MABPRState) -> None:
    """What the loop actually did — the part a scalar score would hide."""
    base = final["base_agg"][f"{KPI}_mean"]
    print("\n" + "=" * 78)
    print(f"TRAJECTORY   baseline {KPI} = {base:.4f} h")

    print(f"  {'round':>5} " + " ".join(f"{lbl:>9}" for lbl in _TRAJ_LABELS)
          + "   plan     (* = |z|>3)")
    rows = final.get("tried_plans", [])
    n_runs = int(final.get("sim_config", {}).get("n_runs", 20))

    lowest = min(rows, key=lambda p: p["median_h"], default=None)
    tied = [p for p in rows if lowest and not _sig_better(lowest, p, n_runs)]
    for p in rows:
        med, conf = p["median_h"], p.get("mode") == "confirming"
        label = f"{p['round']:>4}c" if conf else f"{p['round']:>5}"
        mark = ("  <- best" if p is lowest and len(tied) == 1
                else "  <- best (tied)" if p in tied else "")
        kp = p.get("kpis") or {}
        cells = []
        for key in _TRAJ_KEYS:
            e = kp.get(key) or {}
            pct = e.get("pct")
            if pct is None and key == "throughput_h_median":
                pct = p.get("pct_vs_baseline")
            cells.append("        ?" if pct is None else
                         f"{pct:>8.1f}%{'*' if e.get('significant') else ' '}")
        print(f"  {label} " + " ".join(cells) + f"   {p['ordering']}{mark}")
        ce = p.get("case_endings") or {}
        if ce.get("significant"):
            print(f"  {'':>5}   case outcomes: {ce.get('headline', '')}")
    if lowest and len(tied) > 1:
        names = ", ".join(f"{p['round']}{'c' if p.get('mode') == 'confirming' else ''}"
                          for p in tied)
        print(f"  ({len(tied)} measurements are statistically indistinguishable: {names} "
              f"— |z| < 3 between them, so there is no ranking to report)")
    print("  (Nc = the confirmation simulation spent inside round N — a measured "
          "design, not a round)")

    if any(r.get("kpis") for r in rows):
        keys = [k for k, _, _ in REPORT_KPIS
                if any((r.get("kpis") or {}).get(k) for r in rows)]
        print("\n  ALL MEASURES (Δ vs baseline, and its own z — * = |z| > 3)")
        head = "  " + f"{'round':>5} "
        for k in keys:
            head += f"{next(l for kk, l, _ in REPORT_KPIS if kk == k):>21}"
        print(head)
        for r in rows:
            km = r.get("kpis") or {}
            if not km:
                continue
            lbl = f"{r['round']:>4}c" if r.get("mode") == "confirming" else f"{r['round']:>5}"
            line = f"  {lbl} "
            for k in keys:
                e = km.get(k)
                if not e:
                    line += f"{'':>21}"; continue
                line += (f"{e['pct']:>+9.1f}% {e['z']:>+7.1f}"
                         f"{'*' if e['significant'] else ' '}" + " " * 2)
            print(line)
        print("  (the loop converges on the FIRST column only — DECISIONS F needs one "
              "rule — but a candidate or a plan that moves any of the others has an "
              "effect, and the account must say so)")

    if rows and lowest and rows[-1] not in tied:
        print(f"\n  ⚠ the FINAL round is SIGNIFICANTLY worse than the best measured "
              f"one ({lowest['round']}{'c' if lowest.get('mode') == 'confirming' else ''} "
              f"at {lowest['pct_vs_baseline']:+.1f}%). The loop stops on a RULE "
              f"(budget / patience), not on optimality — there is no plan score to "
              f"optimise (M1), so the last row is where it stopped, not its answer.")

    print("\nNODES VISITED")
    for t in final.get("trace", []):
        node = t.get("node")
        if node == "SIMULATE":
            extra = f"mode={t.get('mode')} runs={t.get('runs')} budget_left={t.get('sim_budget_left')}"
        elif node == "CRITIC_INVESTIGATE":
            c = t.get("confirmation", {})
            extra = (f"round={t.get('round')} hypotheses={t.get('hypotheses')} "
                     f"reads={t.get('reads')} confirmation={c.get('granted')} ({c.get('reason')})")
        elif node == "PROPOSER":

            extra = (f"round={t.get('round')} priors_read={t.get('priors_read')} "
                     f"plan={t.get('plan')}")
        elif node == "ARBITER_GATE":
            extra = (f"round={t.get('round')} lookups={t.get('lookups')} "
                     f"pruned={t.get('pruned')} compliance_blocked={t.get('compliance_blocked')}")
            if t.get("rollback"):
                extra += "  [rolled back the non-compliance prunes]"
            if t.get("halted"):
                extra += "  [HALTED: compliance blocked every candidate]"
        elif node == "ARBITER_CALIBRATE":
            extra = (f"round={t.get('round')} lookups={t.get('lookups')} "
                     f"accepted={t.get('accepted')} refused={t.get('refused')}")
        elif node == "APPLY":
            extra = f"mode={t.get('mode')} plan={t.get('plan')}"
            if t.get("structural"):
                extra += f"  [Phase-3 refused, order cannot fix: {t['structural']}]"
            if t.get("reused"):
                extra += f"  [SIMULATION SKIPPED — {t['reused']}]"
        elif node == "CRITIC_CONCLUDE":
            d = t.get("directive") or {}
            extra = f"round={t.get('round')} directive={d.get('intent')} {d.get('targets')}"

            v = t.get("verdicts") or {}
            if v:
                extra += "\n" + " " * 23 + "verdicts: " + ", ".join(
                    f"{k}={val}" for k, val in sorted(v.items()))
        else:
            extra = " ".join(f"{k}={v}" for k, v in t.items() if k != "node")
        print(f"  {node:20} {extra}")

    print(f"\nBUDGETS  sim={final.get('sim_budget')}  confirmations={final.get('confirm_budget')}"
          f"  rounds_without_improvement={final.get('rounds_without_improvement')}")
    if final.get("blocked_candidates"):
        print(f"COMPLIANCE-BLOCKED  {final['blocked_candidates']}")
    cs = final.get("contradictions") or []
    if cs:
        print(f"\nWRITTEN BACK TO KNOWLEDGE ({len(cs)} contradictions)")
        for c in cs:
            mark = "✓" if c.get("accepted") else "✗"
            print(f"  {mark} [{c.get('id')}] {c.get('action'):22} {c.get('subject')}")
            print(f"      prior: {str(c.get('prior_said'))[:90]}")
            print(f"      meas : {str(c.get('measured_said'))[:90]}")
            if not c.get("accepted"):
                print(f"      refused: {c.get('refused_reason')}")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
