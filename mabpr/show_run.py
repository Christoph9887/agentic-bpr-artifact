"""Show a finished run: every round's plan and measurement, then result.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mabpr import datasets
from mabpr.explain import render, render_final, write_final_report

def _tag_from_argv(argv: list[str], default: str = "v9") -> str:
    for i, a in enumerate(argv):
        if a == "--tag" and i + 1 < len(argv):
            return argv[i + 1]
    return default

def _merged_names_from_disk(round_dir, detector_results=None) -> list:
    """Every RENAME this round made, read from the artefacts it left behind."""

    import json as _json
    from pathlib import Path as _Path

    log = _Path(round_dir) / "phase3" / "phase3_log.json"
    if not log.exists():
        return []
    try:
        entries = _json.loads(log.read_text(encoding="utf-8")).get("entries") or []
    except Exception:
        return []
    out = []
    for e in entries:
        name = (e.get("adapt_report") or {}).get("composite_name")
        if name and name not in out:
            out.append(name)
    if detector_results:
        try:
            from mabpr.evidence.footprint import read_footprints, rename_records
            out = out + rename_records(
                read_footprints(_Path(round_dir) / "phase3", detector_results))
        except Exception:
            pass
    return out

def _replay_no_gain(run: Path, aggs: dict, n_runs: int) -> int:
    """Re-run the loop's own stopping counter over what is on disk."""

    from mabpr.apply_redesign import design_fingerprint
    from mabpr.graph import convergence

    rounds = sorted((d for d in run.glob("round*") if d.name[5:].isdigit()),
                    key=lambda d: int(d.name[5:]))
    fps: dict = {}
    incumbent, no_gain = None, 0
    for d in rounds:
        fp = design_fingerprint(d / "phase3") or ""
        agg = aggs.get(d.name)
        if agg is None and fp:
            src = next((n for n, f in fps.items() if f == fp and n in aggs), None)
            agg = aggs.get(src) if src else None
        fps[d.name] = fp
        if agg is None:
            continue
        _, no_gain, incumbent = convergence(agg, incumbent, n_runs, no_gain, baseline_z=0.0)
    return no_gain

def _round_no(p) -> int:
    """The round number from a directory name, suffix and all."""

    tail = p.name.split("round")[-1]
    digits = ""
    for ch in tail:
        if not ch.isdigit():
            break
        digits += ch
    return int(digits or 0)

def _suffix_no(p) -> int:
    tail = p.name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() and "_" in p.name else 1

def rebuild_result(dataset: str, tag: str, *, select: bool = True) -> Path:
    """Regenerate result.json from what the run left on disk."""

    from mabpr.graph import _select_fn, _sig_better, multi_kpi
    from mabpr.measure import (aggregate_across_runs, compare_case_endings,
                               kpi_from_csv, scalar_z)

    ds = datasets.get(dataset)
    run = ds.out_dir / "loop" / tag
    base = json.loads((ds.compare_dir / "comparison.json").read_text(encoding="utf-8"))
    base_agg, n_runs = base["baseline"], int(base["config"]["n_runs"])
    KPI = "throughput_h_median"

    gate_blocked, saw_ruling = set(), False
    for g in run.glob("*round*/gate_ruling.json"):
        r = json.loads(g.read_text(encoding="utf-8"))
        saw_ruling = True
        if r.get("enforced", True):
            gate_blocked |= set(r.get("blocked") or [])

    contradicted = set()
    for cid in sorted(gate_blocked):
        for p3 in run.glob("*round*/phase3/phase3_log.json"):
            entries = json.loads(p3.read_text(encoding="utf-8")).get("entries") or []
            if any(e.get("candidate_id") == cid
                   and (e.get("adapt_report") or {}).get("applied") for e in entries):
                contradicted.add(cid)
                break
    if contradicted:
        print(f"  ! gate_ruling says {sorted(contradicted)} were blocked, but Phase-3 "
              f"BUILT them — trusting the build. (Runs before 2026-08-14 persisted "
              f"the raw per-round answer instead of the enforced outcome.)")
        gate_blocked -= contradicted

    prior = {}
    rp = run / "result.json"
    if rp.exists():
        try:
            prior = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            prior = {}

    tried, blocked, aggs = [], [], {}
    for d in sorted(run.glob("*round*"), key=lambda p: (_round_no(p),
                                                        p.name.startswith("confirm"),
                                                        _suffix_no(p))):
        expl = d / "explanation.json"
        summ, csvs = d / "summary.json", sorted((d / "sim").glob("sim_*.csv"))
        if summ.exists():
            js = json.loads(summ.read_text(encoding="utf-8"))
            agg, n = js["aggregate"], js["n_runs"]
        elif csvs:

            usable = []
            for c in csvs:
                try:
                    usable.append(kpi_from_csv(c))
                except Exception as e:
                    print(f"  [rebuild] {d.name}/{c.name} unreadable "
                          f"({type(e).__name__}) — skipped")
            if not usable:
                print(f"  [rebuild] {d.name}: no readable simulation log "
                      f"({len(csvs)} file(s) present but none parsed); this round "
                      f"contributes no measurement")
                continue
            if len(usable) < len(csvs):
                print(f"  [rebuild] {d.name}: {len(usable)} of {len(csvs)} runs "
                      f"readable — its z is computed over fewer runs than the rest")
            agg, n = aggregate_across_runs(usable), len(usable)
        else:
            continue
        aggs[d.name] = agg
        delta, z = scalar_z(base_agg, agg, KPI, n_runs)

        p3 = d / "phase3" / "phase3_log.json"
        if expl.exists() and not d.name.startswith("confirm"):
            plan = json.loads(expl.read_text(encoding="utf-8")).get("plan_before") or []
        elif p3.exists():
            plan = [e["candidate_id"] for e in
                    json.loads(p3.read_text(encoding="utf-8")).get("entries", [])]
        else:
            plan = []

        built = None
        if p3.exists():
            from mabpr.evidence.footprint import _changed_anything
            _ents = json.loads(p3.read_text(encoding="utf-8")).get("entries", [])
            built = [e["candidate_id"] for e in _ents
                     if (e.get("adapt_report") or {}).get("applied")
                     and _changed_anything(e.get("adapt_report") or {})]
        tried.append({
            "round": int(d.name.split("round")[1]),
            "mode": "confirming" if d.name.startswith("confirm") else "normal",
            "ordering": plan, "built": built, "median_h": agg[f"{KPI}_mean"],
            "median_std": agg[f"{KPI}_std"], "vs_baseline_z": z,

            "kpis": multi_kpi(base_agg, agg, n_runs),

            "case_endings": compare_case_endings(
                base_agg, agg, n_runs,
                merged=_merged_names_from_disk(d, ds.detector_results)),
            "pct_vs_baseline": delta / base_agg[f"{KPI}_mean"] * 100})
        if expl.exists():
            for e in json.loads(expl.read_text(encoding="utf-8")).get("dropped") or []:
                if "COMPLIANCE" in str(e.get("why_dropped") or ""):
                    blocked.append(e["candidate"])

    build_history: dict = {}
    for p3 in sorted(run.glob("*round*/phase3/phase3_log.json")):
        rnd = int(p3.parent.parent.name.split("round")[1])
        for e in json.loads(p3.read_text(encoding="utf-8")).get("entries") or []:
            ar = e.get("adapt_report") or {}
            build_history.setdefault(e["candidate_id"], []).append(
                {"round": rnd, "applied": bool(ar.get("applied")),
                 "reason": ar.get("reason") or ""})

    state = {"paths": {"out_dir": str(ds.out_dir), "run_id": tag,
                       "detector_results": str(ds.detector_results),
                       "overlay": str(ds.overlay)},
             "build_history": build_history,
             "sim_config": {"n_runs": n_runs, "dataset": ds.tag},
             "base_agg": base_agg, "tried_plans": tried, "gate_enforced": True,

             "sim_budget": 20 - len(aggs),
             "rounds_without_improvement": _replay_no_gain(run, aggs, n_runs),
             "current_plan": tried[-1]["ordering"] if tried else [],

             "compliance_final": prior.get("compliance_final"),
             "compliance_advisory": ((prior.get("compliance_final") or {})
                                     .get("per_round_blocks_during_loop")
                                     or [{"round": None, "blocked": sorted(gate_blocked)}]),
             "blocked_candidates": (
                 sorted((prior.get("compliance_final") or {}).get("barred") or [])
                 if prior.get("compliance_final")
                 else (sorted(gate_blocked) if saw_ruling
                       else sorted(set(blocked)) or None))}

    out = write_final_report(state, sig_better=_sig_better,
                             select_fn=_select_fn(state) if select else None)

    if not select and prior.get("selection"):
        rep = json.loads(out.read_text(encoding="utf-8"))
        if not rep.get("selection"):
            rep["selection"] = prior["selection"]
            rep["selection_carried_forward"] = (
                "from the previous result.json — this rebuild ran with --no-select, "
                "and a selection cannot be re-derived from the round directories")
            out.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str),
                           encoding="utf-8")

    return out

def show(dataset: str, tag: str, *, rounds: bool = True) -> None:
    ds = datasets.get(dataset)
    run = ds.out_dir / "loop" / tag
    print("=" * 78)
    print(f"BPI {ds.tag}   run {tag}   {run}")
    print("=" * 78)
    if not run.exists():
        print("  (no such run on disk)")
        return

    result = run / "result.json"
    if result.exists():
        r = json.loads(result.read_text(encoding="utf-8"))
        print(render_final(r))
    else:
        print("  result.json not written — the run did not finish")

    if not rounds:
        return
    print("\n" + "-" * 78)
    print("PER-ROUND ACCOUNT")
    print("-" * 78)
    for d in sorted(run.glob("round*"), key=lambda p: int(p.name[5:] or 0)):
        f = d / "explanation.json"
        if not f.exists():
            print(f"\n{d.name}: no explanation.json")
            continue
        print("\n" + render(json.loads(f.read_text(encoding="utf-8"))))

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    tag = _tag_from_argv(sys.argv)
    tags = ([datasets.from_argv(sys.argv).tag] if "--dataset" in sys.argv
            else ["2012", "2017"])
    for t in tags:
        if "--rebuild" in sys.argv:
            print(f"rebuilt -> "
                  f"{rebuild_result(t, tag, select='--no-select' not in sys.argv)}")
        show(t, tag, rounds="--no-rounds" not in sys.argv)
        print()

if __name__ == "__main__":
    main()
