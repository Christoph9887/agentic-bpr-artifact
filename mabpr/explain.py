"""EXPLANATION — why each candidate was kept or dropped, and what the kept ones did."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

def _contribution(sc_row: dict | None) -> dict:
    """The four-column account, compressed to what a reader needs."""
    if not sc_row:
        return {"measured": False, "reason": "not applied this round"}
    wait = sc_row.get("waiting") or []
    dur = sc_row.get("duration") or []
    res = (sc_row.get("resources") or {}).get("significant") or []

    net = sc_row.get("net") or {}
    thr = net.get("throughput_h_per_case")
    thr_sig = bool(net.get("throughput_significant"))
    gains, losses = net.get("measures_improved"), net.get("measures_worsened")
    out: dict[str, Any] = {
        "measured": True,
        "throughput_h_per_case": thr,
        "throughput_sd": net.get("throughput_h_per_case_sd"),
        "throughput_z": net.get("throughput_h_per_case_z"),
        "throughput_significant": thr_sig,
        "waiting": [f"{w['activity']} z={w['z']:+.1f}" for w in wait],
        "duration": [f"{d['activity']} z={d['z']:+.1f}" for d in dur],
        "resources_significant": len(res),
    }
    if not net.get("any_measurable_effect", bool(wait or dur or res)):

        out["summary"] = (
            "nothing attributable to it alone — every measure it touches is shared"
            if sc_row.get("unattributable_activities")
            else "no measurable effect on any of the four columns")
    else:
        out["summary"] = (
            f"{gains} improved / {losses} worsened activity measures"
            + ("" if thr is None else
               f"; {thr:+.3f} h per case"
               + (f" (z={net['throughput_h_per_case_z']:+.1f})" if thr_sig
                  else f" — NOT significant (+/-{net.get('throughput_h_per_case_sd') or 0:.3f}), "
                       f"do not read as an effect")))
    sp = sc_row.get("spillover")
    if sp:
        out["spillover_h_per_case"] = sp.get("h_per_case")
        out["h_per_case_including_spillover"] = net.get("h_per_case_including_spillover")
        out["spillover_note"] = (
            f"{sp['h_per_case']:+.3f} h per case landed on "
            f"{len(sp.get('activities') or [])} activities it does not touch"
            + (f" (z={sp['z']:+.1f})" if sp.get("significant") else " — not significant")
            + "; it was the only candidate applied, so that is its doing too")
    if sc_row.get("shared_with"):
        sh = net.get("throughput_h_per_case_shared")
        out["attribution_caveat"] = (
            f"shares {sc_row['unattributable_activities']} with "
            f"{sc_row['shared_with']} — that part is attributable to neither"
            + (f" ({sh:+.3f} h per case left unassigned)" if sh else ""))

    noop = sc_row.get("structural_noop") or {}
    if noop:
        out["summary"] = (
            "the handler made NO edit to the model — it aimed at "
            f"{noop.get('aimed_at') or 'nothing resolvable'} and changed none of "
            "it, so there is nothing for the four columns to measure")
    exo = sc_row.get("exogenous_saving") or {}
    if exo:
        out["exogenous_saving"] = (
            f"the log says it also removes {exo['hours_per_occurrence']:.3f} h per "
            f"occurrence ({exo['field']}) — {exo['why']}. The simulator cannot see "
            f"this, so none of the columns above contain it; it is an estimate, "
            f"not a measurement.")
        if exo.get("DISAGREES_WITH_MEASUREMENT"):
            out["exogenous_conflict"] = exo["DISAGREES_WITH_MEASUREMENT"]

    absorbed = sc_row.get("exogenous_absorbed_by_the_edit")
    if absorbed:
        out["exogenous_absorbed_by_the_edit"] = absorbed
    iso = sc_row.get("isolated_in_earlier_rounds") or {}
    if iso:
        out["attribution_settled_by"] = [
            f"'{a}': {h['alone']} was measured alone at {h['h_per_case']:+.3f} h/case "
            f"in round {h['round']}{'c' if h.get('mode') == 'confirming' else ''}"
            for a, hits in iso.items() for h in hits]
    return out

def _contradiction(directive: dict | None, prev: list, new: list) -> str:
    """One sentence when the new plan does the opposite of what the directive asked."""

    d = directive or {}
    intent = str(d.get("intent") or "")
    targets = [str(t) for t in (d.get("targets") or [])]
    gone = [t for t in targets if t in prev and t not in new]
    if intent == "keep" and gone:
        return (f"the directive asked to KEEP {gone} and the plan dropped "
                f"{'it' if len(gone) == 1 else 'them'}"
                + (f" (the Critic's intent {d['unexecutable_intent']!r} was not "
                   f"executable and code read it as `keep`)"
                   if d.get("unexecutable_intent") else ""))
    if intent == "drop":
        stayed = [t for t in targets if t in new]
        if stayed:
            return f"the directive asked to DROP {stayed} and the plan kept them"
    if intent == "add":
        absent = [t for t in targets if t not in new]
        if absent:
            return f"the directive asked to ADD {absent} and the plan omits them"
    return ""

def _handler_self_report(build_history: "dict | None", cid: str,
                         scorecard: "dict | None" = None) -> dict:
    """The most recent handler report for *cid*, or {} if it never built."""

    runs = (build_history or {}).get(cid) or []
    for rec in reversed(runs):
        if rec.get("applied") and rec.get("handler"):
            return dict(rec["handler"])
    if any(rec.get("applied") for rec in runs):
        return {"unrecorded": ("this candidate built successfully but no handler "
                               "report was kept for it — absence of a record, NOT "
                               "evidence that the handler changed nothing")}
    return {}

def explain_round(round_no: int, *, previous_plan: list[str], new_plan: list[str],
                  scorecard: dict, assessment: dict, proposal: dict,
                  gate_ruling: dict | None = None,
                  build_history: dict | None = None,
                  stage: str = "proposed") -> dict:
    """Assemble the round's human-readable account. Pure/deterministic."""

    sc = (scorecard or {}).get("candidates") or {}
    _a = (assessment or {}).get("assessment") or {}

    verdicts = {e.get("candidate_id"): e for e in (_a.get("per_candidate") or [])
                if e.get("candidate_id")}
    decisions = (proposal or {}).get("decisions") or {}

    unbuilt = {}
    for cid, recs in (build_history or {}).items():
        last = [r for r in recs if not r.get("applied")]
        if last and not any(r.get("applied") for r in recs[-1:]):
            unbuilt[cid] = last[-1]
    gate = gate_ruling or {}
    gate_reason = {r.get("candidate_id"): r for r in (gate.get("rulings") or [])}
    blocked = set(gate.get("blocked") or [])
    pruned = set(gate.get("pruned") or [])

    prev, new = list(previous_plan or []), list(new_plan or [])
    kept, dropped, added = [], [], []

    for cid in prev:
        v = verdicts.get(cid, {})
        entry = {
            "candidate": cid,
            "heuristic": (sc.get(cid) or {}).get("heuristic"),
            "contribution": _contribution(sc.get(cid)),

            "what_the_handler_did": _handler_self_report(build_history, cid),
            "critic_verdict": v.get("verdict"),
            "critic_confidence": v.get("confidence"),
            "critic_note": v.get("note"),

            "critic_evidence_ref": v.get("evidence_ref"),

            "critic_dimensions": v.get("dimensions") or {},
        }
        if v.get("verdict") == "exogenous_only":
            entry["exogenous_only"] = (
                "the simulator cannot represent what this candidate removes, so "
                "every measured column is empty by construction. The log evidence "
                "is the whole account — this is NOT 'measured and found useless'.")
        if v.get("verdict") == "covered":
            entry["attribution_unresolved"] = (
                "the Critic could not separate this candidate's effect from another "
                "candidate touching the same activities; its contribution is still open")
        if cid in new:
            if stage == "measured":

                entry["why_kept"] = (
                    f"in the plan that was measured this round; "
                    f"Critic verdict: {v.get('verdict') or 'not judged'}"
                    + (f" — {v['note']}" if v.get("note") else ""))
            else:
                entry["why_kept"] = decisions.get(cid) or (proposal or {}).get("rationale", "")
                entry["reason_is_candidate_specific"] = cid in decisions
            kept.append(entry)
        else:
            if cid in unbuilt:
                entry["why_dropped"] = (
                    f"Phase-3 could not build it: {unbuilt[cid].get('reason', '')}")
                entry["contribution"] = {
                    "measured": False,
                    "reason": "never applied, so it has no measured effect"}
                entry["order_dependent"] = unbuilt[cid].get("order_dependent")
            elif cid in blocked:
                entry["why_dropped"] = "COMPLIANCE — the Arbiter forbade it, and a " \
                                       "compliance block is never rolled back"
                entry["ruling"] = (gate_reason.get(cid) or {}).get("reason")
            elif cid in pruned:
                entry["why_dropped"] = "pruned by the Arbiter before this plan ran"
                entry["ruling"] = (gate_reason.get(cid) or {}).get("reason")
            else:
                entry["why_dropped"] = decisions.get(cid) or (proposal or {}).get("rationale", "")
                entry["reason_is_candidate_specific"] = cid in decisions
                entry["responds_to"] = (proposal or {}).get("responds_to")
            dropped.append(entry)

    for cid in new:
        if cid in prev:
            continue
        added.append({
            "candidate": cid,
            "heuristic": (sc.get(cid) or {}).get("heuristic"),
            "why_added": decisions.get(cid) or (proposal or {}).get("rationale", ""),
            "reason_is_candidate_specific": cid in decisions,
            "previous_contribution": _contribution(sc.get(cid)),
        })

    un = (scorecard or {}).get("unattributed") or {}
    return {
        "round": round_no,
        "stage": stage,

        **({"plan_spillover": {
            "h_per_case": un.get("h_per_case"),
            "z": un.get("h_per_case_z"),
            "significant": un.get("significant"),
            "charged_to": un.get("caused_by"),
            "activities": [r["activity"] for r in (un.get("activities") or [])][:8],
            "note": un.get("note"),
        }} if un.get("activities") else {}),

        **({"accounting": (scorecard or {}).get("accounting")}
           if (scorecard or {}).get("accounting") else {}),
        "plan_before": prev,
        "plan_after": new,
        "kept": kept,
        "dropped": dropped,
        "added": added,
        "directive": (assessment or {}).get("directive"),

        "directive_followthrough": directive_followthrough(
            (assessment or {}).get("directive"), prev, new),
        **({"contradicts_directive": _contradiction(
            (assessment or {}).get("directive"), prev, new)}
           if _contradiction((assessment or {}).get("directive"), prev, new) else {}),
        "unresolved_attribution": sorted(
            e["candidate"] for e in kept + dropped if e.get("attribution_unresolved")),
        "generic_reasons": sorted(
            e["candidate"] for e in kept + dropped + added
            if e.get("reason_is_candidate_specific") is False),

        "offered_unmeasured": list(
            (proposal or {}).get("unmeasured_offered") or []),
        "offered_unmeasured_unanswered": list(
            (proposal or {}).get("unmeasured_unanswered") or []),

        "dropped_exogenous_only": list(
            (proposal or {}).get("dropped_exogenous_only") or []),

        "per_candidate_missing": list(_a.get("per_candidate_missing") or []),

        "per_candidate_malformed": list(_a.get("per_candidate_malformed") or []),
        "schema_retry": _a.get("schema_retry") or None,

        "clean_probe": _a.get("clean_probe") or None,

        "plan_vs_baseline": _a.get("plan_vs_baseline"),
        "plan_reading": _a.get("plan_reading"),
        "confirmation_counterfactual": _a.get("confirmation_counterfactual"),
        "note": (("This round's plan and what each candidate in it contributed. "
                  "No next plan was chosen (the loop ended here, or has not reached "
                  "the Proposer yet), so there is nothing dropped or added. "
                  if stage == "measured" else "")
                 + "Contribution numbers are measured (scorecard); verdicts are the "
                 "Critic's judgment of them; keep/drop is the Proposer's choice, "
                 "except where the Arbiter overruled it. `generic_reasons` lists "
                 "candidates whose reason fell back to the plan-level rationale — "
                 "for those, 'why this one' is not actually answered."),
    }

def write_round_explanation(out_dir: str | Path, payload: dict) -> Path:
    p = Path(out_dir) / "explanation.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                 encoding="utf-8")
    return p

def render(payload: dict) -> str:
    """The same content as plain text, for a terminal or a paper appendix."""
    sp = payload.get("plan_spillover") or {}
    L = [f"ROUND {payload['round']}"
         + (f"   FINAL PLAN {payload['plan_after']}" if payload.get("stage") == "measured"
            else f"   {payload['plan_before']} -> {payload['plan_after']}")]
    if sp.get("activities"):
        L.append(f"  PLAN SPILLOVER  {sp['h_per_case']:+.3f} h/case"
                 + (f" (z={sp['z']:+.1f})" if sp.get("significant") else " — not significant")
                 + f" on {len(sp['activities'])}+ activities no candidate touches"
                 + (f", charged to {sp['charged_to']}" if sp.get("charged_to")
                    else ", belongs to the plan not to any one candidate"))
    for title, key, why in (("KEPT", "kept", "why_kept"),
                            ("DROPPED", "dropped", "why_dropped"),
                            ("ADDED", "added", "why_added")):
        rows = payload.get(key) or []
        if not rows:
            continue
        L.append(f"\n{title} ({len(rows)})")
        for e in rows:
            c = e.get("contribution") or e.get("previous_contribution") or {}
            L.append(f"  {e['candidate']:9} {str(e.get('heuristic'))[:20]:22} "
                     f"{c.get('summary', '')}")
            if e.get("critic_verdict"):
                L.append(f"      verdict : {e['critic_verdict']} "
                         f"({e.get('critic_confidence')}) — {str(e.get('critic_note'))[:100]}")
                dims = e.get("critic_dimensions") or {}
                if dims:
                    L.append("      dims    : " + "  ".join(
                        f"{k}={v}" for k, v in dims.items()))
                if e.get("critic_evidence_ref"):
                    L.append(f"      rests on: {e['critic_evidence_ref']}")
            if c.get("spillover_note"):
                L.append(f"      spill   : {c['spillover_note']}")
            if c.get("exogenous_saving"):
                L.append(f"      log-only: {c['exogenous_saving']}")
            if c.get("exogenous_conflict"):
                L.append(f"      ⚠ CONFLICT: {c['exogenous_conflict'][:150]}")
            for line in (c.get("attribution_settled_by") or []):
                L.append(f"      settled : {line}")
            if c.get("attribution_caveat"):
                L.append(f"      caveat  : {c['attribution_caveat']}")
            L.append(f"      {why:8}: {str(e.get(why) or e.get('ruling') or '')[:150]}")
    return "\n".join(L)

def _stop_reason(st: dict) -> str:
    """Which of `route_after_conclude`'s exits ended the run."""

    from mabpr.graph import PATIENCE

    if int(st.get("sim_budget", 1)) <= 0:
        return "simulation budget exhausted"
    no_gain = int(st.get("rounds_without_improvement", 0))
    if no_gain >= PATIENCE:
        return f"no significant improvement for {no_gain} consecutive rounds"
    rnd, cap = int(st.get("round", 0)), int(st.get("max_rounds", 0))
    if cap and rnd >= cap:
        return f"round cap ({cap}) reached — a backstop, not a convergence result"
    return (f"the gate left nothing that may legally run, or the loop ended without "
            f"a stated condition (round {rnd}, no_gain {no_gain})")

def _final_overruled(final_state: dict) -> list:
    """Candidates the FINAL review barred after the ROUNDS had let them run."""

    barred = set(((final_state.get("compliance_final") or {}).get("barred")) or [])
    if not barred:
        return []
    ran_in: dict = {}
    for t in (final_state.get("tried_plans") or []):
        for cid in (t.get("ordering") or []):
            if cid in barred:
                ran_in.setdefault(cid, []).append(
                    f"r{t.get('round')}{'c' if t.get('mode') == 'confirming' else ''}")
    out = []
    for cid in sorted(barred):
        where = ran_in.get(cid) or []
        out.append({
            "candidate": cid,
            "rounds_that_let_it_run": where,
            "note": (
                f"The final review barred {cid} after the per-round gate had "
                f"cleared it in {len(where)} measured plan(s) "
                f"({', '.join(where)}). Every measurement containing it is "
                f"ineligible, so the deliverable comes from a pool the search did "
                f"not optimise. Neither ruling can be checked against ground truth."
                if where else
                f"The final review barred {cid}; no measured plan contained it, so "
                f"nothing became ineligible.")})
    return out

def _roster(final_state: dict, best_plan: list, account: dict | None,
            barred: list) -> list:
    """EVERY detected candidate, in or out of the final plan, with the reason."""

    try:
        cands = json.loads(Path(final_state["paths"]["detector_results"]
                                ).read_text(encoding="utf-8"))
    except Exception:
        return []
    acct = {r["candidate"]: r for r in
            (((account or {}).get("kept") or []) + ((account or {}).get("dropped") or []))}
    build = final_state.get("build_history") or {}
    harmful = final_state.get("harmful_history") or {}
    in_plan, barred_set = set(best_plan), set(barred or [])

    blocked_in_round: dict = {}
    for adv in (final_state.get("compliance_advisory") or []):
        for cid in (adv.get("blocked") or []):
            blocked_in_round.setdefault(cid, []).append(adv.get("round"))

    measured_in: dict = {}
    for t in (final_state.get("tried_plans") or []):
        for cid in (t.get("ordering") or []):
            measured_in.setdefault(cid, []).append(t.get("round"))

    out = []
    for c in cands:
        cid = c["candidate_id"]
        row = {"candidate": cid, "heuristic": c.get("heuristic"),
               "target": c.get("target_activity") or c.get("target_activities")}
        a = acct.get(cid) or {}

        tries = build.get(cid) or []
        never_built = bool(tries) and not any(t.get("applied") for t in tries)

        if cid in barred_set:
            row["status"] = "barred"
            row["why"] = ("compliance barred it (judged on prior knowledge alone, "
                          "DECISIONS B — never overturnable by measurement)")
        elif cid in in_plan:
            row["status"] = "in_plan"
            row["why"] = ((a.get("contribution") or {}).get("summary")
                          or "in the plan that measured best")
            if a.get("critic_verdict"):
                row["verdict"] = a["critic_verdict"]
        elif a:
            row["status"] = "dropped"
            row["why"] = (a.get("why_dropped")
                          or (a.get("contribution") or {}).get("summary")
                          or "taken out by the Proposer")
            if a.get("critic_verdict"):
                row["verdict"] = a["critic_verdict"]
        elif never_built:
            row["status"] = "never_built"
            row["why"] = f"Phase-3 refused it every time: {tries[-1].get('reason') or '?'}"
        elif harmful.get(cid):
            h = harmful[cid][-1]
            row["status"] = "dropped"
            row["why"] = (f"measured harmful in round {h.get('round')} "
                          f"({h.get('verdict')}) and not brought back")
        elif measured_in.get(cid):
            rs = sorted({r for r in measured_in[cid] if r is not None})
            row["status"] = "dropped"
            row["why"] = (f"measured in round{'s' if len(rs) > 1 else ''} "
                          f"{', '.join(str(r) for r in rs)}, but not in the plan "
                          f"that won — the winning round's account has no row for "
                          f"it, so what it contributed is only readable in those "
                          f"rounds' own explanation.json")
        elif blocked_in_round.get(cid):
            rs = sorted({r for r in blocked_in_round[cid] if r is not None})
            row["status"] = "compliance_blocked_in_round"
            row["why"] = (f"compliance-blocked in round"
                          f"{'s' if len(rs) > 1 else ''} "
                          f"{', '.join(str(r) for r in rs)}, so it never entered a "
                          f"plan the search measured. A block is irrevocable "
                          f"(DECISIONS B) and is a judgment on prior knowledge "
                          f"alone, which no measurement in this run could overturn "
                          f"— read that round's gate_ruling.json for the reasoning.")
            row["blocked_in_rounds"] = rs
        else:
            row["status"] = "not_selected"
            row["why"] = "never entered a plan the search measured"
        out.append(row)
    order = {"barred": 0, "compliance_blocked_in_round": 1, "in_plan": 2,
             "dropped": 3, "never_built": 4, "not_selected": 5}
    return sorted(out, key=lambda r: (order.get(r["status"], 9), r["candidate"]))

def _compliance_cost(best: dict, excluded: list, barred: set, n_runs: int) -> "dict | None":
    """What the compliance block took away, when it took away something better."""

    if not excluded or best is None:
        return None
    better = [t for t in excluded
              if (t.get("pct_vs_baseline") is not None
                  and t["pct_vs_baseline"] < (best.get("pct_vs_baseline") or 0))]
    if not better:
        return None
    top = min(better, key=lambda t: t["pct_vs_baseline"])
    gap = (best.get("pct_vs_baseline") or 0) - top["pct_vs_baseline"]
    return {
        "barred": sorted(barred),
        "excluded_measurements": len(excluded),
        "best_excluded_pct": round(top["pct_vs_baseline"], 2),
        "best_excluded_round": f"round{top.get('round')}"
                               + ("c" if top.get("mode") == "confirming" else ""),
        "best_excluded_plan": list(top.get("ordering") or []),
        "gap_pp": round(gap, 2),
        "note": (f"the answer above is {gap:.1f} percentage points WORSE than a "
                 f"plan this run actually measured, which is barred because it "
                 f"contains {sorted(set(top.get('ordering') or []) & barred)}. "
                 f"The block is the reason, not the search. Compliance is judged "
                 f"on prior knowledge alone and cannot be overturned by "
                 f"measurement (DECISIONS B/C), so this gap is the price of that "
                 f"judgement and belongs beside the headline, not below it."),
    }

def measurement_ids(pool: list) -> list:
    """One id per measurement, positionally aligned with `pool`, GUARANTEED unique."""

    base = ["round%s%s" % (t.get("round"), "c" if t.get("mode") == "confirming" else "")
            for t in (pool or [])]
    seen, out = {}, []
    for b in base:
        if base.count(b) == 1:
            out.append(b)
            continue
        seen[b] = seen.get(b, 0) + 1
        out.append("%s#%d" % (b, seen[b]))
    return out

def directive_followthrough(directive: "dict | None", prev: list,
                            new: list) -> dict:
    """How far the plan moved relative to what the directive NAMED."""

    d = directive or {}
    before, after = list(prev or []), list(new or [])
    sb, sa = set(before), set(after)
    named = [str(t) for t in (d.get("targets") or [])]
    left = sorted(sb - sa)
    came = sorted(sa - sb)
    return {
        "intent": str(d.get("intent") or ""),
        "named": named,
        "left_named": [c for c in left if c in named],
        "left_unnamed": [c for c in left if c not in named],
        "added_named": [c for c in came if c in named],
        "added_unnamed": [c for c in came if c not in named],
        "size_before": len(before),
        "size_after": len(after),
        "net_change": len(after) - len(before),
    }

def _with_attribution_share(rows: list) -> list:
    """Attach `share_of_attributed_pct` to each reported candidate row."""

    from mabpr.agents.selector import _attribution_scale

    flat = []
    for r in (rows or []):
        c = (r or {}).get("contribution") or {}
        flat.append({"candidate_id": (r or {}).get("candidate"),
                     "throughput_h_per_case": c.get("throughput_h_per_case")})
    scale = _attribution_scale(flat)
    by_id = {f.get("candidate_id"): f for f in flat}
    out = []
    for r in (rows or []):
        r = dict(r or {})
        c = dict(r.get("contribution") or {})
        share = (by_id.get(r.get("candidate")) or {}).get("share_of_attributed_pct")
        if share is not None:
            c["share_of_attributed_pct"] = share
            if scale.get("attributed_h_per_case_magnitude") is not None:
                c["attributed_h_per_case_magnitude"] = scale[
                    "attributed_h_per_case_magnitude"]
            r["contribution"] = c
        out.append(r)
    return out

def improvement_rule(kpis: dict) -> dict:
    """Does this measurement satisfy the project's own improvement rule?"""

    from mabpr.graph import CONVERGENCE_DIRECTION

    better, worse, no_direction = [], [], []
    for measure, direction in CONVERGENCE_DIRECTION.items():
        entry = kpis.get(measure) or {}
        pct = entry.get("pct")
        if pct is None:
            continue
        if direction is None:
            no_direction.append(measure)
            continue
        if not entry.get("significant"):
            continue
        improved = pct < 0 if direction == "lower" else pct > 0
        (better if improved else worse).append(measure)

    if better and not worse:
        verdict = "better"
    elif worse and not better:
        verdict = "worse"
    elif better and worse:
        verdict = "mixed"
    else:
        verdict = "unchanged"
    return {"verdict": verdict, "better_on": better, "worse_on": worse,
            "no_direction": no_direction,
            "rule": ("significantly better on at least one directional measure "
                     "and significantly worse on none")}

def _gain_source(kpis: dict) -> dict:
    """Is the throughput gain LESS WORK, or the SAME work moving faster?"""

    def g(k):
        e = kpis.get(k) or {}
        return e.get("pct"), bool(e.get("significant"))

    def hours(k):
        """The change in hours per case, for the note. Percentages of two"""

        d = (kpis.get(k) or {}).get("delta")
        return f"{d:+.3f} h/case" if isinstance(d, (int, float)) else "unsized"

    svc, svc_sig = g("duration_h_per_case_mean")
    wait, wait_sig = g("waiting_h_per_case_mean")
    thr, thr_sig = g("throughput_h_median")
    mean, mean_sig = g("throughput_h_mean")
    if svc is None or wait is None or thr is None:
        return {}

    if thr > 0 and thr_sig:
        out = {"throughput_pct": thr, "service_pct": svc, "waiting_pct": wait,
               "mean_pct": mean, "verdict": "worse_than_baseline",
               "throughput_significant": thr_sig}
        parts = []
        if mean is not None and mean < 0 and mean_sig:
            parts.append(f"mean case span FELL {abs(mean):.1f}% at the same time, "
                         f"so the two case-span measures disagree in sign and the "
                         f"plan helps some cases while hurting others")
        if wait < 0 and wait_sig:
            parts.append(f"waiting fell {abs(wait):.1f}%")
        if svc > 0 and svc_sig:
            parts.append(f"service rose {svc:+.1f}%")
        out["note"] = (
            f"⚠️ THE PLAN IS WORSE THAN THE BASELINE ON THE CONVERGENCE MEASURE: "
            f"median case span {thr:+.1f}%"
            + (f" (significant)" if thr_sig else " (not significant)")
            + (". " + "; ".join(parts) + "." if parts else ". ")
            + " Improvement inside the loop is measured against the best plan "
              "ESTABLISHED SO FAR, never against the baseline, so a run that opens "
              "badly can converge without ever beating doing nothing. Read this "
              "before reading any per-candidate contribution.")
        return out
    out = {"throughput_pct": thr, "service_pct": svc, "waiting_pct": wait,
           "service_significant": svc_sig, "waiting_significant": wait_sig}
    less_work = svc_sig and svc < 0

    more_work = svc_sig and svc > 0
    not_faster = (wait >= 0) or not wait_sig
    if less_work and not_faster:
        moved = "rises" if wait >= 0 else "does not significantly fall"
        sig_note = "" if wait_sig else ", not significant"
        out["verdict"] = "volume"
        out["note"] = (
            f"⚠️ The throughput gain of {thr:.1f}% is a WORK-VOLUME reduction, "
            f"not a speed-up: service time per case falls {abs(svc):.1f}% "
            f"(significant) while waiting {moved} ({wait:+.1f}%"
            f"{sig_note}). The redesign makes the "
            f"process DO LESS. That may be exactly the intent — a removed "
            f"rework loop, a deleted activity, a case type routed away — but it "
            f"is a different claim from 'the same work happens faster', and "
            f"whether it is acceptable is a domain judgment this system does "
            f"not make.")
    elif svc_sig and svc < 0 and wait_sig and wait < 0:
        out["verdict"] = "both"
        out["note"] = (f"Service time −{abs(svc):.1f}% AND waiting −"
                       f"{abs(wait):.1f}%, both significant: the process does "
                       f"less work AND queues less.")
    elif wait_sig and wait < 0 and more_work:

        out["verdict"] = "speed_at_a_cost"
        out["note"] = (
            f"⚠️ Waiting falls {abs(wait):.1f}% ({hours('waiting_h_per_case_mean')}"
            f", significant), but the process does MORE work: service time per "
            f"case RISES {svc:+.1f}% ({hours('duration_h_per_case_mean')}, "
            f"significant). The throughput gain of {thr:.1f}% is not a saving "
            f"of effort — a redesign that reroutes cases onto a busier path "
            f"reads like this. Both changes are real; which one weighs more is "
            f"a domain judgment this system does not make, so the two numbers "
            f"are given in the same unit and left unranked.")
    elif wait_sig and wait < 0:
        out["verdict"] = "speed"
        out["note"] = (f"The gain is a QUEUEING improvement: waiting "
                       f"−{abs(wait):.1f}% (significant) with service time "
                       f"{svc:+.1f}% (not significant). The same work is "
                       f"moving faster.")

    elif more_work:
        out["verdict"] = "more_work"
        out["note"] = (
            f"⚠️ Throughput improves {thr:.1f}% while service time per case "
            f"RISES {svc:+.1f}% ({hours('duration_h_per_case_mean')}, "
            f"significant) and waiting does not significantly fall "
            f"({wait:+.1f}%). The process does MORE work and no queueing "
            f"improvement explains the gain; read the per-candidate account "
            f"before treating this as an improvement.")
    else:
        out["verdict"] = "unclear"
        out["note"] = ("Neither service time nor waiting moved significantly; "
                       "the throughput change is not explained by either.")
    return out

_SHORT = {"throughput_h_median": "median", "throughput_h_mean": "mean",
          "waiting_h_per_case_mean": "wait", "duration_h_per_case_mean": "work",
          "utilization_mean": "util"}

def _round_measures(kpis: dict) -> dict:
    """{measure: {pct, z, significant}} — every reported KPI, for one round."""

    out = {}
    for k, e in (kpis or {}).items():
        if not isinstance(e, dict):
            continue
        out[k] = {"pct": e.get("pct"), "z": e.get("z"),
                  "significant": bool(e.get("significant"))}
    return out

def write_final_report(final_state: dict, *, sig_better, select_fn=None) -> Path:
    """RESULT — the one file to open after a run."""

    tried = list(final_state.get("tried_plans") or [])
    paths = final_state.get("paths") or {}
    n_runs = int((final_state.get("sim_config") or {}).get("n_runs", 20))
    run_dir = Path(paths["out_dir"]) / "loop" / paths.get("run_id", "run")

    barred = set(final_state.get("blocked_candidates") or [])
    legal = [t for t in tried if not (set(t.get("ordering") or []) & barred)]
    excluded = [t for t in tried if t not in legal]

    best = None
    for t in legal:
        if best is None or sig_better(t, best, n_runs):
            best = t

    def _defensible_against(t: dict, ref: dict):
        """-> the measure name this one is decisively better on, or None."""

        if ref is None or t is ref:
            return "same_measurement"
        from mabpr.graph import CONVERGENCE_DIRECTION

        tk, rk = (t.get("kpis") or {}), (ref.get("kpis") or {})
        for key in tk.keys() | rk.keys():

            if CONVERGENCE_DIRECTION.get(key) != "lower":
                continue
            a, b = tk.get(key) or {}, rk.get(key) or {}
            za, zb = a.get("z"), b.get("z")
            if za is None or zb is None:
                continue

            ra, rb_ = a.get("redes"), b.get("redes")
            ba, bb = a.get("base"), b.get("base")
            if None in (ra, rb_, ba, bb) or not za or not zb:
                continue
            se_a = abs((ra - ba) / za)
            se_b = abs((rb_ - bb) / zb)
            se = math.sqrt(se_a ** 2 + se_b ** 2)
            if se == 0:
                continue
            if ((ra - rb_) / se) < -3.0:
                return key
        return None

    tied, _admitted = [], {}
    if best:

        _admitted[id(best)] = "median_best_measurement"
        for t in legal:
            if t is best:
                continue
            on_median = not sig_better(best, t, n_runs)
            elsewhere = _defensible_against(t, best)
            if not (on_median or elsewhere):
                continue
            tied.append(t)
            _admitted[id(t)] = ("not_significantly_worse_on_median" if on_median
                                else "decisively_better_on:" + str(elsewhere))

    pool = ([best] + tied) if best is not None else []
    selection = None

    from mabpr.graph import plan_weight
    if select_fn is not None and pool:
        try:
            selection = select_fn(pool)
        except Exception as e:
            selection = {"fallback": f"{type(e).__name__}: {e}"}
        picked = (selection or {}).get("chosen")

        ids = measurement_ids(pool)
        assert len(set(ids)) == len(ids), "measurement ids are not unique: %s" % ids
        by_id = dict(zip(ids, pool))
        if picked in by_id:
            best = by_id[picked]
        elif picked:

            selection = {**(selection or {}),
                         "fallback": ("the chosen id %r is not among the ids code "
                                      "built for the pool (%s); the report kept the "
                                      "median-best measurement" % (picked, ids))}
    if best is not None and pool:

        if selection is None or not (selection or {}).get("chosen"):

            best = min(pool, key=lambda t: (plan_weight(t),
                                            t.get("median_h") if t.get("median_h")
                                            is not None else float("inf"),
                                            t.get("round") or 0))
        tied = [t for t in pool if t is not best]

    def _dir_of(t: dict) -> Path:

        if t.get("artifacts"):
            return Path(t["artifacts"])
        stem = "confirm_round" if t.get("mode") == "confirming" else "round"
        return run_dir / f"{stem}{t.get('round')}"

    account, account_from = None, None

    def _account_of(round_no):
        f = run_dir / ("round%s" % round_no) / "explanation.json"
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _covers(acc, plan):

        got = set(acc.get("plan_before") or [])
        return bool(got) and not [c for c in (plan or []) if c not in got]

    if best is not None:
        _want = list(best.get("ordering") or [])
        _here = int(best.get("round") or 0)
        acc = _account_of(_here)
        if acc is not None and (_covers(acc, _want) or _here <= 1):
            account, account_from = acc, "round%d" % _here
        else:
            for _r in range(_here, 0, -1):
                cand = _account_of(_r)
                if cand is not None and _covers(cand, _want):
                    account, account_from = cand, "round%d" % _r
                    break
            if account is None and acc is not None:

                account, account_from = acc, "round%d" % _here

    report = {
        "dataset": (final_state.get("sim_config") or {}).get("dataset"),
        "run_id": paths.get("run_id"),

        "gate_enforced": bool(final_state.get("gate_enforced", True)),
        "baseline_h": (final_state.get("base_agg") or {}).get("throughput_h_median_mean"),
        "rounds_measured": len(tried),

        "stopped_because": _stop_reason(final_state),
        "best": None if best is None else {
            "plan": list(best.get("ordering") or []),

            "built": list(best.get("built") or []) or None,
            "changed_nothing": list(best.get("changed_nothing") or []) or None,
            "round": best.get("round"), "mode": best.get("mode"),
            "median_h": best.get("median_h"),
            "pct_vs_baseline": best.get("pct_vs_baseline"),
            "z_vs_baseline": best.get("vs_baseline_z"),

            "all_measures": best.get("kpis") or {},
            "gain_source": _gain_source(best.get("kpis") or {}),

            "improvement_rule": improvement_rule(best.get("kpis") or {}),

            "case_outcome_shift": best.get("case_endings"),

            "compliance_cost": _compliance_cost(best, excluded, barred, n_runs),
            "artifacts": str(_dir_of(best)),
        },

        "no_compliant_plan": (
            None if (best is not None or not tried) else
            {"measurements": len(tried), "barred": sorted(barred),
             "note": ("every measured plan contains a candidate the final "
                      "compliance review barred, so this run has no deliverable. "
                      "The measurements are real and are listed under "
                      "`excluded_as_non_compliant` — this is the cost of the "
                      "block, not an absence of results.")}),

        "trajectory": [
            {"round": t.get("round"), "mode": t.get("mode"),

             "kind": t.get("kind"),
             "probe_reason": t.get("probe_reason"),

             "artifacts": t.get("artifacts"),
             "plan": list(t.get("ordering") or []),

             "built": (None if t.get("built") is None
                       else list(t.get("built") or [])),
             "changed_nothing": (None if t.get("changed_nothing") is None
                                 else list(t.get("changed_nothing") or [])),
             "median_h": t.get("median_h"),

             "median_std": t.get("median_std"),
             "pct_vs_baseline": t.get("pct_vs_baseline"),
             "z_vs_baseline": t.get("vs_baseline_z"),
             "improvement_rule": improvement_rule(t.get("kpis") or {}).get("verdict"),

             "measures": _round_measures(t.get("kpis") or {}),
             "case_outcome_shift": ((t.get("case_endings") or {}).get("headline")
                                    if (t.get("case_endings") or {}).get("significant")
                                    else None),
             "is_best": t is best,
             "barred_candidates": sorted(set(t.get("ordering") or []) & barred)}
            for t in tried],
        "measures_significant_in_every_round": sorted({
            k for k, e in (tried[0].get("kpis") or {}).items()
            if all((t.get("kpis") or {}).get(k, {}).get("significant") for t in tried
                   if t.get("kpis"))}) if tried else [],

        "statistically_tied_with_best": [
            {"round": t.get("round"), "mode": t.get("mode"),
             "plan": list(t.get("ordering") or []),
             "pct_vs_baseline": t.get("pct_vs_baseline"),

             "admitted_because": _admitted.get(id(t), "unknown"),

             "measures": _round_measures(t.get("kpis") or {})}
            for t in tied],

        "excluded_as_non_compliant": [
            {"round": t.get("round"), "mode": t.get("mode"),
             "plan": list(t.get("ordering") or []),
             "pct_vs_baseline": t.get("pct_vs_baseline"),
             "barred_candidates": sorted(set(t.get("ordering") or []) & barred)}
            for t in excluded],

        "per_candidate": _with_attribution_share(
            ((account or {}).get("kept") or [])
            + ((account or {}).get("dropped") or [])),
        "per_candidate_from": account_from,

        "selection": selection,

        "final_plan_roster": _roster(final_state,
                                     list((best or {}).get("ordering") or []),
                                     account, sorted(barred)),
        "per_candidate_covers": (account or {}).get("plan_before") or [],

        "per_candidate_unaccounted": ([] if best is None else sorted(
            set(best.get("ordering") or [])
            - set((account or {}).get("plan_before") or []))),
        "per_candidate_caveat": (
            f"The winner is a CONFIRMATION (a subset probe inside round "
            f"{best.get('round')}), which has no per-candidate account of its own. "
            f"The rows below describe {account_from}'s full plan "
            f"{list((account or {}).get('plan_after') or [])}, not the confirmed "
            f"subset {list(best.get('ordering') or [])}."
            if best is not None and best.get("mode") == "confirming" and account
            else None),
        "final_round_plan": list(final_state.get("current_plan") or []),

        "compliance_final": final_state.get("compliance_final"),
        "compliance_final_overruled": _final_overruled(final_state),
        "compliance_blocked": (None if final_state.get("blocked_candidates") is None
                               else list(final_state["blocked_candidates"])),
        "note": ("`best` is the best MEASUREMENT in the run, which is usually not the "
                 "last round. ⚠️ `statistically_tied_with_best` is the SELECTOR'S "
                 "CANDIDATE POOL, not a set of indistinguishable results: a row is "
                 "admitted either because it is not significantly worse than `best` "
                 "on the median, or because it is decisively better on ONE "
                 "directional measure while possibly far worse on the others. Read "
                 "each row's `admitted_because` before describing it, and never "
                 "call the list a tie. `per_candidate` is the winning plan's own "
                 "account: what each surviving candidate contributed on waiting, "
                 "duration, throughput and resources."),
    }
    p = run_dir / "result.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                 encoding="utf-8")
    return p

def render_final(report: dict) -> str:
    b = report.get("best") or {}
    L = [f"RESULT  {report.get('dataset')}  run {report.get('run_id')}",
         f"  baseline        {report.get('baseline_h')} h",
         f"  gate            {'ENFORCED' if report.get('gate_enforced') else 'NOT ENFORCED (ablation arm)'}",
         f"  stopped         {report.get('stopped_because')}",
         f"  BEST PLAN       {b.get('plan')}",
         f"                  {b.get('median_h')} h  "
         f"({b.get('pct_vs_baseline')}%, z={b.get('z_vs_baseline')})  "
         f"from round {b.get('round')}{' (confirmation)' if b.get('mode') == 'confirming' else ''}"]
    cc = (report.get("best") or {}).get("compliance_cost")
    if cc:
        import textwrap as _tw
        L.append("")
        L.append(f"  ⚠ COMPLIANCE COST  barred {cc['barred']} — {cc['excluded_measurements']} "
                 f"measurement(s) excluded")
        L.append(f"                     a measured plan at {cc['best_excluded_pct']}% "
                 f"({cc['best_excluded_round']}) is {cc['gap_pp']} pp better and may not run")
        L += ["                     " + ln
              for ln in _tw.wrap(cc["note"], 74)]
    ncp = report.get("no_compliant_plan") or {}
    if ncp:
        import textwrap
        L.append(f"  ⚠ NO DELIVERABLE — {ncp.get('measurements')} plan(s) were "
                 f"measured and every one of them contains {ncp.get('barred')}")
        for line in textwrap.wrap(str(ncp.get("note") or ""), 74):
            L.append(f"      {line}")
    meas = (report.get("best") or {}).get("all_measures") or {}
    if meas:
        L.append("  ALL MEASURES on the winning plan")
        for k, e in meas.items():
            L.append(f"      {e['label']:>16}  {e['pct']:>+7.1f}%  z={e['z']:>+7.1f}"
                     + ("  SIGNIFICANT" if e["significant"] else "  not significant")
                     + f"   ({e['measures']})")
        ev = report.get("measures_significant_in_every_round") or []
        if ev:
            L.append(f"      significant in EVERY round: "
                     + ", ".join(meas[k]["label"] for k in ev if k in meas))

    gs = (report.get("best") or {}).get("gain_source") or {}
    if gs.get("note"):
        L.append(f"  GAIN SOURCE     [{gs.get('verdict', '?').upper()}]")
        import textwrap
        for line in textwrap.wrap(gs["note"], 74):
            L.append(f"      {line}")
    ce = (report.get("best") or {}).get("case_outcome_shift") or {}
    if ce.get("significant"):
        L.append(f"  CASE OUTCOMES   {ce.get('headline', '')}")
        L.append(f"      {len(ce['significant'])} ending(s) moved significantly. "
                 f"No reported measure can see this: all five are per-case")
        L.append("      averages. Which ending is desirable is a domain question.")
        for x in ce["significant"][:6]:
            L.append(f"      {x['activity']:<32} {x['base']:>7.0f} -> "
                     f"{x['redes']:>7.0f} cases  ({x['delta']:+.0f}, z={x['z']:+.1f})")
    traj = report.get("trajectory") or []
    if traj:
        L.append(f"\n  EVERY ROUND ({len(traj)} measurement(s), in order)")
        for t in traj:
            mark = "  <- BEST" if t.get("is_best") else ""
            pct = t.get("pct_vs_baseline")
            z = t.get("z_vs_baseline")
            rid = ('r%s' % t.get('round')) + ('c' if t.get('mode') == 'confirming' else '')
            L.append(f"    {rid:>5}  "
                     f"{(f'{pct:+6.1f}%' if pct is not None else '     ?')}  "
                     f"z={(f'{z:+7.1f}' if z is not None else '      ?')}  "
                     f"{len(t.get('plan') or []):>2} cand{mark}")

            ms = t.get("measures") or {}
            if ms:
                L.append("           " + "  ".join(
                    f"{_SHORT.get(k, k)} {v['pct']:+.1f}%{'*' if v.get('significant') else ''}"
                    for k, v in ms.items() if v.get("pct") is not None))
            if t.get("case_outcome_shift"):
                L.append(f"           outcomes: {t['case_outcome_shift']}")
            if t.get("barred_candidates"):
                L.append(f"           barred: {t['barred_candidates']} — not eligible")
    tied = report.get("statistically_tied_with_best") or []
    if tied:
        L.append(f"  ⚠ {len(tied)} other measurement(s) were in the Selector's pool "
                 f"(NOT a tie set — read why each is there):")
        for t in tied:
            why = str(t.get("admitted_because") or "?")
            L.append(f"      round {t['round']:<3} {t['pct_vs_baseline']:+8.1f}%  "
                     f"{why:<42} {t['plan']}")

    sel = report.get("selection") or {}
    if sel.get("chosen") or sel.get("fallback"):
        import textwrap
        L.append(f"  CHOSEN BY       the Selector, weighing all measures"
                 + (f"  [gain reading: {sel['gain_reading']}]"
                    if sel.get("gain_reading") else ""))
        for line in textwrap.wrap(str(sel.get("why") or ""), 74):
            L.append(f"      {line}")
        riders = [p for p in (sel.get("passengers") or [])
                  if p.get("kind") not in (None, "", "none")]
        for p in riders:
            L.append(f"      ⚠ passenger {p.get('candidate')}: {p.get('kind')}"
                     + (f" — {p['note']}" if p.get("note") else ""))
        if not riders and sel.get("chosen"):
            L.append("      every candidate in it was judged to earn its place")
        if sel.get("runner_up"):
            for line in textwrap.wrap(f"runner-up: {sel['runner_up']}", 74):
                L.append(f"      {line}")
        if sel.get("caveat"):
            for line in textwrap.wrap(f"caveat: {sel['caveat']}", 74):
                L.append(f"      {line}")
        if sel.get("fallback"):
            L.append(f"      ! {sel['fallback']}")
    exc = report.get("excluded_as_non_compliant") or []
    if exc:
        L.append(f"  ⚠ {len(exc)} measurement(s) EXCLUDED — their plan contains a "
                 f"candidate compliance later barred, so they describe a redesign "
                 f"that may not run:")
        for t in exc:
            L.append(f"      round {t['round']:<3} {t['pct_vs_baseline']:+6.1f}%  "
                     f"barred: {t['barred_candidates']}  {t['plan']}")
    rows = report.get("per_candidate") or []
    if report.get("per_candidate_unaccounted"):
        L.append("  %s %d of the delivered plan's candidates have NO account row: "
                 "%s - the account is borrowed from %s, whose plan did not contain "
                 "them." % (chr(0x26a0) * 2, len(report["per_candidate_unaccounted"]),
                            ", ".join(report["per_candidate_unaccounted"]),
                            report.get("per_candidate_from")))
    if report.get("per_candidate_caveat"):
        L.append(f"  ⚠ {report['per_candidate_caveat']}")
    L.append(f"\n  WHAT EACH SURVIVING CANDIDATE CONTRIBUTED ({len(rows)})"
             + (f"  [from {report['per_candidate_from']}]"
                if report.get("per_candidate_from") else ""))
    if not rows:
        L.append("      (no per-candidate account — the winning round wrote no explanation)")
    for e in rows:
        c = e.get("contribution") or {}
        L.append(f"    {e['candidate']:9} {str(e.get('heuristic'))[:20]:22} {c.get('summary', '')}")
        if e.get("critic_verdict"):
            L.append(f"        verdict: {e['critic_verdict']} ({e.get('critic_confidence')})"
                     + ("   [dropped after this round]" if e.get("why_dropped") else ""))
        if c.get("spillover_note"):
            L.append(f"        spill  : {c['spillover_note']}")
        for line in (c.get("attribution_settled_by") or []):
            L.append(f"        settled: {line}")
        if c.get("attribution_caveat"):
            L.append(f"        caveat : {c['attribution_caveat']}")

    for ov in (report.get("compliance_final_overruled") or []):
        if ov.get("rounds_that_let_it_run"):
            L.append("")
            L.append("  !! THE FINAL COMPLIANCE REVIEW OVERRULED THE ROUNDS")
            L.append(f"     {ov['candidate']} ran in {ov['rounds_that_let_it_run']} "
                     f"and was then barred, so every measurement containing it is "
                     f"ineligible. The answer comes from a pool the search did not "
                     f"optimise; neither ruling has ground truth.")
    roster = report.get("final_plan_roster") or []
    if roster:
        icon = {"in_plan": "KEEP", "dropped": "DROP", "barred": "BARRED",
                "compliance_blocked_in_round": "BLOCKED",
                "never_built": "UNBUILDABLE", "not_selected": "NOT TRIED"}
        L.append(f"\n  EVERY CANDIDATE, IN OR OUT ({len(roster)})")
        for r in roster:
            L.append(f"    [{icon.get(r['status'], r['status']):11}] {r['candidate']:9} "
                     f"{str(r.get('heuristic'))[:20]:22}"
                     + (f" verdict={r['verdict']}" if r.get("verdict") else ""))
            import textwrap
            for line in textwrap.wrap(str(r.get("why") or ""), 84):
                L.append(f"                  {line}")
    cf = report.get("compliance_final") or {}
    if report.get("compliance_blocked"):
        L.append(f"\n  compliance-blocked (never allowed to run): {report['compliance_blocked']}")
        if cf:
            L.append(f"      final ruling asked {cf.get('asked')}; the answer was "
                     f"chosen from {cf.get('eligible_measurements')} of "
                     f"{cf.get('measurements_total')} measured plans")
    elif report.get("compliance_blocked") == [] and cf:
        during = sorted({c for a in (cf.get("per_round_blocks_during_loop") or [])
                         for c in (a.get("blocked") or [])})
        L.append(f"\n  compliance: nothing barred. Reviewed "
                 f"{len(cf.get('candidates_reviewed') or [])} candidate(s) at the end; "
                 f"every measured plan stayed eligible."
                 + (f"  (during the loop the rounds themselves blocked {during})"
                    if during else ""))
    elif report.get("compliance_blocked") is None:
        L.append("\n  compliance-blocked: UNKNOWN — this report was rebuilt from "
                 "the rounds on disk, and the gate's ruling is not persisted there")
    else:

        L.append("\n  compliance: the gate barred nothing in any round on disk. "
                 "No end-of-run review is recorded (this report was rebuilt), so "
                 "the deliverable's own ruling was not re-asked.")
    return "\n".join(L)
