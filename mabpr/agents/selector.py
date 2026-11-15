"""SELECTOR — which measured plan is the ANSWER, judged on all measures at once."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mabpr import config
from mabpr.explain import _gain_source, improvement_rule, measurement_ids
from mabpr.llm.client import respond_json

_OUT_MAX = 4000

SELECT_SYSTEM = """\
You are choosing which MEASURED redesign is the answer of a completed run. Every
plan below was actually simulated; nothing here is a prediction, and you cannot
commission another measurement. Your job is to pick one, and to say plainly what
is wrong with the one you picked.

TWO THINGS MATTER, AND THEY ARE NOT THE SAME:

1. HOW GOOD THE PLAN IS, across ALL the measures — not one of them. Each measure
   carries its own z (computed by code; you never judge significance yourself).
   |z| > 3 is significant; anything else is noise, whatever the percentage says.
   Every option carries `improvement_rule_computed`: whether it satisfies the
   rule the LOOP was stopped on — significantly better on at least one
   directional measure and significantly worse on NONE. Code computed it from
   these same measures with that rule's own directions.
   ⚠️ IT IS A FACT AND IT DOES NOT CHOOSE FOR YOU. An option it calls
   `mixed` can still be the right answer: waiting that falls by hours against
   service that rises by minutes is a real trade. If you take one, say why in
   HOURS, not in percentages. And `utilization_mean` has no direction, so it can
   neither create nor block that verdict.
   A plan whose case time fell a lot while WAITING got significantly worse is not
   simply "better" — it did less work rather than moving work faster, and that is
   a different claim. Say which it is.

2. WHETHER EVERY CANDIDATE IN IT EARNS ITS PLACE. A candidate is a PASSENGER when:
     · `structural_noop`  — the handler edited nothing at all. ⚠️ This must come
                            from POSITIVE evidence: the candidate's row says so,
                            or `what_the_handler_did` shows an edit with no
                            change counters. An EMPTY or `unrecorded`
                            `what_the_handler_did` is a missing record, not a
                            no-op, and a candidate the Critic judged `helped`
                            with a significant contribution is never one. Calling
                            a working candidate a passenger argues for deleting
                            something that works — the costlier mistake of the
                            two;
     · `no_effect`        — applied, but nothing moved on any of its four columns;
     · `unattributable`   — every measure it touches is shared with another
                            applied candidate, so nothing is attributable to it;
     · `negative`         — its own attributable contribution is a worsening.
                            ⚠️ READ `share_of_plan_change_pct` BEFORE USING THIS
                            LABEL. A z answers "is this activity's change
                            distinguishable from noise", never "does this
                            candidate matter to the plan". A candidate has been
                            labelled `negative` off a highly significant z whose
                            share of the plan's total change was a few HUNDREDTHS
                            of a percent; a single-variable counterfactual moved
                            every one of the five measures by under 2% at
                            |z| < 1.2. Dropping such a candidate changes nothing.
                            Its handler had lowered a 0.119 branch to 0.1065.
                            Significance is not magnitude; say which one you
                            mean, and use `share_of_attributed_pct` for the
                            second.
   ⚠️ A candidate may legitimately earn its place on ONE axis only — better
   waiting with flat duration is a real contribution. Do not require breadth.
   ⚠️ `exogenous_only` — the saving is real in the recorded LOG but outside what
   this simulator can represent, so its simulated columns are flat — is NOT
   automatically a passenger: the log measured
   something the simulator structurally cannot represent. Judge it on that
   evidence and say you are doing so.

BOTH ARE PART OF THE ANSWER, NOT ONE AND THEN A TIE-BREAK.

What you return is a redesign a real organisation is being asked to carry out. A
candidate that contributes nothing is work asked of people for nothing, so it
does not become acceptable because the plan's headline number is good. Treat
"every surviving candidate earns its place" as a property the answer must have,
and a passenger that stays as something you have to JUSTIFY in `caveat` — by
name, with the reason — not merely list.

  · Where two plans differ mainly by passengers and are not significantly apart
    on the measures, take the CLEANER one even if its headline is slightly worse.
    "Slightly" is |z| <= 3 on every measure; code gives you the z.
  · Where the cleaner plan IS significantly worse, you may take the stronger one
    — say so, name the passengers you are accepting, and say what each is being
    kept for.
  · Never average the two into a score. A weighted sum across the Devil's
    Quadrangle axes is the scalar fusion this project forbids (M1).

The pool you are given is closed by code and now spans EVERY measure, not the
convergence KPI alone: a plan that lost the median but is decisively better on
waiting, service or utilisation is in the list on purpose. Read all five columns
of every option before choosing — the option with the best first column is not
the answer by default.
"""

_SCHEMA = """\
Return JSON with EXACTLY these keys:
{
  "chosen": "<the id string of the measurement you picked, exactly as given>",
  "why": "one or two sentences: why this plan across the measures",
  "rule_reading": "state `improvement_rule_computed.verdict` for the plan you chose, in those words (better|mixed|worse|unchanged), and if it is not `better` say in ONE sentence why the trade is worth taking anyway, in hours. Code computed it from the same measures with the stopping rule's own directions; it does NOT decide for you. Use ONLY the numbers in this payload",
  "gain_reading": "start from `gain_source_computed.verdict` — note that `worse_than_baseline` means the plan LOSES to the unchanged process on the convergence measure; say so plainly and do not describe a positive percentage as an improvement. — code derived it from these same measures. Explain it in this plan's numbers, or say plainly that you disagree and why. speed|volume|both|speed_at_a_cost|more_work|unclear — is the same work moving faster, is less work being done, or is MORE work being done? cite the measures in hours as given, not only percentages: waiting and service have very different bases, so a small percentage of one can outweigh a large percentage of the other. Use ONLY the numbers in this payload",
  "passengers": [{"candidate": "<id>", "kind": "structural_noop|no_effect|unattributable|negative|none", "note": ""}],
  "runner_up": "<id or empty> — what you would have taken instead and what it cost",
  "caveat": "what a reader must not over-read in this result"
}
Output ONLY the JSON object."""

@dataclass
class Selection:
    chosen: str = ""
    why: str = ""
    gain_reading: str = ""

    rule_reading: str = ""
    passengers: list = field(default_factory=list)
    runner_up: str = ""
    caveat: str = ""
    fallback: str = ""

    model_text_about: dict = field(default_factory=dict)

def _measure_lines(kpis: dict) -> list:
    out = []
    for e in (kpis or {}).values():
        z = e.get("z")
        out.append({
            "measure": e.get("label"), "what": e.get("measures"),
            "pct": e.get("pct"), "z": z,
            "significant": bool(e.get("significant")),
        })
    return out

def _attribution_scale(per: list) -> dict:
    """Each candidate's attributed hours as a share of what the account attributes."""

    vals = [r.get("throughput_h_per_case") for r in per]
    mag = sum(abs(v) for v in vals if isinstance(v, (int, float)))
    if not mag:
        return {}
    net = sum(v for v in vals if isinstance(v, (int, float)))
    for r in per:
        v = r.get("throughput_h_per_case")
        if isinstance(v, (int, float)):
            r["share_of_attributed_pct"] = round(100.0 * v / mag, 2)
    return {
        "attributed_h_per_case_net": round(net, 4),
        "attributed_h_per_case_magnitude": round(mag, 4),
        "attribution_scale_note": (
            "`share_of_attributed_pct` is each candidate's own attributed hours "
            "as a signed percentage of `attributed_h_per_case_magnitude`, the "
            "total size of the effects this account can see. It supplies the "
            "SCALE a z cannot: significance says an effect is distinguishable "
            "from noise, never that it is large. ⚠️ These are ACTIVITY HOURS and "
            "are NOT comparable with the case-span measures above — the two "
            "quantities differ by all the time in which no activity is running."),
    }

def build_options(candidates_for_choice: list, accounts: dict) -> list:
    """The closed set of measured plans, each with every fact code can supply."""

    opts = []

    oids = measurement_ids(candidates_for_choice)
    for oid, t in zip(oids, candidates_for_choice):
        rnd, mode = t.get("round"), t.get("mode")
        acct = accounts.get(f"round{rnd}") or {}
        rows = ((acct.get("kept") or []) + (acct.get("dropped") or []))
        plan = set(t.get("ordering") or [])

        _accounted = {r.get("candidate") for r in rows}
        _no_account = sorted(plan - _accounted)
        per = []
        for r in rows:
            if r.get("candidate") not in plan:
                continue
            c = r.get("contribution") or {}
            per.append({
                "candidate": r.get("candidate"), "heuristic": r.get("heuristic"),
                "critic_verdict": r.get("critic_verdict"),
                "summary": c.get("summary"),
                "throughput_h_per_case": c.get("throughput_h_per_case"),
                "throughput_z": c.get("throughput_z"),
                "throughput_significant": c.get("throughput_significant"),
                "waiting": c.get("waiting"), "duration": c.get("duration"),
                "resources_significant": c.get("resources_significant"),
                "attribution_caveat": c.get("attribution_caveat"),
                "exogenous_saving": c.get("exogenous_saving"),

                "what_the_handler_did": r.get("what_the_handler_did"),
            })

        opts.append({
            "id": oid, "round": rnd, "mode": mode,
            "candidates_with_no_account": _no_account,
            "plan": list(t.get("ordering") or []),

            "changed_the_model": (list(t.get("built"))
                                  if t.get("built") is not None else None),
            "roster_includes_refusals": (
                [c for c in (t.get("ordering") or []) if c not in set(t["built"])]
                if t.get("built") is not None
                and len(t["built"]) != len(t.get("ordering") or []) else None),
            "measures": _measure_lines(t.get("kpis") or {}),

            "gain_source_computed": _gain_source(t.get("kpis") or {}),

            "improvement_rule_computed": improvement_rule(t.get("kpis") or {}),

            **({"case_outcome_shift": (t.get("case_endings") or {}).get("headline"),
                "case_outcome_note": (t.get("case_endings") or {}).get("note")}
               if (t.get("case_endings") or {}).get("significant") else {}),
            **_attribution_scale(per),
            "per_candidate": per,
            "per_candidate_from": (f"round{rnd} (this is a CONFIRMATION — a subset "
                                   f"probe inside that round, so the rows describe "
                                   f"the round's fuller plan)"
                                   if mode == "confirming" else f"round{rnd}"),

            **({} if per else
               {"per_candidate_unavailable": (
                   "no per-candidate account was written for this round — judge it "
                   "on its measures alone, and do NOT read the absence as evidence "
                   "that its candidates contributed nothing")}),
        })
    return opts

def select(options: list, *, model: str | None = None) -> Selection:
    """Pick the answer. Code has already closed the set; the model weighs it."""
    if not options:
        return Selection(fallback="no measured plan to choose from")
    if len(options) == 1:
        o = options[0]
        return Selection(chosen=o["id"], why="the only measurement available",
                         fallback="single option, no judgement needed")

    user = "\n".join([
        "These are every measured plan still eligible to be this run's answer.",
        "Each measure carries its own z, computed by code.",
        "",
        json.dumps(options, ensure_ascii=False, indent=2, default=str),
    ])
    model = model or config.REASONING_MODEL
    items: list[Any] = [{"role": "system", "content": SELECT_SYSTEM},
                        {"role": "user", "content": user},
                        {"role": "user", "content": _SCHEMA}]
    out = respond_json(items, model=model, max_output_tokens=_OUT_MAX)

    ids = {o["id"] for o in options}
    chosen = str(out.get("chosen") or "")
    sel = Selection(
        chosen=chosen, why=str(out.get("why") or ""),
        gain_reading=str(out.get("gain_reading") or ""),
        rule_reading=str(out.get("rule_reading") or ""),
        passengers=list(out.get("passengers") or []),
        runner_up=str(out.get("runner_up") or ""),
        caveat=str(out.get("caveat") or ""))

    if chosen not in ids:

        sel.fallback = (f"the model returned {chosen!r}, which is not one of the "
                        f"measured options {sorted(ids)}; fell back to the first")
        sel.model_text_about = {
            "plan_named_by_the_model": chosen, "why": sel.why,
            "gain_reading": sel.gain_reading, "runner_up": sel.runner_up,
            "caveat": sel.caveat, "passengers": sel.passengers,
            "note": ("Written about a plan that is not among the measured options, "
                     "so it does not describe the recommendation. Kept only so the "
                     "model's answer is auditable."),
        }
        sel.chosen = options[0]["id"]
        sel.why = (f"NO AGENT JUDGEMENT STANDS BEHIND THIS CHOICE. The Selector "
                   f"named {chosen!r}, which was never measured, so code fell back "
                   f"to the first eligible option. Read the measures and the "
                   f"per-change account directly; what the model wrote is under "
                   f"`model_text_about` and is about a different plan.")
        sel.gain_reading = ""
        sel.passengers = []
        sel.runner_up = ""
        sel.caveat = ""
    return sel
