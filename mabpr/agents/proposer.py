"""PROPOSER — ground on the priors, then turn the Critic's directive into a plan."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mabpr import config
from mabpr.evidence.exogenous import exogenous_saving
from mabpr.evidence.footprint import _declared_targets
from mabpr.evidence.scorecard import ledger_digest, ledger_for
from mabpr.knowledge.tools import KnowledgeBox
from mabpr.llm.client import function_calls, respond, respond_json

_PROPOSE_MAX = 6000
_GROUND_MAX = 8000
MAX_GROUND_STEPS = 8

@dataclass
class Proposal:
    """One candidate set to apply next. ``ordering`` is the sequence Phase-3 will"""

    plan: list[str] = field(default_factory=list)
    ordering: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    rationale: str = ""
    responds_to: str = ""

    decisions: dict = field(default_factory=dict)
    fallback: str = ""
    grounding: list[str] = field(default_factory=list)
    prior_vs_measured: str = ""

    unmeasured_unanswered: list = field(default_factory=list)

    dropped_exogenous_only: list = field(default_factory=list)

    unmeasured_offered: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Proposal":
        plan = [str(x) for x in (d.get("plan") or [])]
        ordering = [str(x) for x in (d.get("ordering") or [])] or list(plan)
        return cls(plan=plan, ordering=ordering, params=dict(d.get("params") or {}),
                   rationale=str(d.get("rationale") or ""),
                   responds_to=str(d.get("responds_to") or ""),
                   decisions={str(k): str(v) for k, v in
                              (d.get("decisions") or {}).items()},
                   prior_vs_measured=str(d.get("prior_vs_measured") or ""))

PROPOSE_SYSTEM = """\
You are the Proposer in a closed-loop business-process-redesign system. A Critic
has just measured the current redesign against a fixed baseline and issued a
directive. Your job is to produce the NEXT candidate set.

YOUR ACTION SPACE IS EXACTLY THREE THINGS — nothing else:
  1. WHICH candidates to include (a subset of the detected ones);
  2. In WHAT ORDER they are applied;
  3. Their params (currently always {} — no handler takes one).

You do NOT invent redesigns, do not modify a candidate's target, and do not judge
whether a plan is legal or compliant — a separate Arbiter owns that. You only
choose from the catalogue you are given.

ORDER MATTERS AND IS A REAL LEVER. Candidates are applied in sequence, so one that
deletes an activity can nullify a later one that targets the same activity — that
candidate then produces NO signal at all ("no_sim_signal", see its verdict). If you
want such a candidate to have a chance, place it BEFORE the one that removes its
target. If you deliberately keep it nullified, say so.

THE DIRECTIVE'S INTENT IS ONE OF: drop | add | swap | reorder | keep. Each is
something you can actually do. `keep` means THE PLAN STANDS — the named candidates
earn their place and there is nothing to change about them; if you drop one anyway
you must say what evidence overrides the Critic, and "the directive said so" is not
that evidence. A directive may also carry `unexecutable_intent`: the Critic asked
for something outside your action space (usually a parameter change, which no
handler accepts), code rewrote it to `keep`, and its `reason` still tells you what
the Critic was worried about — address it with ORDER or with a drop, or state that
you are living with it.

RULES:
  - Follow the Critic's directive unless the evidence you are shown contradicts it;
    if you depart from it, say why in the rationale. Cite only what the directive
    ACTUALLY says: it carries `targets` and `avoid`, and an empty `avoid` means no
    pair was forbidden.
  - Never repeat a plan already tried (same ids in the same order) — you are shown
    the history with what each scored.
  - Never include a pair listed in anti_combos.
  - Keep the plan non-empty.
  - A candidate the Critic judged "helped" should normally stay; one judged
    "contended" or "hurt" with no offsetting benefit is the natural thing to drop.
  - READ `MEASURED_HISTORY` BEFORE BRINGING A CANDIDATE BACK. It is every round
    this candidate was measured in: the verdict, its hours-per-case with the
    scatter and whether that was SIGNIFICANT, and which candidates ran alongside
    it. Re-proposing something already measured harmful costs a 20-run
    measurement to re-learn what is written there. Bring it back only when
    something changed that plausibly changes its effect — a different ORDER, or a
    different set around it (`co_applied`) — and say which in its decision.
  - EXOGENOUS_SAVING on a catalogue entry is what the LOG says this candidate
    removes and the simulator cannot represent — overnight waits, handovers,
    batching. It carries `hours_per_case_estimate`, on the same scale as the
    measured throughput column so the two can actually be weighed. "No simulated
    effect" and "no effect" are different claims, and this line is the difference.
    HOW TO USE IT, and the line not to cross:
      · measures FLAT and changed nothing structurally, WITH an EXOGENOUS saving
        — "exogenous" here means the saving is visible in the recorded LOG but
        lies outside anything this simulator can represent, so the simulated
        columns are flat while the evidence is not —
        -> KEEP IT. Its verdict will read `exogenous_only`. The log says it
        removes a real wait; the model has no way to express that wait, which is
        why every column is empty. Keeping it cannot move any simulated number,
        so the plan measurement stays exactly as clean as if it were absent —
        there is no cost to set against the benefit. Dropping such a candidate
        for "no measured effect" is the specific mistake this rule exists to
        stop.
      · has a MEASURED effect -> decide on the measurement. Never let an estimate
        outvote a measured harm. A candidate can measure a significant case-time
        INCREASE while claiming a large per-occurrence saving the simulator
        cannot see — injecting that saving into the baseline can even reverse the
        sign. The harm is real and the plan would carry it; the saving is an
        estimate nothing can check.
        ⚠️ WHEN YOU KEEP ONE ANYWAY, SAY SO IN THOSE WORDS. This rule reads as
        absolute and the magnitudes sometimes are not: a measured harm of well
        under a minute per case, significant only because the run is precise, has
        been kept against an estimated saving orders of magnitude larger. That may well
        be the right call. What was wrong is that nothing recorded it: the
        directive said "keep" with no mention that the candidate's only measured
        contribution is a worsening and its case rests entirely on a number
        nothing can check. A candidate whose MEASURED contribution is negative
        does not earn its place by measurement; if you keep it, name it in your
        reason, give both numbers, and say you are backing the estimate. Never
        add the two together and never rank on the estimate.
      · a STRUCTURAL_NOOP with NO exogenous saving is pure clutter: it changed
        nothing and there is no log evidence to set against that — drop it and
        name the number you decided on.
      · never SUM the two, and never rank candidates on the estimate — its size
        comes from the detector's inputs, not from anything measured.
  - JUDGE ON ALL FOUR DIMENSIONS, not on throughput. Each verdict carries a
    `dimensions` map (waiting / duration / resources / throughput). A candidate
    that improves duration or resource load while leaving throughput flat HAS an
    effect — dropping it needs a reason on the dimension that actually moved.
    Say which dimension your decision rests on.
  - A number that is not SIGNIFICANT is not a small effect, it is no measurement.
    Do not keep or drop a candidate on one. If `VERDICT_CHANGED` appears and no
    round was significant, the disagreement is noise: prefer leaving that
    candidate alone this round over reacting to it.
  - The search is NOT one-way. Candidates with "in_last_plan": false are still
    available — bring one back when the evidence now favours it (for example a
    candidate that was nullified because another deleted its target, which you can
    revive by ORDERING it first, or one dropped earlier whose blocker is now gone).
  - Prefer ONE deliberate change per round over reshuffling everything: the next
    measurement must be interpretable, and a plan that changes five things at once
    tells you nothing about which change mattered.

GROUNDING. You have tools onto the HI-Graph — the literature's PRIOR about each
heuristic (its Devil's Quadrangle impact vector: time / cost / quality /
flexibility) and about each pair (HARD / WEAK / SYNERGY / NEUTRAL, plus whether the
structural condition fires on two specific candidates). Look up what you need
before choosing: the measurement tells you what happened, the prior tells you what
the edit was FOR, and the two together are what makes a selection defensible rather
than reactive.

The prior is a REFUTABLE GUESS, not an instruction. Where the measurement
contradicts it, follow the measurement and SAY SO — that disagreement is evidence
the knowledge base needs correcting, and recording it is part of your job. Do not
bend a verdict to fit a prior, and do not pretend a prior said something it did
not. Look up only what bears on this decision; you are not rewarded for volume."""

PROPOSE_SCHEMA = """\
Return JSON with EXACTLY these keys:
{
  "plan": ["<candidate ids to apply>"],
  "ordering": ["<the same ids, in the order Phase-3 should apply them>"],
  "params": {},
  "rationale": "why this set and this order — cite the directive, the measured
                verdicts, and any prior you looked up",
  "decisions": {"<candidate id>": "ONE sentence: why this candidate is in the plan,
                or why it is out. Cite its measured contribution (the scorecard's
                net line) or the rule that removed it."},
  "responds_to": "critic_round<N>.directive",
  "prior_vs_measured": "where a prior you consulted disagreed with what was
                        measured, or empty if none did"
}
"plan" and "ordering" must contain the SAME ids.
"decisions" must have an entry for EVERY candidate you kept, EVERY candidate you
removed from the previous plan, and EVERY candidate listed under NEVER MEASURED,
AND NOTHING IS STOPPING THEM — a reader has to be able to ask "why is this one
here?" (or "why is it still not here?") about any of them and get an answer
specific to that candidate. For a NEVER MEASURED one, "not worth a slot because
…" is a complete answer; saying nothing is not. Code does not require you to
propose them and never will — it records only whether you answered.
Output ONLY the JSON object."""

def unbuildable(build_history: dict | None) -> dict[str, dict]:
    """Candidates Phase-3 has refused for a reason ORDER CANNOT FIX."""

    out: dict[str, dict] = {}
    for cid, recs in (build_history or {}).items():
        if any(r.get("applied") for r in recs):
            continue
        hard = [r for r in recs if not r.get("applied") and not r.get("order_dependent")]
        if hard:
            out[cid] = {"attempts": len(hard), "rounds": [r["round"] for r in hard],
                        "reason": hard[-1]["reason"]}
    return out

def gate_prune_facts(gate_ruling: dict | None, never_builds: dict | set | None,
                     ) -> dict[str, dict]:
    """Why the gate removed a candidate, and whether that reason STILL HOLDS."""

    out: dict[str, dict] = {}
    if not gate_ruling:
        return out
    blocked = set(gate_ruling.get("blocked") or [])
    cannot_run = blocked | set(never_builds or ())
    survival = gate_ruling.get("survival") or {}
    for row in gate_ruling.get("rulings") or []:
        cid = row.get("candidate_id")
        if row.get("ruling") != "prune" or cid not in set(gate_ruling.get("pruned") or []):
            continue
        basis = row.get("basis")
        if basis == "compliance":
            continue
        removers = set((survival.get(cid) or {}).get("removed_by") or [])
        still = sorted(removers - cannot_run)
        entry = {"basis": basis, "reason": row.get("reason"),
                 "removed_by_on_paper": sorted(removers),
                 "still_removed_by": still}
        if removers and not still:
            entry["note"] = (
                "THE REASON FOR THIS PRUNE NO LONGER HOLDS: every candidate that "
                "would have deleted its target either is compliance-blocked or has "
                "been refused by Phase-3, so the target is still in the model. "
                "Placing it EARLIER than those candidates is a real lever "
                "(DECISIONS A) and it has never been measured.")
        out[cid] = entry
    return out

def unblocked_and_unmeasured(gate_ruling: dict | None,
                             never_builds: dict | set | None,
                             ledger: list | None,
                             build_history: dict | None = None) -> list[dict]:
    """Candidates NOTHING IS STOPPING ANY MORE, and that no round has measured."""

    facts = gate_prune_facts(gate_ruling, never_builds)
    out = []
    seen: set = set()
    for cid, f in sorted(facts.items()):
        if f.get("still_removed_by"):
            continue
        if any(r.get("applied") for r in ledger_for(ledger, cid)):
            continue
        out.append({
            "candidate_id": cid,
            "pruned_because": f.get("reason"),
            "removers_on_paper": f.get("removed_by_on_paper"),
            "why_that_no_longer_holds": ("every one of those is either "
                                         "compliance-blocked or has been refused "
                                         "by Phase-3, so none of them runs and the "
                                         "target is still in the model"),
            "times_measured": 0,
        })
        seen.add(cid)

    banned = set(never_builds or ())
    for cid, recs in sorted((build_history or {}).items()):
        if cid in seen or cid in banned:
            continue
        if any(r.get("applied") for r in recs):
            continue
        if any(r.get("applied") for r in ledger_for(ledger, cid)):
            continue
        probe = next((r["probe"] for r in reversed(recs)
                      if (r.get("probe") or {}).get("builds_alone") is True), None)
        if probe is None:
            continue
        last = next((r for r in reversed(recs) if not r.get("applied")), {})
        out.append({
            "candidate_id": cid,
            "pruned_because": str(last.get("reason") or "")[:300],
            "removers_on_paper": None,
            "why_that_no_longer_holds": ("applied ALONE on the baseline it builds, "
                                         "so that refusal was about the other "
                                         "candidates in the plan, not about this one"),
            "times_measured": 0,
        })
        seen.add(cid)
    return out

def _catalogue(all_candidates: list[dict], footprints: dict[str, dict],
               assessment: dict | None, build_history: dict | None = None,
               harmful_history: dict | None = None,
               ledger: list | None = None,
               gate_ruling: dict | None = None) -> list[dict]:
    """Compact grounding: what each candidate is, what it touches, and how it did."""

    verdicts = {e["candidate_id"]: e for e in
                ((assessment or {}).get("assessment", {}).get("per_candidate") or [])}
    barred = unbuildable(build_history)
    gate_pruned = gate_prune_facts(gate_ruling, set(barred))
    out = []
    for cand in all_candidates:
        cid = cand["candidate_id"]
        fp = footprints.get(cid) or {}
        v = verdicts.get(cid, {})
        entry = {
            "candidate_id": cid,
            "heuristic": cand.get("heuristic") or fp.get("heuristic"),
            "targets": fp.get("activities") or sorted(_declared_targets(cand)),
            "in_last_plan": bool(fp),
            "applied_last_round": fp.get("applied"),
            "skip_reason": fp.get("skip_reason") or None,
            "last_verdict": v.get("verdict"),
            "last_confidence": v.get("confidence"),
            "last_note": v.get("note"),
        }

        hist = ledger_digest(ledger, cid)
        if hist:
            entry["MEASURED_HISTORY"] = hist

        exo = exogenous_saving(cand)
        if exo:
            entry["EXOGENOUS_SAVING"] = exo

        if (fp or {}).get("structural_noop"):
            entry["STRUCTURAL_NOOP"] = (
                "applied=true, but the handler made no change to the model at all. "
                "Measured on BPI 2012 and 2017: the CBW handler's only action is "
                "removing batch_processing rules, and neither model has any — 20 "
                "applications, rules_removed=0 every time. Keeping it cannot help "
                "in any ordering.")

        past = (harmful_history or {}).get(cid)
        if past:
            entry["MEASURED_HARMFUL"] = {
                "rounds": [p["round"] for p in past],
                "verdicts": sorted({p["verdict"] for p in past}),
                "last_note": past[-1].get("note", ""),
                "note": ("Measured to make things WORSE in an earlier round, not "
                         "merely useless. Proposing it again repeats a measurement "
                         "already paid for — unless you place it in a different "
                         "ORDER, which can genuinely change what it does."),
            }

        if cid in gate_pruned and gate_pruned[cid].get("still_removed_by"):
            entry["GATE_PRUNED_LAST_ROUND"] = gate_pruned[cid]
        if cid in barred:
            entry["UNBUILDABLE"] = {
                **barred[cid],
                "note": ("Phase-3 refused to build this for a reason no ordering "
                         "fixes. Do NOT propose it again — the plan slot is wasted."),
            }
        out.append(entry)
    return out

def propose(
    round_no: int,
    all_candidates: list[dict],
    footprints: dict[str, dict],
    assessment: dict | None,
    *,
    knowledge: KnowledgeBox | None = None,
    current_plan: list[str] | None = None,
    tried_plans: list[dict] | None = None,
    gate_ruling: dict | None = None,
    anti_combos: list[list[str]] | None = None,
    contested: list[dict] | None = None,
    blocked: list[str] | None = None,
    build_history: dict | None = None,
    harmful_history: dict | None = None,
    ledger: list | None = None,
    model: str | None = None,
    max_steps: int = MAX_GROUND_STEPS,
) -> Proposal:
    """Ground on the priors, propose the next candidate set, then check it."""

    known = [c["candidate_id"] for c in all_candidates]

    _unmeasured = unblocked_and_unmeasured(gate_ruling, set(unbuildable(build_history)),
                                           ledger, build_history=build_history)
    directive = (assessment or {}).get("directive")
    conflicts = (assessment or {}).get("assessment", {}).get("conflicts") or []

    user = "\n".join([
        f"Round {round_no}. Propose the candidate set for the next round.",
        "",
        "CANDIDATE CATALOGUE (the only ids you may use). It lists EVERY detected "
        "candidate, including ones not in the current plan — 'in_last_plan': false "
        "means it is available to bring back, not that it is forbidden:",
        json.dumps(_catalogue(all_candidates, footprints, assessment, build_history,
                              harmful_history, ledger, gate_ruling),
                   ensure_ascii=False, indent=2),
        "",
        f"CURRENT PLAN (just measured): {current_plan or []}",
        "",
        "CRITIC'S DIRECTIVE:",
        json.dumps(directive, ensure_ascii=False, indent=2),
        "",
        "CONFLICTS THE CRITIC FOUND:",
        json.dumps(conflicts, ensure_ascii=False, indent=2),
        "",
        "PLANS ALREADY TRIED (do not repeat):",
        json.dumps(tried_plans or [], ensure_ascii=False, indent=2, default=str),
        "",
        f"ANTI-COMBOS (never together): {anti_combos or []}",
        "",

        *(["WHICH CLAIMANT THE MEASURED SIGN RULES OUT. For an activity several "
           "applied candidates touch, the account attributes it to none of them "
           "(sole-owner rule) -- but a transformation that can only push that "
           "wait the OTHER way cannot be what was measured. This is the handlers' "
           "own reports plus the measured sign; no isolation was run. It says "
           "nothing about whether a ruled-out candidate is worth keeping.",
           json.dumps([{"activity": c.get("activity"),
                        "measured": (c.get("mechanism_direction") or {}).get("measured"),
                        "ruled_out_by_mechanism":
                            (c.get("mechanism_direction") or {}).get("ruled_out_by_mechanism"),
                        "compatible_with_the_measured_sign":
                            (c.get("mechanism_direction") or {}).get(
                                "compatible_with_the_measured_sign"),
                        "per_claimant": (c.get("mechanism_direction") or {}).get("per_claimant")}
                       for c in (contested or []) if c.get("mechanism_direction")],
                      ensure_ascii=False, indent=2),
           ""]
          if any((c or {}).get("mechanism_direction") for c in (contested or []))
          else []),
        f"COMPLIANCE-BLOCKED (the Arbiter ruled these must not be applied; do NOT "
        f"propose them again): {blocked or []}",
        "",

        *(["NEVER MEASURED, AND NOTHING IS STOPPING THEM. The gate removed each of "
           "these because an earlier candidate would delete its target — and none "
           "of those deleters runs (compliance-blocked, or refused by Phase-3), so "
           "the target is still in the model. No round has ever measured them, so "
           "there is no evidence for OR against them. Proposing one costs a plan "
           "slot; you decide whether that is worth it, and code does not:",
           json.dumps(_unmeasured, ensure_ascii=False, indent=2), ""]
          if _unmeasured else []),
        "ARBITER'S SUGGESTED ORDER for the plan it just ruled on (a recommendation "
        "only — it can revive a candidate that was pruned for aiming at something "
        "an earlier candidate deletes; you decide whether it is worth a slot):",
        json.dumps((gate_ruling or {}).get("reorder_suggestion") or None,
                   ensure_ascii=False),
    ])

    model = model or config.REASONING_MODEL
    items: list[Any] = [{"role": "system", "content": PROPOSE_SYSTEM},
                        {"role": "user", "content": user + "\n\nLook up whatever priors "
                                                           "bear on this choice, then stop."}]

    steps = 0
    if knowledge is not None:
        while steps < max_steps:
            resp = respond(items, model=model, tools=knowledge.specs,
                           max_output_tokens=_GROUND_MAX)
            calls = function_calls(resp)
            if not calls:
                break

            items.extend(item.model_dump(exclude_none=True) for item in resp.output)
            for call in calls:
                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError as e:
                    args, result = {}, {"error": f"unparsable arguments: {e}"}
                else:
                    result = knowledge.execute(call.name, args)
                items.append({"type": "function_call_output", "call_id": call.call_id,
                              "output": json.dumps(result, ensure_ascii=False, default=str)})
            steps += 1

    items.append({"role": "user", "content": PROPOSE_SCHEMA})
    out = respond_json(items, model=model, max_output_tokens=_PROPOSE_MAX)

    proposal = Proposal.from_dict(out)
    proposal.grounding = knowledge.grounding() if knowledge else []
    if not proposal.responds_to:
        proposal.responds_to = f"critic_round{round_no - 1}.directive"

    proposal.unmeasured_offered = [r["candidate_id"] for r in _unmeasured]
    proposal.unmeasured_unanswered = [
        cid for cid in proposal.unmeasured_offered
        if cid not in (proposal.decisions or {})]

    _exo_only = {e.get("candidate_id") for e in
                 ((assessment or {}).get("assessment", {}).get("per_candidate") or [])
                 if e.get("verdict") == "exogenous_only"}
    proposal.dropped_exogenous_only = sorted(
        _exo_only & set(current_plan or []) - set(proposal.plan))
    ok, why = validate(proposal, known, tried_plans, anti_combos, blocked,
                       build_history)
    if not ok:
        repaired, rung = _mechanical_valid(
            directive, current_plan or known, known, tried_plans, anti_combos,
            blocked, build_history=build_history, ledger=ledger)
        why = f"{why}; fallback used: {rung}"
        repaired.rationale = (f"[code fallback: {why}] "
                              f"applied the directive mechanically. "
                              f"Model's rejected proposal: {proposal.plan}")
        repaired.responds_to = proposal.responds_to
        repaired.grounding = proposal.grounding
        repaired.prior_vs_measured = proposal.prior_vs_measured
        repaired.fallback = why
        return repaired
    return proposal

def validate(p: Proposal, known: list[str], tried_plans: list[dict] | None,
             anti_combos: list[list[str]] | None,
             blocked: list[str] | None = None,
             build_history: dict | None = None) -> tuple[bool, str]:
    """Code's rules for a proposal. The model chooses; code checks."""
    if not p.plan:
        return False, "empty plan"
    banned = sorted(set(blocked or []) & set(p.plan))
    if banned:
        return False, f"compliance-blocked candidate(s) in the plan: {banned}"

    dead = sorted(set(unbuildable(build_history)) & set(p.plan))
    if dead:
        return False, (f"candidate(s) Phase-3 has already refused to build for a "
                       f"reason ordering cannot fix: {dead}")
    unknown = [c for c in p.plan if c not in known]
    if unknown:
        return False, f"invented candidate ids {unknown} (catalogue is {known})"
    if set(p.plan) != set(p.ordering):
        return False, f"plan {sorted(p.plan)} and ordering {sorted(p.ordering)} disagree"
    if len(set(p.ordering)) != len(p.ordering):
        return False, f"ordering repeats an id: {p.ordering}"
    for prev in tried_plans or []:
        if list(prev.get("ordering") or prev.get("plan") or []) == list(p.ordering):
            return False, f"plan already tried in round {prev.get('round')}"
    for pair in anti_combos or []:
        if set(pair) <= set(p.plan):
            return False, f"anti-combo {pair} is in the plan"
    return True, "ok"

def _mechanical(directive: dict | None, current_plan: list[str], known: list[str],
                tried_plans: list[dict] | None,
                anti_combos: list[list[str]] | None,
                blocked: list[str] | None = None,
                build_history: dict | None = None) -> Proposal:
    """Deterministic realisation of the directive — the safety net, not the path."""

    ban = set(blocked or []) | set(unbuildable(build_history))
    plan = [c for c in (current_plan or known) if c in known and c not in ban]
    intent = (directive or {}).get("intent")
    targets = [str(t) for t in ((directive or {}).get("targets") or [])]

    if intent == "drop":
        plan = [c for c in plan if c not in targets]
    elif intent == "add":
        plan = plan + [t for t in targets
                       if t in known and t not in plan and t not in ban]
    elif intent == "swap":

        keep_ = [c for c in plan if c not in targets]
        plan = keep_ + [t for t in targets
                        if t in known and t not in plan and t not in ban]
    elif intent == "reorder":

        front = [t for t in targets if t in plan]
        plan = front + [c for c in plan if c not in front]

    if not plan:
        plan = [c for c in (current_plan or known) if c in known and c not in ban]
    return Proposal(plan=list(plan), ordering=list(plan), params={},
                    rationale="mechanical fallback", responds_to="")

def _worst_first(plan: list[str], ledger: list | None) -> list[str]:
    """The plan's candidates, weakest measured contribution first."""

    def rank(cid: str) -> tuple:
        rows = [r for r in (ledger or [])
                if r.get("candidate_id") == cid and r.get("applied")]
        if not rows:
            return (2, 0.0)
        last = rows[-1]
        if last.get("structural_noop"):
            return (3, 0.0)
        thr = last.get("thr_h_per_case") or 0.0
        if last.get("thr_significant") and thr > 0:
            return (0, -thr)
        if not last.get("any_measurable_effect"):
            return (1, 0.0)
        return (2, thr)
    return sorted(plan, key=lambda c: (rank(c), plan.index(c)))

def _mechanical_valid(directive: dict | None, current_plan: list[str],
                      known: list[str], tried_plans: list[dict] | None,
                      anti_combos: list[list[str]] | None,
                      blocked: list[str] | None = None,
                      build_history: dict | None = None,
                      ledger: list | None = None) -> tuple[Proposal, str]:
    """The fallback, CHECKED — and able to escape when the directive alone cannot."""

    def _check(p: Proposal) -> bool:
        return validate(p, known, tried_plans, anti_combos, blocked, build_history)[0]

    base = _mechanical(directive, current_plan, known, tried_plans, anti_combos,
                       blocked, build_history=build_history)
    if _check(base):
        return base, "the directive, applied mechanically"

    if (directive or {}).get("intent") == "keep":
        return base, ("the directive is `keep` and its plan is already measured — "
                      "that is convergence, so nothing was shed")

    for cid in _worst_first(list(base.ordering), ledger):
        trimmed = [c for c in base.ordering if c != cid]
        if not trimmed:
            continue
        p = Proposal(plan=list(trimmed), ordering=list(trimmed), params={},
                     rationale="mechanical fallback", responds_to="")
        if _check(p):
            return p, (f"the directive would have repeated a measured plan, so "
                       f"{cid} — the weakest measured candidate in it — was dropped too")

    return base, ("no deterministic variation of the directive is unmeasured; "
                  "this round rebuilds a design already simulated")
