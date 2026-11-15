"""CRITIC — a hypothesis-driven investigation loop (v1)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from mabpr import config
from mabpr.evidence.tools import ToolBox
from mabpr.llm.client import function_calls, respond, respond_json

MAX_STEPS = 12
_ORIENT_MAX = 6000
_STEP_MAX = 8000
_CONCLUDE_MAX = 8000

VERDICTS = ("helped", "neutral", "hurt", "covered", "contended", "no_sim_signal",
            "exogenous_only")
CONFIDENCES = ("measured", "attributed", "hypothesized")

INTENTS = ("drop", "add", "swap", "reorder", "keep")

def normalize_directive(d: "dict | None", per_candidate: "list | None" = None,
                        scorecard: "dict | None" = None) -> "dict | None":
    """Force a directive into something the Proposer can execute, and legal."""

    if not d:
        return d

    protected = {e.get("candidate_id") for e in (per_candidate or [])
                 if e.get("verdict") == "exogenous_only"}
    if protected and str(d.get("intent")) == "drop":
        targets = list(d.get("targets") or [])
        kept = [t for t in targets if t in protected]
        if kept:
            remaining = [t for t in targets if t not in protected]
            d = {**d, "targets": remaining,
                 "refused_drop": kept,
                 "refused_drop_reason": (
                     f"{kept} are `exogenous_only`: the simulator cannot represent "
                     f"their mechanism, which is not the same as their doing "
                     f"nothing. Keeping them costs no simulation slot (the design "
                     f"is byte-identical either way), so there is nothing to weigh "
                     f"against the log-measured saving they carry.")}
            if not remaining:
                d = {**d, "intent": "keep", "targets": []}

    hurt = {e.get("candidate_id") for e in (per_candidate or []) if e.get("verdict") == "hurt"}
    if hurt:
        intent0 = str(d.get("intent") or "")
        targets0 = set(d.get("targets") or [])
        if intent0 == "keep":
            kept_hurt = sorted(hurt & targets0)
        elif intent0 in ("drop", "swap"):
            kept_hurt = sorted(hurt - targets0)
        else:
            kept_hurt = sorted(hurt)
        if kept_hurt:

            _rows = (scorecard or {}).get("candidates") or {}
            _costs = []
            for c in kept_hurt:
                net = (_rows.get(c) or {}).get("net") or {}
                thr = net.get("throughput_h_per_case")
                z = net.get("throughput_z")
                if isinstance(thr, (int, float)):
                    _costs.append("%s costs %+.4f h per case%s on its own account"
                                  % (c, thr,
                                     (" (z=%+.1f)" % z) if isinstance(z, (int, float)) else ""))
                else:
                    _costs.append("%s has no sole-owner account (its activities are "
                                  "shared), so its own cost is NOT ESTABLISHED -- "
                                  "which is not the same as zero" % c)
            d = {**d, "keeps_hurt": kept_hurt,
                 "keeps_hurt_cost": _costs,
                 "keeps_hurt_note": (f"the Critic's own per_candidate verdict is `hurt` for "
                                     f"{kept_hurt}, and this directive leaves them in the "
                                     f"plan. " + "; ".join(_costs) + ".")}

    intent = str(d.get("intent") or "")
    if intent in INTENTS:
        return d
    return {**d, "intent": "keep", "unexecutable_intent": intent,
            "note": (f"intent {intent!r} is not something the Proposer can carry out "
                     f"(its action space is selection and ORDER only; every handler's "
                     f"params are {{}}). Read as `keep`: the reason still stands, but "
                     f"the only levers are which candidates run and in what order.")}

@dataclass
class Hypothesis:
    """A falsifiable guess about what the simulation shows. Hypotheses are the"""

    id: str
    claim: str
    about: list[str] = field(default_factory=list)
    status: str = "open"
    confidence: str = "hypothesized"
    evidence: list[str] = field(default_factory=list)
    why_open: str = ""

    needs_confirmation: bool = False
    confirmation_priority: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        try:
            prio = int(d.get("confirmation_priority") or 0)
        except (TypeError, ValueError):
            prio = 0
        return cls(
            id=str(d.get("id") or "H?"),
            claim=str(d.get("claim") or ""),
            about=[str(x) for x in (d.get("about") or [])],
            status=str(d.get("status") or "open"),
            confidence=str(d.get("confidence") or "hypothesized"),
            evidence=[str(x) for x in (d.get("evidence") or [])],
            why_open=str(d.get("why_open") or ""),
            needs_confirmation=bool(d.get("needs_confirmation")),
            confirmation_priority=prio,
        )

_RULES = """\
RULES OF EVIDENCE (identical in every stage; never invent a rule):

1. Deterministic code already decided what is SIGNIFICANT (|z| > 3). You never
   re-judge significance — you judge meaning.
2. Read WAITING z and DURATION z SEPARATELY for each activity:
     waiting z > +3, duration z within [-3,+3]  -> CONTENTION (queueing for
        resources). NOT "the activity itself got slower".
     waiting z > +3 and duration z > +3         -> HANDLER BUG (the edit made the
        work itself slower — a code defect, not a conflict).
     waiting z < -3                             -> BENEFIT.
3. ATTRIBUTION. A candidate's FOOTPRINT is the record of what its handler
   actually edited in the model — `footprint.activities` lists the activities it
   touched, and `applied` says whether the edit happened at all. It is a fact
   read back out of the build, not a claim.
   An activity belongs to candidate C if it is in C's footprint.activities. An
   activity in NO candidate's footprint is queueing SPILLOVER: something moved on
   an activity nobody edited, because coupling in this process runs through
   QUEUES, not through shared activity sets.
4. CONFIDENCE, never mixed in one tone:
     measured     = a z you were handed, or a result proven by a confirmation simulation.
     attributed   = causal claim supported by a footprint, NOT counterfactually proven.
     hypothesized = a guess with no footprint support.
   Without a confirmation simulation, every cross-candidate causal claim is at
   most "attributed".
5. A candidate whose footprint has applied=false did nothing: its target was
   already removed by another candidate (see skip_reason). That is NOT a conflict.
"""

_ORIENT_SYSTEM = f"""\
You are the Critic in a closed-loop business-process-redesign system, at the
ORIENT stage. You are shown the cheap evidence slice: case-level KPIs, the
per-activity waiting/duration z table (sorted by |waiting z|), and each
candidate's footprint.

{_RULES}

READ THE SCORECARD FIRST. `bundle.scorecard.candidates` already answers, for
every applied candidate, what it did to waiting, duration, resource utilisation
and hours-per-case throughput. That account is complete and deterministic — do
NOT restate it as a hypothesis. A candidate whose columns are all empty did
nothing; that is settled, not suspicious.

YOUR ONLY JOB HERE: hypotheses about what the scorecard CANNOT settle. There are
three kinds, and nothing else qualifies:

  1. SHARED CREDIT — an activity in `unattributable_activities`, claimed by two or
     more candidates. TWO things settle it before a hypothesis is warranted.

     FIRST look at `isolated_in_earlier_rounds`. Sharing is a property of the
     ROUND, not of the activity: an earlier round in which only ONE claimant was
     applied already measured that claimant on that activity. Each entry gives the
     round and the hours it moved there. Compare it with this round's number —
     the difference is the other claimant's marginal share. Report the measured
     figures; do NOT turn them into a percentage split, because the effects are
     often not additive: two claimants can EACH be sufficient on their own, in
     which case no ratio exists to report.

     THEN, if no earlier round isolates it, use the MECHANISM: the claimants are
     listed under `claimants` with what each one structurally did, and the
     heuristics differ.
        waiting up, duration flat            -> queueing -> the candidate that
                                                MERGED A RESOURCE POOL
        an activity disappeared              -> the candidate that DELETED it
        a new activity appeared              -> the candidate that made a SUBPROCESS
        duration up on the activity itself   -> the handler got slower, not a queue
     Only raise a hypothesis when the mechanisms genuinely cannot be told apart.
  2. UNOWNED CHANGE — a significant change on an activity in NO candidate's
     footprint. That is spillover; it needs explaining.
  3. MECHANISM IN DOUBT — the scorecard shows the size of an effect but not why,
     and knowing why would change what to do next.

Do not write final verdicts — later stages test these and conclude.
A good hypothesis is falsifiable and says what would settle it.

SIMULATION IS THE LAST RESORT, NOT THE DEFAULT. There are two confirmation runs
for the ENTIRE run. Spend one only on a question that no amount of reading can
answer — and reading includes get_adapt_report (what each handler actually did),
get_resource_load (which resources are saturated) and get_case_examples (what a
real case did). Interpreting the simulation you already have is the job; asking
for another one is an admission that interpretation was not enough.
"""

_ORIENT_SCHEMA = """\
Return JSON: {"hypotheses": [{"id": "H1", "claim": "...", "about": ["<candidate id>", "<activity>"],
"why_open": "what evidence would settle this"}]}
Give 2-6 hypotheses, ids H1, H2, ... Output ONLY the JSON object."""

def orient(round_no: int, bundle: dict, memory: Any | None = None,
           *, model: str | None = None) -> list[Hypothesis]:
    user = [f"Round {round_no}. Evidence slice:", json.dumps(bundle, ensure_ascii=False,
                                                             indent=2, default=str)]
    if memory:
        user += ["", "Observations carried from earlier rounds:",
                 json.dumps(memory, ensure_ascii=False, default=str)]
    user += ["", _ORIENT_SCHEMA]
    out = respond_json(
        [{"role": "system", "content": _ORIENT_SYSTEM}, {"role": "user", "content": "\n".join(user)}],
        model=model, max_output_tokens=_ORIENT_MAX)
    return [Hypothesis.from_dict(h) for h in (out.get("hypotheses") or [])]

_INVESTIGATE_SYSTEM = f"""\
You are the Critic at the INVESTIGATE stage. You hold hypotheses from ORIENT and
a set of tools. The evidence slice you already have is in the user message —
never call a tool for something already in it.

{_RULES}

WHAT THE TOOLS ADD (none of this is in the slice):
  get_resource_load      WHO a queue formed against: per-resource utilization /
                         task load with z, and the pool each resource sits in.
                         This is how you turn "waiting rose" into a mechanism.
  get_candidate_evidence what a candidate's detector originally CLAIMED. Compare
                         that prior against the measured result: a benefit that
                         never landed is itself a finding.
  get_adapt_report       what the edit actually changed structurally.
  get_case_examples      EXPENSIVE, hard-capped. Concrete traces. Only when
                         aggregate evidence cannot settle the hypothesis.

Every call must name the hypothesis it serves. Investigate the biggest signals
first. Stop calling tools as soon as further reading would not change a verdict —
you are not rewarded for volume. When you stop, you will be asked to update the
hypotheses and decide about a confirmation simulation.
"""

_DECIDE_SCHEMA = """\
Return JSON with EXACTLY these keys:
{
  "hypotheses": [{"id": "H1", "claim": "...", "about": [...],
                  "status": "open|confirmed|refuted|unresolved",
                  "confidence": "measured|attributed|hypothesized",
                  "evidence": ["what you read that bears on it"],
                  "why_open": "if still open, what would settle it",
                  "needs_confirmation": true|false,
                  "confirmation_priority": 0}],
  "confirmation_request": null OR
    {"hypothesis_id": "H1",
     "plan": ["<candidate ids to simulate together>"],
     "why_worth_it": "why reading cannot settle this and what the run would prove"}
}
Keep every hypothesis (updated), do not drop any.

MARK EVERY HYPOTHESIS, then request the one you marked most decisive.
  · `needs_confirmation` — true only when READING CANNOT SETTLE IT and a
    simulation would. If your own reasoning called a confirmation "optional" for
    a hypothesis, that is `false`.
  · `confirmation_priority` — 1 for the single most decisive, 2 for the next,
    and so on. 0 for every hypothesis marked false.
  · `confirmation_request.hypothesis_id` MUST be the hypothesis you gave
    priority 1. Code checks this and REFUSES a request that names a different
    one — not because code judges which matters (it cannot), but because your
    two answers must agree. A refused request costs no budget: mark and request
    consistently next round.
Output ONLY the JSON object."""

def removal_fact_lines(measured, unmeasured) -> list:
    """What DECIDE is told about counterfactuals already in hand — FACTS only."""

    out = []
    if measured:
        out.append(
            "REMOVALS ALREADY MEASURED THIS RUN — each of these is a settled "
            "counterfactual: two measurements whose plans differ by exactly this "
            "one candidate. Asking the same question again re-measures a design "
            "the run already has.")
        for r in measured:
            m = r["measures"]
            parts = []
            for key, name in (("throughput_h_median", "median"),
                              ("throughput_h_mean", "mean"),
                              ("waiting_h_per_case_mean", "waiting"),
                              ("duration_h_per_case_mean", "service")):
                if key in m:
                    parts.append("%s %+.2f%% -> %+.2f%% (%+.1f pp)"
                                 % (name, m[key]["with"], m[key]["without"],
                                    m[key]["pp"]))
            out.append("  - without %s (base of %d, rounds %s vs %s): %s"
                       % (r["candidate"], r["base_size"], r["with_round"],
                          r["without_round"], "; ".join(parts)))
    if unmeasured:
        out.append(
            "REMOVALS NOT YET MEASURED, for candidates in the current plan: "
            + ", ".join(unmeasured)
            + ". No measurement in this run isolates any of these, so a "
              "hypothesis about one of them cannot be settled by reading the "
              "tables you already have.")
    return out

def _decide_prompt(confirm_left: int, per_round_left: int, applied: list[str],
                  removal_lines: list = None) -> str:
    if confirm_left <= 0 or per_round_left <= 0:
        return ("CONFIRMATION BUDGET: exhausted. Set confirmation_request to null and mark "
                "hypotheses you cannot settle as \"unresolved\".")
    _REMOVALS = ("\n".join(removal_lines) + "\n\n") if removal_lines else ""
    return f"""\
CONFIRMATION SIMULATION — you may request at most ONE, and only if it is worth it.
Budget left: {confirm_left} for the whole run, {per_round_left} this round.

A confirmation simulation re-runs the process with a modified candidate set and
returns a COMPLETE new per-activity table — so one well-chosen run can settle
several hypotheses at once. It is deliberately scarce: simulating every
combination is exactly what this system exists to avoid. It also costs minutes.

{_REMOVALS}PLAN SHAPE — this matters, and it is checked. The candidates currently applied are:
  {applied}
Your "plan" is the set to apply TOGETHER in the confirmation run. It MUST be that
list with at least one candidate REMOVED — normally exactly the suspect. A plan
that keeps all of them re-runs the design already measured and will be REJECTED:
it buys no information. "Does the effect disappear without X?" is the question one
run answers most cleanly, and it re-measures every other candidate for free.
Do NOT instead propose keeping the suspect while dropping everything else:
isolating a candidate changes the whole competitive context, so the result would
not be comparable to the run you already have.

Request one ONLY if ALL hold:
  (a) reading genuinely cannot settle the hypothesis — and you have already read
      what is relevant to it (a request about a hypothesis you investigated with
      no tool call will be REJECTED);
  (b) the answer would change a verdict or a recommendation;
  (c) it is the highest-value question you could ask with this one run.
Otherwise set confirmation_request to null. Prefer null over a marginal request.

FIRST WALK THE HYPOTHESES ONE BY ONE and say, for each, whether reading can
settle it — THEN rank the ones it cannot. Do not decide what to request before
you have been through all of them. The next stage records that ranking as
`needs_confirmation` + `confirmation_priority` per hypothesis, and code REFUSES a
request that names anything other than your own rank-1. This costs no budget, but
it costs the round: on BPI 2012 the reasoning called one hypothesis's confirmation
"required" and another's "optional", requested the optional one, and the run's
two confirmations were spent without ever testing the candidate that mattered.
⚠️ "Which of these matters most" is YOUR judgment and code never makes it — code
only checks that the answer you give here is the answer you give next."""

def investigate(hypotheses: list[Hypothesis], bundle: dict, toolbox: ToolBox,
                *, model: str | None = None, confirm_left: int = 2,
                per_round_left: int = 1, removal_lines: list = None,
                max_steps: int = MAX_STEPS) -> tuple[list[Hypothesis], dict | None, dict]:
    """Run the audited tool loop, then extract updated hypotheses + an optional"""

    items: list[Any] = [
        {"role": "system", "content": _INVESTIGATE_SYSTEM},
        {"role": "user", "content": "\n".join([
            "Evidence slice already in hand:",
            json.dumps(bundle, ensure_ascii=False, indent=2, default=str),
            "", "Hypotheses to test:",
            json.dumps([asdict(h) for h in hypotheses], ensure_ascii=False, indent=2),
            "", "Call tools as needed, then stop.",
        ])},
    ]

    steps, hit_cap = 0, False
    while steps < max_steps:
        resp = respond(items, model=model, tools=toolbox.specs, max_output_tokens=_STEP_MAX)
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
                result = toolbox.execute(call.name, args)
            items.append({"type": "function_call_output", "call_id": call.call_id,
                          "output": json.dumps(result, ensure_ascii=False, default=str)})
        steps += 1
    else:
        hit_cap = True

    applied = [cid for cid, fp in (bundle.get("footprints") or {}).items()
               if fp.get("applied")]
    items.append({"role": "user", "content": "\n\n".join([
        _decide_prompt(confirm_left, per_round_left, applied, removal_lines),
        _DECIDE_SCHEMA,
    ])})
    out = respond_json(items, model=model, max_output_tokens=_STEP_MAX)

    updated = [Hypothesis.from_dict(h) for h in (out.get("hypotheses") or [])]
    if not updated:
        updated = hypotheses
    req = out.get("confirmation_request") or None
    return updated, req, {"tool_rounds": steps, "hit_max_steps": hit_cap}

def top_confirmation_ids(hypotheses) -> set:
    """The ids the model itself ranked most decisive."""

    marked = [h for h in (hypotheses or []) if getattr(h, "needs_confirmation", False)]
    if not marked:
        return set()
    prios = [(getattr(h, "confirmation_priority", 0) or 99) for h in marked]
    best = min(prios)
    return {h.id for h, p in zip(marked, prios) if p == best}

def confirmation_gate(req: dict | None, toolbox: ToolBox, *, confirm_left: int,
                      per_round_left: int,
                      applied: list[str] | None = None,
                      hypotheses: list | None = None) -> tuple[bool, str]:
    """May this confirmation simulation run?the model may request,"""

    if not req:
        return False, "no request"
    if confirm_left <= 0:
        return False, "run-level confirmation budget exhausted"
    if per_round_left <= 0:
        return False, "per-round confirmation budget exhausted"
    plan = [str(c) for c in (req.get("plan") or [])]
    if not plan:
        return False, "request carries no plan"

    if applied and not (set(applied) - set(plan)):
        return False, ("plan removes no applied candidate, so it re-runs the design "
                       f"already measured (applied={sorted(applied)})")
    hyp = str(req.get("hypothesis_id") or "")
    if not toolbox.reads_for(hyp):
        return False, (f"read-first gate: nothing was read about {hyp or '(unnamed)'}; "
                       "cheap evidence must be exhausted before spending a simulation")

    if hypotheses is not None:
        top = top_confirmation_ids(hypotheses)
        if not top:
            return False, ("self-inconsistent: a confirmation was requested but NO "
                           "hypothesis was marked needs_confirmation. Reading cannot "
                           "settle something only if you say which something; mark "
                           "the hypothesis and request it again next round")
        if hyp not in top:
            ranked = ", ".join(f"{h.id}(p{h.confirmation_priority})"
                               for h in hypotheses
                               if getattr(h, "needs_confirmation", False))
            return False, (f"self-inconsistent: {hyp} was requested, but you ranked "
                           f"{'/'.join(sorted(top))} most decisive [{ranked}]. Code does "
                           f"not judge which matters — it only refuses a request that "
                           f"contradicts your own ranking. No budget was spent")
    return True, "granted"

_CONCLUDE_SYSTEM = f"""\
You are the Critic at the CONCLUDE stage. The investigation is over. Turn the
hypotheses and their evidence into the final cards.

{_RULES}

6. VERDICT per candidate — pick exactly one. What each word means HERE:
     helped         its own measures got better on net.
     neutral        it was applied, it edited the model, and nothing moved.
                    "Nothing happened" is a real finding, not a missing answer.
     hurt           its own measures got worse on net.
     contended      the harm is QUEUEING, not slower work: waiting up, duration
                    flat. A mechanism, not automatically a net loss.
     covered        you cannot tell what it did, because every measure it touches
                    is shared with another applied candidate. NOT "no effect" —
                    "not established". Say so, and say who it is shared with.
     no_sim_signal  it never actually ran: the handler refused it, or an earlier
                    candidate had already removed its target. There is nothing to
                    measure, so do not read this as evidence against it.
     exogenous_only the LOG shows a saving the SIMULATOR cannot represent, so the
                    simulated columns are flat while the log evidence is not.
                    Flat-in-simulation is not the same as worthless.
   A candidate whose own activity now queues, with no offsetting benefit, is a
   net loss — a drop candidate.

7. CONFLICT CARDS capture genuine resource or routing INTERACTIONS. First, an
   exclusion: a candidate that could not apply because another candidate deleted
   its target ("nullified") is NOT a conflict. Record that only in its
   no_sim_signal verdict, naming the remover; never emit a conflict card for it
   and never put a nullified pair in "avoid". Three types:
   - contention: a shared or newly merged resource pool makes an activity queue —
     waiting z up, duration z flat. SELF-contention counts: if a candidate's own
     pooled activity now queues against the rest of the process, emit a card with
     that candidate in "between" and that activity as "at". This is the primary
     conflict to surface. If SEVERAL candidates' footprints contain the contended
     activity, list each you can tie to it — do not silently pick one.
   - displacement: one activity's waiting drops sharply while another's rises,
     and the case-level improvement is smaller than the sum of the local gains.
     Requires a real down/up pair; never use it for a nullified relationship.
   - frequency_coupling: candidate B changed routing (footprint gateways_dropped
     > 0 or paths_renormalized > 0) so a DIFFERENT, STILL-EXISTING activity now
     occurs materially more or less often. Emit ONLY IF: the affected activity has
     status "changed" with real numeric n.base AND n.redes (never a REMOVED or NEW
     activity — a deletion is already covered by that candidate's own verdict);
     the two n values differ by more than a few percent (> 5%); and it is not B's
     own target. Cite both n numbers. A candidate with gateways_dropped = 0 AND
     paths_renormalized = 0 must not appear. If no still-existing activity's n
     moved materially, emit NO frequency_coupling card.

8. FIELD DISCIPLINE (these are checked):
   - "at" must be a concrete activity name from significant_deltas, never a
     placeholder like "routing_edits".
   - "magnitude_h" is a size in HOURS or null. It is NOT a pool size, a count, or
     a z. If you do not have an hours figure, use null.
   - "z" is the z of the change at "at", or null.
   - "avoid" holds pairs of TWO DIFFERENT candidate ids that should not be
     combined again. Never pair a candidate with itself; use [] if none.
   - every candidate in "between" must have footprint evidence tying it to "at".

9. A hypothesis proven by a confirmation simulation is "measured" — say so, and
   cite both runs. Everything else stays "attributed" or "hypothesized".

10. NEVER DIRECT THE REMOVAL OF AN `exogenous_only` CANDIDATE. Its verdict says
    the simulator cannot represent its mechanism, not that it does nothing, and
    the plan you are shaping is the REDESIGN THAT GETS DELIVERED — dropping it
    discards a benefit the log measured, purely because the simulator could not
    score it.
    The tempting argument is that it "does not need a simulation slot". That is
    true and it is backwards: KEEPING it does not need one either. A structural
    no-op leaves the design byte-identical, so APPLY and SIMULATE behave exactly
    as if it were absent. There is no cost to weigh against the log's figure.
    A run has already lost a candidate this way: the Critic gave it
    `exogenous_only`, quoted its own log saving in the note, and directed a drop
    in the same breath. If you judge it `exogenous_only`, it stays.
"""

_CONCLUDE_SCHEMA = """\
Return JSON with EXACTLY these keys:
{
  "per_candidate": [{"candidate_id": "...", "verdict": "helped|neutral|hurt|covered|contended|no_sim_signal|exogenous_only",
                     "evidence_ref": "activity.<name> or footprint.<cid>",
                     "confidence": "measured|attributed|hypothesized",
                     "dimensions": {"waiting": "better|worse|flat|shared|n/a",
                                    "duration": "better|worse|flat|shared|n/a",
                                    "resources": "better|worse|flat|shared|n/a",
                                    "throughput": "better|worse|flat|shared|n/a"},
                     "note": "one sentence tied to a z, a footprint, or a confirmation run"}],
  "conflicts": [{"type": "contention|displacement|frequency_coupling",
                 "between": ["<cid>"], "at": "<activity name>",
                 "magnitude_h": <hours or null>, "z": <number or null>,
                 "confidence": "measured|attributed|hypothesized", "note": "..."}],
  "plan_vs_baseline": {"verdict": "better|worse|mixed|unchanged",
                       "better_on": ["<measure>"], "worse_on": ["<measure>"]},
  "landed_as_expected": true or false,
  "surprises": "significant changes attributed to no candidate, or empty string",
  "recommendation": "continue|stop|refine",
  "directive": {"intent": "drop|add|swap|reorder|keep", "targets": ["<cid>"],
                "reason": "...", "avoid": [["<cid>", "<other cid>"]]} or null
}
"plan_vs_baseline" is the WHOLE plan against the BASELINE, read from the summary's
directional measures (median, mean, waiting, service — utilisation has no
direction): "better" = significantly better on at least one and worse on none;
"worse" = the reverse; "mixed" = both; "unchanged" = neither. Code computes the
same reading from the same numbers and will ask you to reconcile if they differ.
THE DIRECTIVE MUST BE EXECUTABLE. The Proposer's entire action space is WHICH
candidates run and in WHAT ORDER. It cannot tune a candidate, change its target,
or alter a handler — every handler's params are {}. So:
  drop     remove these from the plan
  add      bring these in (they may be candidates not in the current plan)
  swap     drop one and add another
  reorder  same candidates, different application order — a real lever, because a
           candidate applied earlier can keep its target from being consumed.
           ⚠️ IT IS ALSO THE ONLY DIRECTIVE THAT CAN DESTROY A MEASURED BENEFIT
           WITHOUT SAYING SO. Order decides not just what survives but what can be
           BUILT. Exception candidates compete for one divergence point: applying
           one first consumes a branch and makes the next one's location unique,
           so the reverse order can leave it ambiguous and unbuildable. A run has
           already lost its second-largest contributor this way, to a reorder
           directive whose stated purpose was to KEEP it: the swap made the
           survivor's location ambiguous and it then failed to build.
           So: if two candidates of the SAME structural heuristic (Exception,
           Activity Elimination, Activity Composition) are in the plan and one
           already carries a significant measured benefit, do NOT reorder them
           merely to sharpen attribution. Unresolved attribution costs you a
           question you cannot answer; a failed build costs you the benefit itself.
  keep     the plan stands; these candidates earn their place
Do NOT ask for a parameter change, a threshold, or a "tuned" version of anything:
there is no mechanism for it, and the Proposer will be forced to improvise. If a
candidate helps but causes contention you cannot tune away, your options are to
KEEP it (state the side effect), REORDER it, or DROP it — say which.

Emit one per_candidate entry for EVERY candidate id in the bundle's footprints,
and fill `dimensions` for every one of the four — that is the multi-dimensional
account, and a verdict without it is an opinion. Use "shared" when the only
movement on that dimension sits on an activity another applied candidate also
touches, "flat" when it moved but not significantly, "n/a" when there is nothing
to measure (the candidate was never applied).

USE THE SCORECARD. `bundle.scorecard.candidates` already holds, for every APPLIED
candidate, its measured effect on four columns — waiting, duration, resource
utilisation, and hours-per-case throughput. Base each verdict on all four:

Read the row's `net` line first — it is the summary the four columns add up to.
`net` counts SOLE-OWNER measures only: anything on an activity another applied
candidate also touches has already been excluded, because it cannot be assigned.

THE FOUR COLUMNS RANK EQUALLY, AND YOUR VERDICT MUST SAY WHICH ONES MOVED.
  waiting    per activity — queueing the case sits through
  duration   per activity — the work itself
  resources  per resource — load on the people who run those activities
  throughput hours a case spends at this candidate's activities
Never collapse them into one number and never let a single empty column decide.
A candidate can be `helped` on duration and `contended` on waiting at the same
time; say both. A run has already produced large, highly significant movement on
the activity-level measures while the case-level KPI barely moved and round after
round tied. A report keyed on one measure would have said nothing worked.

Your job is to report what the redesign DID, not to decide whether it was worth
it. A candidate that changes waiting on one
activity, or the load on a set of resources, and nothing else, HAS an effect —
say so plainly. Do not downgrade a finding because the case-level KPI did not
move: hours at an activity decompose the MEAN case time while the loop's KPI is
the MEDIAN, and activities overlap (activity time is ~3x the case span on these
logs), so a real local effect routinely does not show up in the case number.

WHAT THE SIMULATOR CANNOT SEE — `exogenous_saving`, on candidates that have it.
The mined model has no overnight wait, no handover delay, no batching: a redesign
that removes one of those scores ~0 in every column above, and a report built only
on the columns would call it useless. This line carries what the LOG measured that
the model is blind to (hours per occurrence, and which field it came from).

It is an ESTIMATE, from evidence, not a simulation result. So:
  · REPORT it whenever it is present — a candidate whose whole benefit is
    exogenous deserves to be seen, not silently dropped;
  · NEVER add it to a measured column or use it to rank candidates against each
    other. Its magnitude comes from the detector, so ranking on it ranks the
    detector's inputs;
  · when it carries DISAGREES_WITH_MEASUREMENT, say BOTH in your note. A
    candidate can measure a significant case-time INCREASE while its log evidence
    claims a large exogenous saving per occurrence; injecting that saving into
    the baseline and re-simulating can reverse the sign outright —
    the injection buries the harm and flips its sign. Your verdict follows the
    MEASURED column; the estimate goes in the note.

THE RECONCILIATION — `scorecard.accounting`. READ IT BEFORE THE ROWS.
  The four columns measure ACTIVITY HOURS. The loop ranks on CASE SPAN. On a
  baseline with fitted extraneous-delay timers these are nearly independent:
  most of a case's span is time with nothing running.
  · `case_span_not_visible_to_any_column` — how much of the span change no
    column can show. When it dwarfs `activity_hours_h_per_case`, the rows below
    are not the story, however tidy they look.
  · `applied_candidates_that_changed_routing_or_waiting` — who edited a gateway,
    a loop probability or a timer, and by how much. These are the candidates that
    can move the span while every column stays balanced.
  · `median_and_mean_disagree_in_sign` — THE STRONGEST SIGNAL ON THIS PAGE when
    it is true. The mean and the median of case span moving in OPPOSITE
    directions cannot be caused by an activity getting faster or slower; that
    moves both the same way. It means the case MIX changed — cases were routed
    onto different paths. Read `shape_note`, then look ONLY at the routing
    candidates: nothing else in the plan is capable of it.
  · `activity_hours_unexplained_share` with `contested_activities` — the OTHER
    way a large effect goes unowned, and it needs no routing change at all. The
    sole-owner rule gives an activity that several applied candidates touch to
    NONE of them, so their rows all read `covered` while the activity itself
    moved by hours. `contested_activities` names the activity, the hours, and
    the exact candidates competing for it. When the unexplained share is large,
    those candidates are the shortlist — not the whole plan.
    ⚠️ A run has already had the MAJORITY of its plan's activity hours
    unassigned, all on ONE activity that three candidates touched, while service
    time rose steeply. Every one of the three read `covered`.
  A candidate reading `covered` / "nothing attributable to it alone" is NOT
  thereby harmless. If the span moved a lot, the invisible part is large, and
  that candidate changed routing or waiting, then it is the leading suspect and
  a confirmation simulation is exactly the instrument for settling it — spend one.
  ⚠️ This has cost a whole run: a candidate that conditioned one gateway was
  judged `covered` in three consecutive rounds, and the plan carrying it measured
  several times WORSE than baseline — while the largest number any row could show
  was around one hour.

TWO EXTRA LINES YOU MUST READ.
  · `scorecard.unattributed` — change on activities NO applied candidate touches:
    queueing spillover the plan caused. When only ONE candidate was applied it is
    charged to it (`caused_by`, and a `spillover` line on that candidate's row),
    because by elimination it can be nobody else. With several applied it belongs
    to the plan and to no single candidate — do not assign it by intuition.
  · `structural_noop` on a candidate's row: the handler reported success and
    changed NOTHING in the model. Do not reach for `covered`, and do not spend
    a confirmation to find out what removing it does — the answer is already
    known to be "nothing". Which verdict it takes depends on whether it also
    carries an `exogenous_saving`; both cases are stated below. Report the one
    the evidence fits.
  · a candidate's `net.h_per_case_including_spillover`, when present, is what it
    did to the process as a whole rather than only to its own activities. An
    isolated candidate has been measured improving its OWN activity while
    spilling a THIRD of that gain back onto a dozen others.

ALL FOUR COLUMNS ARE SIGNIFICANCE-GATED. `throughput_h_per_case` counts as
evidence ONLY when `throughput_significant` is true. When it is false the number
is a coin flip — it carries `throughput_h_per_case_sd` so you can see how wide —
and you must NOT let it move a verdict in either direction. This is not a
formality: one candidate has been called `helped` in one round and `hurt` two
rounds later on two numbers that both sit inside a single noise band.

  · `structural_noop` AND an `exogenous_saving` -> "exogenous_only". This is a
                                    REAL benefit the simulator has no way to
                                    express, not an absent one: the mined model
                                    has no batching, no overnight gap, no
                                    handover delay, so the handler's edit leaves
                                    it byte-identical and all four columns read
                                    empty. Quote the log figure in the note. Do
                                    NOT call it `neutral` — that says "measured
                                    and did nothing", and it is the reason every
                                    candidate of this kind was discarded. Never
                                    spend a confirmation on it either: removing it
                                    provably changes nothing in the model.
  · `structural_noop` with NO exogenous saving -> "neutral". Nothing was edited
                                    and nothing is claimed; that is genuinely
                                    empty.
  · net empty AND `unattributable_activities` empty -> "neutral". Doing nothing
                                    IS a finding, and this is the only shape that
                                    establishes it.
  · net empty BUT `unattributable_activities` non-empty -> NOT "neutral". Nothing
                                    was established either way. Try to settle it
                                    from `claimants` (below); if you cannot,
                                    "covered".
  · net better (mostly negative) -> "helped", even when it also causes contention
                                    somewhere: report the side effect in the note,
                                    do not let it flip the verdict.
  · net worse                    -> "hurt".
  · waiting up, duration flat, and the harm is queueing -> "contended".

"contended" describes a MECHANISM, not a net loss. A candidate whose own
activities improved by z=-13 while pushing a queue downstream is "helped" with a
noted side effect — calling it "contended" reads as a reason to drop the one thing
that worked.

Activities listed under `unattributable_activities` are touched by more than one
applied candidate: their change belongs to neither, so say so rather than crediting
or blaming one of them. Before falling back to "covered", try to settle it from
`claimants`, which lists what each claimant's handler STRUCTURALLY did. The
heuristics leave different marks, and the shape of the change usually names the
one responsible: waiting up with duration flat is queueing, and only a candidate
that merged a resource pool (`pool_merged`) can cause queueing — one that
encapsulated a subprocess or deleted a task cannot. When the mechanism settles it,
say so in the note and give the real verdict; "covered" is for when it does not.
Output ONLY the JSON object."""

def budget_facts(confirm_left: int) -> list:
    """What CONCLUDE is told about the confirmation budget — FACTS, no advice."""

    if confirm_left > 0:
        return [f"CONFIRMATION BUDGET after this round: {confirm_left} left for the run."]
    return [
        "CONFIRMATION BUDGET after this round: 0. No confirmation simulation will run "
        "for the rest of this run, so a hypothesis still open now can only be settled "
        "by measuring a DIFFERENT plan in a later round — one your directive changes "
        "(add / drop / swap / reorder). A `keep` on an unchanged plan rebuilds the "
        "same design and is measured as no new information.",
    ]

def round_budget_facts(round_no, no_gain, max_rounds, patience,
                       design_repeat_of=None) -> list:
    """What CONCLUDE is told about the ROUND budget - FACTS, no advice."""

    slack = max(0, int(patience) - int(no_gain))
    cap_left = max(0, int(max_rounds) - int(round_no))
    lines = [
        "ROUND BUDGET: the stopping rule ends the run after {p} consecutive "
        "rounds that fail to beat the best plan measured so far. That counter "
        "stands at {n}, so {s} such round(s) of slack remain (round cap: {c} "
        "more round(s)).".format(p=patience, n=no_gain, s=slack, c=cap_left),
    ]
    if design_repeat_of is not None:
        lines.append(
            "The design measured THIS round is byte-identical to round {r}'s, so "
            "no simulation ran and this round produced no new measurement - and "
            "it still counted toward the stopping rule above. A directive that "
            "leaves the plan unchanged repeats that.".format(r=design_repeat_of))
    if slack == 0:
        lines.append(
            "With 0 rounds of slack, this run ends unless THIS round beat the "
            "incumbent. A hypothesis still open now will not be settled by a "
            "later round, because there is no later round.")
    return lines

def plan_fact_verdict(plan_fact):
    """Code's reading of the plan against the baseline, in the schema's words."""

    if not plan_fact:
        return None
    better, worse = plan_fact.get("better_on") or [], plan_fact.get("worse_on") or []
    if better and not worse:
        return "better"
    if worse and not better:
        return "worse"
    if better and worse:
        return "mixed"
    return "unchanged"

def plan_reading_check(judged: dict, plan_fact) -> dict:
    """Does the Critic's stated plan-level reading match what was measured?"""

    measured = plan_fact_verdict(plan_fact)
    stated = str(((judged or {}).get("plan_vs_baseline") or {}).get("verdict") or "").lower() or None
    return {"stated": stated, "measured": measured,
            "consistent": (measured is None) or (stated == measured),
            "better_on": (plan_fact or {}).get("better_on") or [],
            "worse_on": (plan_fact or {}).get("worse_on") or []}

def plan_fact_lines(plan_fact, *, prior_stated=None,
                    baseline_hours=None) -> list:
    """The measured plan-vs-baseline reading as prompt lines — facts only."""
    v = plan_fact_verdict(plan_fact)
    if v is None:
        return []
    better, worse = plan_fact.get("better_on") or [], plan_fact.get("worse_on") or []
    lines = [f"PLAN vs BASELINE (measured, |z|>3 on the directional measures): {v} — "
             f"significantly better on {better or 'none'}, significantly worse on "
             f"{worse or 'none'}."]

    if baseline_hours:
        b = baseline_hours.get("base_median_h")
        r_ = baseline_hours.get("plan_median_h")
        if b is not None and r_ is not None:
            gap = r_ - b
            lines.append(
                "On the convergence measure (median case span) the baseline is "
                "{b:.1f} h per case and this plan is {r:.1f} h ({g:+.1f} h per "
                "case). {verdict}".format(
                    b=b, r=r_, g=gap,
                    verdict=("The plan LOSES to leaving the process unchanged on "
                             "that measure." if gap > 0 else
                             "The plan beats leaving the process unchanged on "
                             "that measure.")))
    if prior_stated:
        lines.append(f"Your previous answer gave plan_vs_baseline.verdict = {prior_stated!r}, "
                     f"which does not match the measurement above. Answer again, "
                     f"consistently with it; the directive is still yours to decide.")
    return lines

def confirmation_fact_lines(conf_fact) -> list:
    """The confirmation as a PLAN-LEVEL counterfactual — facts only."""

    if not conf_fact:
        return []
    removed = conf_fact.get("removed") or []
    added = conf_fact.get("added") or []
    what = (f"removing {removed}" if removed and not added
            else f"adding {added}" if added and not removed
            else f"removing {removed} and adding {added}")
    v = conf_fact.get("verdict") or {}
    better, worse = v.get("better_on") or [], v.get("worse_on") or []
    rows = conf_fact.get("per_measure") or []
    lines = [f"CONFIRMATION AS A PLAN-LEVEL COUNTERFACTUAL — the confirmation plan is the "
             f"round's plan {what}. Against the BASELINE, full plan -> confirmation plan:"]
    for r in rows:
        lines.append(f"    {r['measure']}: {r['full_pct']:+.1f}% -> {r['conf_pct']:+.1f}%")
    lines.append(f"  Confirmation vs full plan (|z|>3, directional measures): significantly "
                 f"better on {better or 'none'}, significantly worse on {worse or 'none'}.")
    lines.append(f"  That difference is the measured effect of {what} from this plan.")
    return lines

def blind_spot_facts(rows) -> list:
    """State the candidates no automatic mechanism can remove, with what is known."""

    if not rows:
        return []
    out = ["NO AUTOMATIC MECHANISM IN THIS LOOP CAN REMOVE THESE CANDIDATES.",
           "  The subset check only drops a candidate whose OWN attributable throughput",
           "  is positive and significant; the isolation probes only ever queue",
           "  candidates whose handler reported a routing change; `keeps_hurt` needs a",
           "  `hurt` verdict with a measured cost. These satisfy none of the three, so",
           "  each one leaves the plan only if YOUR directive drops it or a confirmation",
           "  you request leaves it out."]
    for r in rows:
        h = r.get("own_throughput_h_per_case")
        bits = ["own attributable throughput %s h per case" % (
            "not established (0.0)" if h in (0, 0.0) else
            ("%+.4f" % float(h) if h is not None else "none"))]
        if r.get("own_z") is not None:
            bits.append("z=%s" % r["own_z"])
        bits.append("not significant" if not r.get("own_significant") else "significant")
        if r.get("pool_resources_behind_its_targets"):
            bits.append("it put %s resources behind its targets"
                        % r["pool_resources_behind_its_targets"])
        if r.get("activities"):
            bits.append("activities %s" % (r["activities"][:4]))
        out.append("  %s (%s): %s." % (r.get("candidate_id"), r.get("heuristic"),
                                       "; ".join(bits)))
    return out

def conclude_message(round_no: int, bundle: dict, hypotheses: list,
                     confirmation: "dict | None", *, confirm_left: int = 2,
                     plan_fact=None, prior_stated=None, conf_fact=None,
                     round_fact=None, baseline_hours=None, schema_bad=None,
                     follow_fact=None, blind_fact=None) -> str:
    """The user message CONCLUDE sends. Split out so the prompt can be tested."""
    parts = [
        f"Round {round_no}. Evidence slice:",
        json.dumps(bundle, ensure_ascii=False, indent=2, default=str),
        "", "Hypotheses after investigation:",
        json.dumps([h if isinstance(h, dict) else asdict(h) for h in hypotheses],
                   ensure_ascii=False, indent=2),
    ]
    if confirmation:
        parts += [
            "", "CONFIRMATION SIMULATION RESULT — a second run with a modified candidate set.",
            f"plan simulated: {confirmation.get('plan')}",
            f"hypothesis under test: {confirmation.get('hypothesis_id')}",
            "Its evidence slice (compare against the one above; conclusions it proves "
            "are MEASURED):",
            json.dumps(confirmation.get("bundle"), ensure_ascii=False, indent=2, default=str),
        ]
        parts += [""] + confirmation_fact_lines(conf_fact)
    parts += [""] + plan_fact_lines(plan_fact, prior_stated=prior_stated,
                                    baseline_hours=baseline_hours)
    parts += [""] + budget_facts(confirm_left)
    if round_fact:
        parts += [""] + round_budget_facts(**round_fact)

    _ff = directive_facts(follow_fact)
    if _ff:
        parts += [""] + _ff
    _bf = blind_spot_facts(blind_fact)
    if _bf:
        parts += [""] + _bf
    parts += ["", _CONCLUDE_SCHEMA]
    if schema_bad:
        parts += [""] + schema_retry_lines(schema_bad)
    return "\n".join(parts)

def per_candidate_violations(out: dict) -> list:
    """Rows of the Critic's answer that do not satisfy the schema it was given."""

    bad = []
    for i, e in enumerate((out or {}).get("per_candidate") or []):
        if not isinstance(e, dict):
            bad.append({"index": i, "problem": "not an object"})
            continue
        cid = e.get("candidate_id")
        v = e.get("verdict")
        if not cid:
            bad.append({"index": i, "problem": "no candidate_id",
                        "keys": sorted(e)[:8]})
        elif not v:
            bad.append({"index": i, "candidate_id": cid,
                        "problem": "no verdict", "keys": sorted(e)[:8]})
        elif v not in VERDICTS:
            bad.append({"index": i, "candidate_id": cid,
                        "problem": "verdict %r is not one of %s" % (v, list(VERDICTS))})
    return bad

def schema_retry_lines(bad: list) -> list:
    """What the retry says: the offending rows, and nothing about what to answer."""
    if not bad:
        return []
    rows = ", ".join(
        "%s(%s)" % (b.get("candidate_id") or "row %d" % b.get("index", -1),
                    b.get("problem")) for b in bad[:8])
    return [
        "YOUR PREVIOUS ANSWER DID NOT SATISFY THE SCHEMA. These `per_candidate` "
        "rows are incomplete: " + rows + ".",
        "Every row needs a `candidate_id` and a `verdict`, and the verdict must be "
        "one of " + "|".join(VERDICTS) + ". Answer again with those rows complete. "
        "Nothing else about your answer is being questioned, and which verdict "
        "each candidate deserves is still yours to decide.",
    ]

def directive_facts(ft: "dict | None") -> list:
    """State how far the LAST directive's plan moved, to the agent that writes"""

    if not ft or not isinstance(ft, dict):
        return []
    out = ["WHAT YOUR LAST DIRECTIVE PRODUCED (facts; the plan is the "
           "Proposer's to choose, so a departure is not an error):"]
    named = ft.get("named") or []
    out.append("  - it named %d candidate(s)%s with intent %r."
               % (len(named), (" (%s)" % ", ".join(named)) if named else "",
                  ft.get("intent") or ""))
    out.append("  - the plan went from %d candidate(s) to %d (net %+d)."
               % (ft.get("size_before") or 0, ft.get("size_after") or 0,
                  ft.get("net_change") or 0))
    un = ft.get("left_unnamed") or []
    if un:
        out.append("  - %d candidate(s) LEFT THE PLAN that the directive did not "
                   "name: %s." % (len(un), ", ".join(un)))
    ad = ft.get("added_unnamed") or []
    if ad:
        out.append("  - %d were ADDED that it did not name: %s."
                   % (len(ad), ", ".join(ad)))
    if not un and not ad:
        out.append("  - nothing moved that the directive did not name.")
    return out

def conclude(round_no: int, bundle: dict, hypotheses: list[Hypothesis],
             confirmation: dict | None, *, model: str | None = None,
             confirm_left: int = 2, plan_fact=None, conf_fact=None,
             round_fact=None, baseline_hours=None, follow_fact=None,
             blind_fact=None) -> dict:
    user = conclude_message(round_no, bundle, hypotheses, confirmation,
                            confirm_left=confirm_left, plan_fact=plan_fact,
                            conf_fact=conf_fact, round_fact=round_fact,
                            baseline_hours=baseline_hours,
                            follow_fact=follow_fact, blind_fact=blind_fact)
    out = _conclude_once(user, model)

    check = plan_reading_check(out, plan_fact)
    bad = per_candidate_violations(out)
    if (not check["consistent"]) or bad:
        user2 = conclude_message(round_no, bundle, hypotheses, confirmation,
                                 follow_fact=follow_fact, blind_fact=blind_fact,
                                 confirm_left=confirm_left, plan_fact=plan_fact,
                                 prior_stated=(None if check["consistent"]
                                               else check["stated"]),
                                 conf_fact=conf_fact, round_fact=round_fact,
                                 baseline_hours=baseline_hours,
                                 schema_bad=bad or None)
        out2 = _conclude_once(user2, model)
        still = per_candidate_violations(out2)

        if len(still) <= len(bad):
            out, bad_after = out2, still
        else:
            bad_after = bad
        out["plan_reading"] = {**plan_reading_check(out, plan_fact),
                               "first_answer": check["stated"],
                               "retried": True,
                               "retry_reason": ([] if check["consistent"] else ["plan_reading"])
                                               + (["schema"] if bad else [])}
        out["schema_retry"] = {"violations": bad, "after_retry": bad_after,
                               "retried": True}
    else:
        out["plan_reading"] = {**check, "retried": False}
        out["schema_retry"] = {"violations": [], "after_retry": [], "retried": False}
    out["per_candidate_missing"] = _uncovered(bundle, out.get("per_candidate"))
    return out

def _conclude_once(user: str, model) -> dict:
    return respond_json(
        [{"role": "system", "content": _CONCLUDE_SYSTEM}, {"role": "user", "content": user}],
        model=model, max_output_tokens=_CONCLUDE_MAX)

def _uncovered(bundle: dict, per_candidate: "list | None") -> list:
    """Applied candidates the write-up gave no row."""

    applied = {cid for cid, fp in ((bundle or {}).get("footprints") or {}).items()
               if (fp or {}).get("applied")}
    answered = {str(e.get("candidate_id")) for e in (per_candidate or [])
                if isinstance(e, dict)}
    return sorted(applied - answered)
