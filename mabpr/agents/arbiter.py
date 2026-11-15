"""ARBITER — the gatekeeper, and the one who corrects the knowledge."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mabpr import config
from mabpr.evidence.footprint import _declared_targets
from mabpr.knowledge.overlay import Contradiction, KnowledgeOverlay, overlay_gate
from mabpr.knowledge.tools import KnowledgeBox, _reason_param
from mabpr.llm.client import function_calls, respond, respond_json

MAX_STEPS = 8
_STEP_MAX = 8000
_OUT_MAX = 6000

DELETING_HEURISTICS = {"Activity Elimination", "Exception"}

REMOVING_HEURISTICS = {"Activity Elimination"}

def _consumed_by(cand: dict) -> set[str]:
    """Activities this candidate makes unavailable to a candidate applied later."""

    h = cand.get("heuristic")
    if h == "Exception":
        excl = (cand.get("variant_diff") or {}).get("exception_exclusive_activities")
        return {str(a) for a in (excl or []) if a}
    if h in REMOVING_HEURISTICS:
        return set(_declared_targets(cand))
    return set()

def target_survival(candidate: dict, ordering: list[str],
                    by_id: dict[str, dict]) -> dict:
    """Which of a candidate's declared targets survive, applied in THIS order."""

    targets = set(_declared_targets(candidate))
    cid = candidate.get("candidate_id")
    deleted_before: set[str] = set()
    earlier_removers: set[str] = set()
    for other in ordering:
        if other == cid:
            break
        oc = by_id.get(other)
        if oc:
            eats = _consumed_by(oc)
            deleted_before |= eats
            if eats & targets:
                earlier_removers.add(other)

    removed = sorted(targets & deleted_before)
    surviving = sorted(targets - deleted_before)
    return {
        "candidate_id": cid,
        "heuristic": candidate.get("heuristic"),
        "targets": sorted(targets),
        "removed_by_earlier_candidate": removed,

        "removed_by_earlier": sorted(earlier_removers),
        "removed_by": sorted({o for o in ordering
                              if o != cid and _consumed_by(by_id.get(o) or {}) & targets}),
        "surviving": surviving,
        "fraction_surviving": (len(surviving) / len(targets)) if targets else None,
        "signal": ("no_targets" if not targets
                   else "all_targets_removed" if not surviving
                   else "some_targets_removed" if removed
                   else "intact"),
    }

def activities_deleted_by(plan: list[str], by_id: dict[str, dict]) -> dict[str, str]:
    """{activity: candidate_id} for every activity this plan would DELETE — the"""

    out: dict[str, str] = {}
    for cid in plan:
        c = by_id.get(cid)
        if c and c.get("heuristic") in REMOVING_HEURISTICS:
            for a in _declared_targets(c):
                out.setdefault(a, cid)
    return out

class ArbiterBox(KnowledgeBox):
    """The knowledge tools, plus the two things only the Arbiter needs: what a"""

    def __init__(self, detector_results_path: str | Path,
                 overlay_path: str | Path | None = None,
                 phase3_dir: str | Path | None = None) -> None:
        super().__init__(detector_results_path, overlay_path)
        self.phase3_dir = Path(phase3_dir) if phase3_dir else None

    @property
    def specs(self) -> list[dict]:
        return super().specs + [
            {
                "type": "function",
                "name": "check_target_survival",
                "description": (
                    "For one candidate, applied in a given ORDER: which of its target "
                    "activities survive and which were already deleted by an earlier "
                    "candidate. Use it to tell a candidate that does nothing at all "
                    "from one that merely does less — and to test whether reordering "
                    "would revive it."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": _reason_param(),
                        "candidate_id": {"type": "string"},
                        "ordering": {"type": "array", "items": {"type": "string"},
                                     "description": "The plan in application order."},
                    },
                    "required": ["reason", "candidate_id", "ordering"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_apply_outcome",
                "description": (
                    "What Phase-3 actually did with the applied plan: which candidates "
                    "built, which were skipped and why. This is the GROUND TRUTH for a "
                    "HARD_CONFLICT claim — if a pair the prior called impossible was "
                    "built, the prior was wrong."),
                "parameters": {
                    "type": "object",
                    "properties": {"reason": _reason_param()},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        ]

    def _dispatch(self, name: str, args: dict) -> dict:
        if name == "check_target_survival":
            by_id = self._detector_by_id()
            cid = str(args["candidate_id"])
            if cid not in by_id:
                raise KeyError(f"unknown candidate {cid!r}")
            return target_survival(by_id[cid],
                                   [str(x) for x in (args.get("ordering") or [])], by_id)
        if name == "get_apply_outcome":
            return self.apply_outcome()
        return super()._dispatch(name, args)

    def apply_outcome(self) -> dict:
        """Phase-3's verdict on the applied plan, read off phase3_log.json."""
        if self.phase3_dir is None:
            return {"error": "no applied plan yet (nothing has been built this round)"}
        log = self.phase3_dir / "phase3_log.json"
        if not log.exists():
            return {"error": f"no phase3_log.json under {self.phase3_dir}"}
        data = json.loads(log.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        return {
            "built": True,
            "candidates": [
                {"candidate_id": e["candidate_id"], "heuristic": e.get("heuristic"),
                 "status": e.get("status"),
                 "applied": bool((e.get("adapt_report") or {}).get("applied")),
                 "skip_reason": (e.get("adapt_report") or {}).get("reason")}
                for e in entries
            ],
        }

_DOMAIN_KNOWN = """   Be explicit that this is a prior judgement about {domain} and
   that you may be wrong — that honesty is part of the output, not a hedge."""
_DOMAIN_UNKNOWN = """   ⚠️ YOU ARE NOT TOLD WHAT INDUSTRY OR JURISDICTION THIS PROCESS BELONGS TO, AND
   YOU MUST NOT ASSUME ONE. Judge from the activity names alone. Do not import
   obligations from a sector the names merely resemble; if a name is only
   meaningful under an assumed industry, say so and let the ruling rest on the
   name, not on the sector.
   Be explicit that this is a prior judgement and that you may be wrong — that
   honesty is part of the output, not a hedge."""

def gate_system(domain: str = "") -> str:
    """The GATE system prompt for one dataset. Empty domain -> assume nothing."""
    line = (_DOMAIN_KNOWN.format(domain=domain.strip()) if (domain or "").strip()
            else _DOMAIN_UNKNOWN)
    return _GATE_SYSTEM.replace("__DOMAIN_LINE__", line)

_GATE_SYSTEM = """\
You are the Arbiter, at the GATE. A plan has been proposed; you decide whether it
may run, and whether any candidate in it should be pruned first. You do NOT judge
whether a candidate is a good idea — measurement does that, and the Critic reads it.
You judge only two things.

1. COMPLIANCE — may these activities be deleted at all?
   Judge from your own domain knowledge of the process (activity names + context).
   You are given NO evidence for this and must not ask for any: no measurement can
   tell you whether a step is legally required, so none is offered.
   THE ASYMMETRY IS LAW:
     · "must not delete"  -> a HARD constraint. Say it whenever a step plausibly
       carries a legal, regulatory, audit or consumer-protection obligation.
     · "may delete"       -> only means you found no reason to forbid it. It is a
       WEAK statement and must never be phrased as an endorsement.
   Better to hold back a good candidate than to delete a legally required activity.
__DOMAIN_LINE__

2. TARGET REMOVAL — is a candidate still doing anything?
   Candidates apply IN ORDER, so one that deletes an activity can leave a later
   candidate aiming at something that no longer exists. Use check_target_survival
   to get the facts, then judge SEMANTICS, which the counts alone cannot settle:
   Map the situation to the `basis` value you must return:
     · every target gone      -> the candidate can no longer do ANYTHING (applying
       it would change nothing). Prune it with basis "nullified", and say which
       candidate removed its target.
     · some targets gone      -> it can still do part of its job: basis
       "degraded". Usually merely weakened, so KEEP it — unless what survives no
       longer serves the candidate's purpose, and then say which target that is.
     · nothing gone           -> basis "ok".
     · barred on prior knowledge rather than on survival -> basis "compliance".
       That is the only basis that is never rolled back.
   Reordering can revive a candidate: you may test an alternative order with the
   tool and recommend one. You recommend; you do not rewrite the plan yourself.

Investigate before ruling. State each ruling as its own item with a reason."""

_GATE_SCHEMA = """\
Return JSON with EXACTLY these keys:
{
  "compliance": [
    {"activity": "<name>", "deleted_by": "<candidate id>",
     "verdict": "must_not_delete" | "no_objection",
     "confidence": "measured|attributed|hypothesized",
     "reason": "why — and for must_not_delete, which obligation you think applies"}
  ],
  "candidate_rulings": [
    {"candidate_id": "...", "ruling": "keep" | "prune",
     "basis": "compliance" | "nullified" | "degraded" | "ok",
     "reason": "one sentence citing the survival facts or the obligation"}
  ],
  "reorder_suggestion": ["<candidate ids>"] or null,
  "note": "anything the next agent should know, or empty"
}
Rule on EVERY activity the plan would delete and EVERY candidate in the plan.
Output ONLY the JSON object."""

@dataclass
class GateRuling:
    plan_in: list[str] = field(default_factory=list)
    plan_out: list[str] = field(default_factory=list)
    compliance: list[dict] = field(default_factory=list)
    rulings: list[dict] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    reorder_suggestion: list[str] | None = None

    survival: dict = field(default_factory=dict)
    note: str = ""
    rollback_note: str = ""
    halted: bool = False
    grounding: list[str] = field(default_factory=list)

def resolve_rulings(ordering: list[str], rulings: list[dict],
                    deleters: set[str] | None = None,
                    survival: dict | None = None,
                    never_builds: set | None = None,
                    compliance: list | None = None,
                    ) -> tuple[list[str], list[str], list[str], str]:
    """Turn the Arbiter's per-candidate rulings into the plan that will actually run."""

    prune = [r["candidate_id"] for r in rulings
             if r.get("ruling") == "prune" and r.get("candidate_id") in ordering]
    claimed = [c for c in prune
               if any(r.get("candidate_id") == c and r.get("basis") == "compliance"
                      for r in rulings)]

    blocked = [c for c in claimed if deleters is None or c in deleters]

    verdicts = [c for c in (compliance or [])
                if isinstance(c, dict) and c.get("verdict") == "must_not_delete"]
    said_must_not = {c.get("deleted_by") for c in verdicts}
    escaped = [c for c in ordering
               if c in said_must_not and c not in blocked
               and (deleters is None or c in deleters)]
    if escaped:
        blocked = [c for c in ordering if c in set(blocked) | set(escaped)]

        prune = [c for c in ordering if c in set(prune) | set(escaped)]

    demoted = [c for c in claimed if c not in blocked]
    soft = [c for c in prune if c not in blocked]

    revived = []
    if survival:
        cannot_run = set(blocked) | set(never_builds or ())
        for cid in list(soft):
            fact = survival.get(cid) or {}

            removers = set(fact.get("removed_by_earlier")
                           if "removed_by_earlier" in fact else (fact.get("removed_by") or []))

            if removers and (removers <= cannot_run):
                revived.append(cid)
        if revived:
            soft = [c for c in soft if c not in revived]
            prune = [c for c in prune if c not in revived]

    plan_out = [c for c in ordering if c not in prune]
    notes = []
    if revived:

        notes.append(
            f"{revived} were pruned as having no surviving target, but every "
            f"candidate that would have removed those targets will not run "
            f"(compliance-blocked, or refused by the build step), so the targets "
            f"survive and they were given back")
    if demoted:
        notes.append(f"compliance was claimed for {demoted}, which delete nothing — "
                     f"kept as an ordinary prune, not an irrevocable block")
    if not plan_out and soft:
        plan_out = [c for c in ordering if c not in blocked]
        soft = []
        notes.append("the gate pruned every candidate; the revocable prunes were "
                     "rolled back so the round has something to run" +
                     (f", the {len(blocked)} compliance block(s) were not"
                      if blocked else " (there were no compliance blocks)"))
    rollback = "; ".join(notes)
    pruned = sorted(set(blocked) | set(soft), key=ordering.index)
    if escaped:
        rollback = ((rollback + "  ") if rollback else "") + (
            "%s: the compliance verdict said must_not_delete while the "
            "candidate ruling said keep; the restrictive reading was enforced "
            "(DECISIONS B)." % ", ".join(escaped))
    return plan_out, pruned, sorted(set(blocked), key=ordering.index), rollback

def gate(round_no: int, ordering: list[str], all_candidates: list[dict],
         box: ArbiterBox, *, model: str | None = None,
         never_builds: set | None = None, domain: str = "",
         max_steps: int = MAX_STEPS) -> GateRuling:
    """ⓐ Rule on a proposed plan before it is applied."""
    by_id = {c["candidate_id"]: c for c in all_candidates}
    deletions = activities_deleted_by(ordering, by_id)
    survival = {cid: target_survival(by_id[cid], ordering, by_id)
                for cid in ordering if cid in by_id}

    user = "\n".join([
        f"Round {round_no}. Rule on this proposed plan before it is applied.",
        "",
        f"PLAN (application order): {ordering}",
        "",
        "ACTIVITIES THIS PLAN WOULD DELETE (rule on every one):",
        json.dumps(deletions, ensure_ascii=False, indent=2),
        "",
        "TARGET SURVIVAL under this order (facts; the semantics are yours):",
        json.dumps(survival, ensure_ascii=False, indent=2),
        "",
        "Investigate what you need, then rule.",
    ])

    model = model or config.REASONING_MODEL
    items: list[Any] = [{"role": "system", "content": gate_system(domain)},
                        {"role": "user", "content": user}]
    _tool_loop(items, box, model, max_steps)
    items.append({"role": "user", "content": _GATE_SCHEMA})
    out = respond_json(items, model=model, max_output_tokens=_OUT_MAX)

    rulings = out.get("candidate_rulings") or []

    plan_out, pruned, blocked, rollback = resolve_rulings(
        ordering, rulings, deleters=set(deletions.values()), survival=survival,
        never_builds=never_builds, compliance=out.get("compliance") or [])

    return GateRuling(
        plan_in=list(ordering), plan_out=plan_out,
        compliance=out.get("compliance") or [], rulings=rulings,
        pruned=pruned, blocked=blocked,
        reorder_suggestion=out.get("reorder_suggestion") or None,
        survival=survival,
        note=str(out.get("note") or ""), rollback_note=rollback,

        halted=not plan_out,
        grounding=box.grounding(),
    )

_CALIBRATE_SYSTEM = """\
You are the Arbiter, CALIBRATING. The round is measured. Your job is to find where
a PRIOR disagreed with a MEASUREMENT and decide whether the knowledge base should
be corrected — this is how the system learns, and it is the only thing that edits
what later rounds are told.

The priors in play, and what can refute each:
  · HI-Graph pair relation (HARD / WEAK / SYNERGY / NEUTRAL)
      HARD claims two edits cannot build a legal model. Phase-3 settles it: if it
      BUILT them, the claim was false. Cheapest correction there is — but check
      first with get_pair_relation whether the relation's structural CONDITION even
      fired on those two candidates. If it did not, the claim was never tested and
      there is no contradiction.
  · HI-Graph impact vector (time / cost / quality / flexibility)
      Refuted when the measurement moves the opposite way — cite the z.
  · COMPLIANCE
      NOT refutable. No measurement is evidence against it; "faster without it" is
      not an argument. Never propose a correction touching compliance.

Discipline:
  · A correction needs EVIDENCE — a z, a Phase-3 outcome, a plan comparison. A
    correction without evidence will be refused by the gate, and rightly.
  · "The prior held up" is a real finding: record it as kind "prior_confirmed" with
    action "note_only".
  · Prefer "note_only" when the evidence is suggestive but thin. Changing a prior
    on weak evidence is worse than leaving it and saying why.
  · One contradiction per disagreement. Do not restate the same one twice."""

_CALIBRATE_SCHEMA = """\
Return JSON with EXACTLY these keys:
{
  "contradictions": [
    {"kind": "hard_wrongly_marked|weak_understated|impact_wrong|prior_confirmed|other",
     "subject": ["<heuristic>"] or ["<heuristic a>", "<heuristic b>"],
     "prior_said": "what the knowledge base claims",
     "measured_said": "what this round measured",
     "evidence": ["a z, a Phase-3 outcome, a plan comparison"],
     "action": "downgrade_hard_to_weak|upgrade_weak_to_hard|strengthen_weak|revise_impact|note_only",
     "confidence": "measured|attributed|hypothesized",
     "note": ""}
  ],
  "summary": "one sentence on what the knowledge base learned this round, or that it learned nothing"
}
Subjects must be heuristic NAMES (e.g. "Case Assignment"), not candidate ids.
Return an empty list if nothing genuinely disagreed. Output ONLY the JSON object."""

@dataclass
class CalibrationResult:
    contradictions: list[dict] = field(default_factory=list)
    accepted: list[dict] = field(default_factory=list)
    refused: list[dict] = field(default_factory=list)
    summary: str = ""
    overlay_path: str = ""
    overlay_summary: dict = field(default_factory=dict)
    grounding: list[str] = field(default_factory=list)

def calibrate(round_no: int, assessment: dict | None, proposal: dict | None,
              box: ArbiterBox, overlay: KnowledgeOverlay,
              *, plan_measures: dict | None = None,
              exogenous: dict | None = None,
              model: str | None = None, max_steps: int = MAX_STEPS) -> CalibrationResult:
    """ⓑ Reconcile priors against this round's measurements; write back what is earned."""
    a = (assessment or {}).get("assessment", {})
    user = "\n".join([
        f"Round {round_no}. Reconcile the priors against what was measured.",
        "",

        "WHAT THE WHOLE PLAN DID, measure by measure (each with its own z):",
        json.dumps(plan_measures or {}, ensure_ascii=False, indent=2, default=str),
        "",

        "WHAT THE SIMULATOR CANNOT REPRESENT (measured in the LOG, per candidate):",
        json.dumps(exogenous or {}, ensure_ascii=False, indent=2, default=str),
        "",
        "MEASURED VERDICTS (per candidate — `dimensions` says which of waiting / "
        "duration / resources / throughput moved):",
        json.dumps(a.get("per_candidate") or [], ensure_ascii=False, indent=2),
        "",
        "MEASURED CONFLICTS:",
        json.dumps(a.get("conflicts") or [], ensure_ascii=False, indent=2),
        "",
        "WHAT THE PROPOSER ALREADY NOTICED (its own prior-vs-measured note):",
        json.dumps((proposal or {}).get("prior_vs_measured") or "", ensure_ascii=False),
        "",
        "Look up the priors you need and check what Phase-3 actually built, then rule.",
    ])

    model = model or config.REASONING_MODEL
    items: list[Any] = [{"role": "system", "content": _CALIBRATE_SYSTEM},
                        {"role": "user", "content": user}]
    _tool_loop(items, box, model, max_steps)
    items.append({"role": "user", "content": _CALIBRATE_SCHEMA})
    out = respond_json(items, model=model, max_output_tokens=_OUT_MAX)

    outcome = box.apply_outcome()
    accepted, refused = [], []
    for i, raw in enumerate(out.get("contradictions") or [], start=1):
        c = Contradiction.from_dict(raw, round_no, i)
        ok, why = overlay_gate(c, apply_outcome=_outcome_for(c, box, outcome))
        c.accepted, c.refused_reason = ok, ("" if ok else why)
        overlay.add(c)
        (accepted if ok else refused).append(asdict(c))

    path = overlay.save()
    return CalibrationResult(
        contradictions=[asdict(c) for c in overlay.contradictions if c.round == round_no],
        accepted=accepted, refused=refused,
        summary=str(out.get("summary") or ""),
        overlay_path=str(path), overlay_summary=overlay.summary(),
        grounding=box.grounding(),
    )

def _outcome_for(c: Contradiction, box: ArbiterBox, outcome: dict) -> dict | None:
    """Evidence the gate needs for a HARD downgrade: did Phase-3 build the pair, and"""

    if c.action != "downgrade_hard_to_weak" or len(c.subject) != 2 or outcome.get("error"):
        return outcome if not outcome.get("error") else None
    applied = [e for e in outcome.get("candidates", []) if e.get("applied")]
    hs = {e["heuristic"] for e in applied}
    both_applied = all(h in hs for h in c.subject)
    fired = None
    if both_applied:
        by_id = box._detector_by_id()
        for x in applied:
            for y in applied:
                if x["heuristic"] == c.subject[0] and y["heuristic"] == c.subject[1]:
                    r = box._relation(c.subject[0], c.subject[1],
                                      x["candidate_id"], y["candidate_id"])
                    if r.get("condition_fires"):
                        fired = True
                    elif fired is None:
                        fired = False
    return {"built": bool(outcome.get("built") and both_applied), "condition_fired": fired}

def _tool_loop(items: list[Any], box: ArbiterBox, model: str, max_steps: int) -> int:
    """Let the model investigate until it stops asking. Every call is audited."""
    steps = 0
    while steps < max_steps:
        resp = respond(items, model=model, tools=box.specs, max_output_tokens=_STEP_MAX)
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
                result = box.execute(call.name, args)
            items.append({"type": "function_call_output", "call_id": call.call_id,
                          "output": json.dumps(result, ensure_ascii=False, default=str)})
        steps += 1
    return steps
