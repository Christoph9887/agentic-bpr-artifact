"""KNOWLEDGE TOOLS — the HI-Graph, on demand, for the Proposer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mabpr.foundation.knowledge.phase2_hi_graph import HEURISTICS, build_hi_graph
from mabpr.evidence.footprint import _declared_targets
from mabpr.evidence.tools import CHEAP, ReadEntry
from mabpr.knowledge.overlay import KnowledgeOverlay

_ORDERING_HEURISTICS = {"Knock-out", "Resequencing"}
_SET_HEURISTICS = {"Case Assignment"}

def _reason_param() -> dict:
    return {"type": "string",
            "description": "Why you are looking this up (one short phrase). Required."}

@dataclass(frozen=True)
class _CandidateView:
    """The two attributes the HI-Graph's condition predicates read."""

    affected_activities: frozenset
    target_role: str

    @classmethod
    def of(cls, cand: dict) -> "_CandidateView":
        h = cand.get("heuristic", "")
        role = ("activity_ordering" if h in _ORDERING_HEURISTICS
                else "activity_set" if h in _SET_HEURISTICS
                else "single_activity")
        return cls(frozenset(_declared_targets(cand)), role)

class KnowledgeBox:
    """Executes the Proposer's knowledge lookups, recording every one."""

    def __init__(self, detector_results_path: str | Path,
                 overlay_path: str | Path | None = None) -> None:
        self.detector_results_path = Path(detector_results_path)

        self.overlay = KnowledgeOverlay(overlay_path) if overlay_path else None
        self.read_log: list[ReadEntry] = []
        self._cache: dict[str, Any] = {}
        self._step = 0
        self._graph = None
        self._detectors: list[dict] | None = None

    def graph(self):
        if self._graph is None:

            from mabpr.knowledge import extensions as _ext
            self._graph = _ext.build_extended_hi_graph()
        return self._graph

    def _detector_by_id(self) -> dict[str, dict]:
        if self._detectors is None:
            self._detectors = json.loads(
                self.detector_results_path.read_text(encoding="utf-8"))
        return {c["candidate_id"]: c for c in self._detectors}

    @property
    def specs(self) -> list[dict]:

        self.graph()
        from mabpr.foundation.knowledge import phase2_hi_graph as _hg
        heuristics = list(_hg.HEURISTICS)
        return [
            {
                "type": "function",
                "name": "get_heuristic_prior",
                "description": (
                    "What the literature PREDICTS a redesign heuristic does: its "
                    "Devil's Quadrangle impact vector (time / cost / quality / "
                    "flexibility), the rationale and the source. Use it to reason "
                    "about what a candidate is FOR before deciding to keep or drop "
                    "it — the measured result tells you what happened, this tells "
                    "you what was supposed to happen."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": _reason_param(),
                        "heuristic": {"type": "string", "enum": heuristics},
                    },
                    "required": ["reason", "heuristic"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_pair_relation",
                "description": (
                    "The prior interaction between two heuristics: HARD_CONFLICT / "
                    "WEAK_CONFLICT / SYNERGY / NEUTRAL, with the rationale, the "
                    "source, and whether the relation's structural condition "
                    "actually fires on two specific candidates. Consult it before "
                    "putting two candidates in one plan."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": _reason_param(),
                        "heuristic_a": {"type": "string", "enum": heuristics},
                        "heuristic_b": {"type": "string", "enum": heuristics},
                        "candidate_a": {"type": ["string", "null"],
                                        "description": "Optional candidate id, to also test the condition."},
                        "candidate_b": {"type": ["string", "null"],
                                        "description": "Optional candidate id, to also test the condition."},
                    },
                    "required": ["reason", "heuristic_a", "heuristic_b",
                                 "candidate_a", "candidate_b"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_candidate_evidence",
                "description": (
                    "The detector's ORIGINAL evidence for one candidate — the "
                    "inefficiency it claimed and how big. Use it to judge whether a "
                    "candidate is worth keeping in the plan at all."),
                "parameters": {
                    "type": "object",
                    "properties": {"reason": _reason_param(),
                                   "candidate_id": {"type": "string"}},
                    "required": ["reason", "candidate_id"],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: dict) -> dict:
        self._step += 1
        why = str(args.get("reason", ""))
        key = json.dumps([name, {k: v for k, v in sorted(args.items()) if k != "reason"}],
                         sort_keys=True, default=str)
        if key in self._cache:
            self.read_log.append(ReadEntry(self._step, name, args, CHEAP, why, cached=True))
            return {"cached": True, "note": "You already looked this up.", **self._cache[key]}

        try:
            result = self._dispatch(name, args)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            self.read_log.append(ReadEntry(self._step, name, args, CHEAP, why, error=msg))
            return {"error": msg}

        self._cache[key] = result
        self.read_log.append(ReadEntry(self._step, name, args, CHEAP, why))
        return result

    def _dispatch(self, name: str, args: dict) -> dict:
        if name == "get_heuristic_prior":
            return self._prior(str(args["heuristic"]))
        if name == "get_pair_relation":
            return self._relation(str(args["heuristic_a"]), str(args["heuristic_b"]),
                                  args.get("candidate_a"), args.get("candidate_b"))
        if name == "get_candidate_evidence":
            return self._evidence(str(args["candidate_id"]))
        raise KeyError(f"unknown tool {name!r}")

    def _prior(self, heuristic: str) -> dict:
        iv = self.graph().get_impact(heuristic)
        out = {"heuristic": heuristic, "impact": iv.to_dict(),
               "impact_numeric": iv.to_numeric(),
               "rationale": iv.rationale, "source": iv.source}
        notes = self.overlay.impact_notes(heuristic) if self.overlay else []
        if notes:
            out["measured_contradictions"] = notes
            out["note"] = ("Earlier rounds measured something this prior did not "
                           "predict — see measured_contradictions.")
        return out

    def _relation(self, h_a: str, h_b: str,
                  cid_a: str | None, cid_b: str | None) -> dict:
        rel = self.graph().get_relation(h_a, h_b)
        out = {"heuristic_a": h_a, "heuristic_b": h_b,
               "relation_type": rel.relation_type,
               "rationale": rel.rationale, "source": rel.source}

        ov = self.overlay.relation_override(h_a, h_b) if self.overlay else None
        if ov:
            out["literature_relation_type"] = rel.relation_type
            out["relation_type"] = ov["relation_type"]
            out["superseded_by_measurement"] = ov

        if cid_a and cid_b:
            by_id = self._detector_by_id()
            a, b = by_id.get(str(cid_a)), by_id.get(str(cid_b))
            if a is None or b is None:
                out["condition_fires"] = None
                out["condition_note"] = f"unknown candidate: {cid_a if a is None else cid_b}"
            else:
                va, vb = _CandidateView.of(a), _CandidateView.of(b)
                try:
                    out["condition_fires"] = bool(rel.applies(va, vb))
                except Exception as e:
                    out["condition_fires"] = None
                    out["condition_note"] = f"condition not evaluable: {type(e).__name__}: {e}"
                out["condition_text"] = rel.condition.text
                out["shared_activities"] = sorted(va.affected_activities
                                                  & vb.affected_activities)
        return out

    def _evidence(self, cid: str) -> dict:
        det = self._detector_by_id().get(cid)
        if det is None:
            raise KeyError(f"unknown candidate {cid!r}")
        return {"candidate_id": cid, "heuristic": det.get("heuristic"),
                "evidence": det.get("evidence"),
                "additional_context": (det.get("additional_context") or "")[:1200]}

    def audit(self) -> dict:
        return {
            "total_reads": len(self.read_log),
            "cached_hits": sum(1 for r in self.read_log if r.cached),
            "errors": sum(1 for r in self.read_log if r.error),
            "log": [r.__dict__ for r in self.read_log],
        }

    def grounding(self) -> list[str]:
        """Which priors the proposal actually rests on — the audit trail that makes"""

        out = []
        for r in self.read_log:
            if r.error:
                continue
            a = {k: v for k, v in r.args.items() if k != "reason" and v is not None}
            out.append(f"knowledge.{r.tool}({json.dumps(a, sort_keys=True, default=str)})")
        return out
