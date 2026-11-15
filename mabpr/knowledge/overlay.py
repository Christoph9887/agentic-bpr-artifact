"""KNOWLEDGE OVERLAY — what the measurements taught us, layered over the literature."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ACTIONS = (
    "downgrade_hard_to_weak",
    "upgrade_weak_to_hard",
    "strengthen_weak",
    "revise_impact",
    "note_only",
)
KINDS = ("hard_wrongly_marked", "weak_understated", "impact_wrong",
         "prior_confirmed", "other")

@dataclass
class Contradiction:
    """One disagreement between a prior and a measurement, with its provenance."""
    id: str
    round: int
    kind: str
    subject: list[str]
    prior_said: str
    measured_said: str
    evidence: list[str]
    action: str
    confidence: str
    note: str = ""
    accepted: bool = True
    refused_reason: str = ""

    @classmethod
    def from_dict(cls, d: dict, round_no: int, seq: int) -> "Contradiction":
        return cls(
            id=str(d.get("id") or f"C{round_no}-{seq}"),
            round=round_no,
            kind=str(d.get("kind") or "other"),
            subject=[str(x) for x in (d.get("subject") or [])],
            prior_said=str(d.get("prior_said") or ""),
            measured_said=str(d.get("measured_said") or ""),
            evidence=[str(x) for x in (d.get("evidence") or [])],
            action=str(d.get("action") or "note_only"),
            confidence=str(d.get("confidence") or "hypothesized"),
            note=str(d.get("note") or ""),
        )

def _pair_key(subject: list[str]) -> str | None:
    return "|".join(sorted(subject)) if len(subject) == 2 else None

class KnowledgeOverlay:
    """The learned layer: load, gate, append, persist, and apply."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.contradictions: list[Contradiction] = []
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.contradictions = [Contradiction(**c) for c in raw.get("contradictions", [])]

    def relation_override(self, h_a: str, h_b: str) -> dict | None:
        """The latest accepted relation change for a heuristic pair, if any."""
        key = _pair_key([h_a, h_b])
        newest = None
        for c in self.contradictions:
            if not c.accepted or _pair_key(c.subject) != key:
                continue
            if c.action == "downgrade_hard_to_weak":
                newest = ("WEAK_CONFLICT", c)
            elif c.action == "upgrade_weak_to_hard":
                newest = ("HARD_CONFLICT", c)
            elif c.action == "strengthen_weak":
                newest = ("WEAK_CONFLICT", c)
        if newest is None:
            return None
        rel, c = newest
        return {"relation_type": rel, "from_round": c.round, "contradiction_id": c.id,
                "measured_said": c.measured_said, "action": c.action}

    def impact_notes(self, heuristic: str) -> list[dict]:
        """Recorded disagreements with a heuristic's impact vector."""
        return [{"contradiction_id": c.id, "round": c.round,
                 "measured_said": c.measured_said, "action": c.action}
                for c in self.contradictions
                if c.accepted and c.action == "revise_impact" and heuristic in c.subject]

    def add(self, c: Contradiction) -> None:
        self.contradictions.append(c)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "note": ("Learned layer over foundation's HI-Graph. foundation is the "
                     "literature prior and is never edited; everything here was "
                     "produced by measurement inside the loop."),
            "contradictions": [asdict(c) for c in self.contradictions],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        return self.path

    def summary(self) -> dict:
        acc = [c for c in self.contradictions if c.accepted]
        return {"total": len(self.contradictions), "accepted": len(acc),
                "refused": len(self.contradictions) - len(acc),
                "by_action": {a: sum(1 for c in acc if c.action == a)
                              for a in ACTIONS if any(c.action == a for c in acc)}}

def overlay_gate(c: Contradiction, *, apply_outcome: dict | None = None) -> tuple[bool, str]:
    """May this contradiction change the knowledge base? Code decides, not the model."""

    if c.action not in ACTIONS:
        return False, f"unknown action {c.action!r}"
    if c.kind not in KINDS:
        return False, f"unknown kind {c.kind!r}"
    if not c.evidence:
        return False, "no evidence cited — a correction without evidence is an opinion"

    target = " ".join([*c.subject, c.prior_said]).lower()
    if "complian" in target or "合规" in target:
        return False, ("compliance has no ground truth and is not falsifiable by "
                       "measurement (DECISIONS B/C)")

    if c.action == "note_only":
        return True, "recorded without changing the prior"

    if len(c.subject) != 2 and c.action != "revise_impact":
        return False, f"a relation change needs exactly two heuristics, got {c.subject}"

    if c.action == "downgrade_hard_to_weak":
        if not apply_outcome:
            return False, "no Phase-3 outcome supplied; the HARD claim was not tested"
        if not apply_outcome.get("built"):
            return False, "Phase-3 did not build this plan, so HARD was not refuted"
        if apply_outcome.get("condition_fired") is False:
            return False, ("the relation's condition did not fire on these candidates, "
                           "so the HARD claim never applied to them")
    return True, "granted"
