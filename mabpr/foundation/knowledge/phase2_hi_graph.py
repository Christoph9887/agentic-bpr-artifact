"""phase2_hi_graph.py"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

HEURISTICS: tuple[str, ...] = (
    "Activity Elimination",
    "Case Assignment",
    "Case-based Work",
    "Exception",
    "Knock-out",
    "Parallelism",
    "Resequencing",
)

DIMENSIONS: tuple[str, ...] = ("time", "cost", "quality", "flexibility")

IMPACT_LABELS = (
    "strong_positive",
    "weak_positive",
    "neutral",
    "weak_negative",
    "strong_negative",
)

IMPACT_NUMERIC: dict[str, int] = {
    "strong_positive":  +2,
    "weak_positive":    +1,
    "neutral":           0,
    "weak_negative":    -1,
    "strong_negative":  -2,
}

def _validate_impact_label(label: str) -> None:
    if label not in IMPACT_NUMERIC:
        raise ValueError(
            f"Invalid impact label {label!r}. Must be one of {IMPACT_LABELS}"
        )

RELATION_TYPES = (
    "HARD_CONFLICT",
    "WEAK_CONFLICT",
    "SYNERGY",
    "NEUTRAL",
)

def _pred_activity_overlap(min_overlap: int = 1) -> Callable:
    """Returns a predicate: candidates share at least `min_overlap` activities."""
    def pred(a, b) -> bool:
        return len(a.affected_activities & b.affected_activities) >= min_overlap
    pred.__name__ = f"activity_overlap(>={min_overlap})"
    return pred

def _pred_ordering_touches_activity() -> Callable:
    """Knock-out × (single-activity heuristic): conflict iff the single"""

    def pred(a, b) -> bool:
        if a.target_role == "activity_ordering" and b.target_role == "single_activity":
            ordering_cand, single_cand = a, b
        elif b.target_role == "activity_ordering" and a.target_role == "single_activity":
            ordering_cand, single_cand = b, a
        else:
            return False
        return bool(ordering_cand.affected_activities & single_cand.affected_activities)
    pred.__name__ = "ordering_touches_activity"
    return pred

@dataclass
class ConditionEncoding:
    text: str
    predicate: Callable[[Any, Any], bool]

def _cond_share_activity() -> ConditionEncoding:
    return ConditionEncoding(
        text=(
            "applies when the two candidates share at least one "
            "affected activity"
        ),
        predicate=_pred_activity_overlap(min_overlap=1),
    )

def _cond_ordering_touches() -> ConditionEncoding:
    return ConditionEncoding(
        text=(
            "applies when the single-activity candidate targets an activity "
            "that appears in the other candidate's ordering sequence"
        ),
        predicate=_pred_ordering_touches_activity(),
    )

@dataclass
class TypeRelation:
    """One cell in the heuristic-pair relation matrix."""
    heuristic_a: str
    heuristic_b: str
    relation_type: str
    condition: ConditionEncoding
    rationale: str
    source: str

    def __post_init__(self) -> None:
        if self.relation_type not in RELATION_TYPES:
            raise ValueError(
                f"Invalid relation_type {self.relation_type!r}. "
                f"Must be one of {RELATION_TYPES}"
            )
        if self.heuristic_a not in HEURISTICS:
            raise ValueError(f"Unknown heuristic: {self.heuristic_a!r}")
        if self.heuristic_b not in HEURISTICS:
            raise ValueError(f"Unknown heuristic: {self.heuristic_b!r}")

    def applies(self, cand_a, cand_b) -> bool:
        """Machine check: does this relation trigger on these two candidates?"""
        return self.condition.predicate(cand_a, cand_b)

@dataclass
class ImpactVector:
    """Devil's Quadrangle impact assessment for a single heuristic."""
    heuristic: str
    time: str
    cost: str
    quality: str
    flexibility: str
    rationale: str
    source: str

    def __post_init__(self) -> None:
        if self.heuristic not in HEURISTICS:
            raise ValueError(f"Unknown heuristic: {self.heuristic!r}")
        for d in DIMENSIONS:
            _validate_impact_label(getattr(self, d))

    def to_dict(self) -> dict[str, str]:
        return {d: getattr(self, d) for d in DIMENSIONS}

    def to_numeric(self) -> dict[str, int]:
        return {d: IMPACT_NUMERIC[getattr(self, d)] for d in DIMENSIONS}

@dataclass
class HIGraph:
    """The Heuristic Interaction Graph (Layers 1 + 2)."""

    type_relations: dict[tuple[str, str], TypeRelation] = field(default_factory=dict)

    impact_vectors: dict[str, ImpactVector] = field(default_factory=dict)

    def get_relation(self, h_a: str, h_b: str) -> TypeRelation:
        """Return the TypeRelation between two heuristics.  Order is normalized"""

        key = tuple(sorted((h_a, h_b)))
        if key not in self.type_relations:
            raise KeyError(
                f"No relation defined between {h_a!r} and {h_b!r}. "
                f"Available keys: {sorted(self.type_relations.keys())}"
            )
        return self.type_relations[key]

    def get_impact(self, heuristic: str) -> ImpactVector:
        if heuristic not in self.impact_vectors:
            raise KeyError(f"No impact vector for {heuristic!r}")
        return self.impact_vectors[heuristic]

    def validate_completeness(self) -> list[str]:
        """Check that the graph is complete: all heuristic pairs have a"""

        missing: list[str] = []
        for i, h_i in enumerate(HEURISTICS):
            if h_i not in self.impact_vectors:
                missing.append(f"missing impact_vector: {h_i!r}")
            for j in range(i, len(HEURISTICS)):
                h_j = HEURISTICS[j]
                key = tuple(sorted((h_i, h_j)))
                if key not in self.type_relations:
                    missing.append(f"missing type_relation: {key}")
        return missing

def build_hi_graph() -> HIGraph:
    """Construct and return the populated HI-Graph."""

    graph = HIGraph()

    graph.impact_vectors["Activity Elimination"] = ImpactVector(
        heuristic="Activity Elimination",
        time="strong_positive",
        cost="strong_positive",
        quality="neutral",
        flexibility="weak_negative",
        rationale=(
            "Removing a non-value-adding step shortens throughput and lowers "
            "operational cost; flexibility is reduced because one option is "
            "no longer available."
        ),
        source="Reijers & Mansar 2005, Table 1 (Task elimination)",
    )

    graph.impact_vectors["Case Assignment"] = ImpactVector(
        heuristic="Case Assignment",
        time="weak_positive",
        cost="neutral",
        quality="weak_positive",
        flexibility="weak_negative",
        rationale=(
            "Handling one case by one resource reduces handover overhead and "
            "improves case familiarity, at the cost of resource flexibility."
        ),
        source="Reijers & Mansar 2005, Table 1 (Order assignment)",
    )

    graph.impact_vectors["Case-based Work"] = ImpactVector(
        heuristic="Case-based Work",
        time="strong_positive",
        cost="weak_negative",
        quality="neutral",
        flexibility="neutral",
        rationale=(
            "Switching from batch to case-by-case removes queueing delays but "
            "loses batching economies of scale."
        ),
        source="Reijers & Mansar 2005, Table 1 (Order-based work)",
    )

    graph.impact_vectors["Exception"] = ImpactVector(
        heuristic="Exception",
        time="weak_positive",
        cost="neutral",
        quality="weak_positive",
        flexibility="weak_negative",
        rationale=(
            "Isolating exceptional cases accelerates the dominant path and "
            "protects its quality, at the cost of less flexibility on the "
            "exception branch."
        ),
        source="Reijers & Mansar 2005, Table 1 (Exception)",
    )

    graph.impact_vectors["Knock-out"] = ImpactVector(
        heuristic="Knock-out",

        time="weak_positive",
        cost="strong_positive",
        quality="neutral",
        flexibility="neutral",
        rationale=(
            "Reordering checks so that high-rejection, low-effort checks come "
            "first avoids wasted downstream work, reducing cost strongly and "
            "time slightly."
        ),
        source="Reijers & Mansar 2005, Table 1 (Knock-out); reordering interpretation",
    )

    graph.impact_vectors["Parallelism"] = ImpactVector(
        heuristic="Parallelism",
        time="strong_positive",
        cost="weak_negative",
        quality="weak_negative",
        flexibility="weak_negative",
        rationale=(
            "Concurrent execution shortens throughput but introduces "
            "synchronization overhead and coordination complexity."
        ),
        source="Reijers & Mansar 2005, Table 1 (Parallelism)",
    )

    graph.impact_vectors["Resequencing"] = ImpactVector(
        heuristic="Resequencing",
        time="weak_positive",
        cost="weak_positive",
        quality="neutral",
        flexibility="neutral",
        rationale=(
            "Postponing non-critical activities avoids wasted work on cases "
            "that terminate early; time and cost improve weakly, since only "
            "ordering changes."
        ),
        source="Reijers & Mansar 2005, Table 1 (Resequencing)",
    )

    def add(h_a: str, h_b: str, rel: str, cond: ConditionEncoding,
            rationale: str, source: str) -> None:
        key = tuple(sorted((h_a, h_b)))
        graph.type_relations[key] = TypeRelation(
            heuristic_a=key[0],
            heuristic_b=key[1],
            relation_type=rel,
            condition=cond,
            rationale=rationale,
            source=source,
        )

    add(
        h_a="Activity Elimination", h_b="Parallelism",
        rel="HARD_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "An activity cannot simultaneously be removed and parallelized "
            "with another activity; the two transformations are mutually "
            "exclusive on the shared target."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Activity Elimination", h_b="Resequencing",
        rel="HARD_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "An activity marked for removal cannot also be moved to a new "
            "position; the elimination pre-empts any reordering."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Activity Elimination", h_b="Knock-out",
        rel="HARD_CONFLICT", cond=_cond_ordering_touches(),
        rationale=(
            "A knock-out reordering assumes a fixed set of check activities; "
            "removing one of those checks invalidates the reordering."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Parallelism", h_b="Case Assignment",
        rel="HARD_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "Consolidating activities to a single resource is incompatible "
            "with executing those same activities in parallel; a single "
            "resource cannot physically parallelize."
        ),
        source="Structural-compatibility analysis; "
               "Reijers & Mansar 2005, Section 4.3",
    )

    add(
        h_a="Knock-out", h_b="Resequencing",
        rel="HARD_CONFLICT", cond=_cond_ordering_touches(),
        rationale=(
            "Resequencing an activity out of the knock-out sequence would "
            "destroy the ordering that knock-out has optimized."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Activity Elimination", h_b="Case Assignment",
        rel="WEAK_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "If an activity slated for elimination is also part of a resource "
            "consolidation set, consolidating it becomes pointless; choose "
            "one intent."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Activity Elimination", h_b="Case-based Work",
        rel="WEAK_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "Removing a batched activity makes the batching discussion moot; "
            "if both are proposed on the same activity, elimination dominates."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Activity Elimination", h_b="Exception",
        rel="WEAK_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "When an activity targeted for elimination overlaps with an "
            "exception branch, the two transformations interact: the "
            "elimination may become unnecessary or the exception isolation "
            "may fold the activity into a separate branch first."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Parallelism", h_b="Resequencing",
        rel="WEAK_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "Moving an activity's position and parallelizing it with another "
            "both reshape the local flow; the two transformations compete "
            "for the same structural slot."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Parallelism", h_b="Knock-out",
        rel="WEAK_CONFLICT", cond=_cond_ordering_touches(),
        rationale=(
            "Parallelizing a knock-out check undermines its purpose: the "
            "knock-out is meant to short-circuit downstream work, which "
            "parallel branches may already start. Reijers & Mansar explicitly "
            "warn about this interaction."
        ),
        source="Reijers & Mansar 2005, Section 4.2 (Parallelism trade-offs)",
    )

    add(
        h_a="Parallelism", h_b="Exception",
        rel="WEAK_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "Parallelizing activities that overlap an exception branch "
            "creates synchronization complications across the main and "
            "exception paths; the two changes need to be coordinated."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Knock-out", h_b="Exception",
        rel="WEAK_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "When a knock-out check belongs to an activity set that is also "
            "part of an exception branch, isolating the exception may remove "
            "the knock-out's rationale on the main path."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Resequencing", h_b="Case Assignment",
        rel="WEAK_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "If a resource-consolidated activity is moved to a different "
            "position, the consolidated resource assignment may no longer "
            "be contiguous with its sibling activities; both changes should "
            "be coordinated."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Resequencing", h_b="Case-based Work",
        rel="WEAK_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "A batched activity moved to a new position may enter a context "
            "where batching is no longer possible; the order of application "
            "matters."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Resequencing", h_b="Exception",
        rel="WEAK_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "Moving an activity whose role is redefined by exception "
            "isolation may place it in an inconsistent context; coordinate "
            "the two changes."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Case Assignment", h_b="Case-based Work",
        rel="WEAK_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "Both modify the resource/execution behaviour of an activity. "
            "Case-by-case processing combined with single-resource assignment "
            "can overwhelm the resource; joint application needs capacity "
            "checks."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Case Assignment", h_b="Exception",
        rel="WEAK_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "Resource consolidation over a set that includes exception-branch "
            "activities entangles main-path staffing with exception-handling "
            "staffing; the two assignments should be reconciled."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Case-based Work", h_b="Exception",
        rel="WEAK_CONFLICT", cond=_cond_share_activity(),
        rationale=(
            "Case-by-case reprocessing of an activity that also becomes part "
            "of an exception branch duplicates coordination concerns across "
            "main and exception paths."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Parallelism", h_b="Case-based Work",
        rel="NEUTRAL", cond=_cond_share_activity(),
        rationale=(
            "Parallelism changes structural ordering; Case-based Work changes "
            "batching behaviour. They address independent process aspects."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Knock-out", h_b="Case Assignment",
        rel="NEUTRAL", cond=_cond_share_activity(),
        rationale=(
            "Knock-out reordering addresses the sequence of checks; case "
            "assignment addresses who performs them. They are orthogonal "
            "unless one subsumes the other's activities."
        ),
        source="Structural-compatibility analysis",
    )

    add(
        h_a="Knock-out", h_b="Case-based Work",
        rel="NEUTRAL", cond=_cond_share_activity(),
        rationale=(
            "Knock-out addresses check ordering; Case-based Work addresses "
            "batching. Independent in most scenarios."
        ),
        source="Structural-compatibility analysis",
    )

    for h in HEURISTICS:
        add(
            h_a=h, h_b=h,
            rel="NEUTRAL", cond=_cond_share_activity(),
            rationale=(
                f"Two {h!r} candidates on different targets are independent. "
                "Same-target overlap is unlikely given detector deduplication."
            ),
            source="Structural-compatibility analysis",
        )

    missing = graph.validate_completeness()
    if missing:
        raise RuntimeError(
            "HI-Graph is incomplete. Missing entries:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    return graph

CONSTRAINT_VALUES = ("must_not_degrade", "must_improve", "no_constraint")

@dataclass
class Goal:
    """Runtime user preference over the Devil's Quadrangle dimensions."""

    weights: dict[str, float]
    constraints: dict[str, str]
    description: str = ""

    def __post_init__(self) -> None:
        for d in DIMENSIONS:
            self.weights.setdefault(d, 0.0)
            self.constraints.setdefault(d, "no_constraint")

        for d, w in self.weights.items():
            if d not in DIMENSIONS:
                raise ValueError(f"Unknown dimension in weights: {d!r}")
            if w < 0:
                raise ValueError(
                    f"Weight for {d!r} must be non-negative, got {w}"
                )
        for d, c in self.constraints.items():
            if d not in DIMENSIONS:
                raise ValueError(f"Unknown dimension in constraints: {d!r}")
            if c not in CONSTRAINT_VALUES:
                raise ValueError(
                    f"Invalid constraint for {d!r}: {c!r}. "
                    f"Must be one of {CONSTRAINT_VALUES}"
                )

        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("At least one weight must be positive")

    def normalized_weights(self) -> dict[str, float]:
        """Return weights normalized to sum to 1.0."""
        total = sum(self.weights.values())
        return {d: self.weights[d] / total for d in DIMENSIONS}

    @classmethod
    def balanced(cls) -> "Goal":
        """Default goal: equal weight across all four dimensions."""
        return cls(
            weights={d: 0.25 for d in DIMENSIONS},
            constraints={d: "no_constraint" for d in DIMENSIONS},
            description=(
                "Balanced optimization across time, cost, quality, and "
                "flexibility with no hard constraints."
            ),
        )

    @classmethod
    def simple(cls, priority_dim: str) -> "Goal":
        """Simple single-priority goal.  The priority dimension gets weight 0.7;"""

        if priority_dim not in DIMENSIONS:
            raise ValueError(
                f"priority_dim must be one of {DIMENSIONS}, got {priority_dim!r}"
            )
        w = {d: 0.1 for d in DIMENSIONS}
        w[priority_dim] = 0.7
        return cls(
            weights=w,
            constraints={d: "no_constraint" for d in DIMENSIONS},
            description=f"Primary optimization target: {priority_dim}.",
        )

    @classmethod
    def custom(
        cls,
        weights: dict[str, float],
        constraints: dict[str, str] | None = None,
        description: str = "",
    ) -> "Goal":
        return cls(
            weights=dict(weights),
            constraints=dict(constraints or {}),
            description=description,
        )

@dataclass
class GoalEvaluator:
    """Combines an HI-Graph impact vector with a Goal to score candidates"""

    hi: HIGraph
    goal: Goal

    def score_candidate(self, cand) -> float:
        """Weighted sum of impact (numeric) dotted with normalized goal weights."""

        impact = self.hi.get_impact(cand.heuristic).to_numeric()
        norm_w = self.goal.normalized_weights()
        return sum(norm_w[d] * impact[d] for d in DIMENSIONS)

    def violates_constraints(self, cand) -> list[str]:
        """Return list of constraint descriptions this candidate violates."""

        impact = self.hi.get_impact(cand.heuristic).to_dict()
        violations: list[str] = []
        for d in DIMENSIONS:
            constraint = self.goal.constraints[d]
            lvl = impact[d]
            if constraint == "must_not_degrade":
                if lvl in ("weak_negative", "strong_negative"):
                    violations.append(
                        f"{d}: must_not_degrade (impact is {lvl})"
                    )
            elif constraint == "must_improve":
                if lvl not in ("weak_positive", "strong_positive"):
                    violations.append(
                        f"{d}: must_improve (impact is {lvl})"
                    )
        return violations

    def rank_candidates(self, cands) -> list[tuple[Any, float, list[str]]]:
        """Return candidates sorted by goal score (descending)."""

        scored = [
            (c, self.score_candidate(c), self.violates_constraints(c))
            for c in cands
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

def heuristics_in_phase1(
    phase1_path: str,
    accepted_only: bool = True,
) -> set[str]:
    """Extract the set of heuristics that appear in a Phase-1 result file."""

    import json
    import warnings
    from pathlib import Path

    path = Path(phase1_path)
    if not path.exists():
        raise FileNotFoundError(f"Phase-1 results file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list at top level of {path}, got {type(data).__name__}"
        )

    active: set[str] = set()
    unknown: set[str] = set()
    for entry in data:
        h = entry.get("heuristic")
        if h not in HEURISTICS:
            unknown.add(str(h))
            continue
        if accepted_only and not entry.get("accepted", False):
            continue
        active.add(h)

    if unknown:
        warnings.warn(
            f"heuristics_in_phase1: skipped {len(unknown)} unknown heuristic "
            f"name(s) in {path}: {sorted(unknown)}",
            RuntimeWarning,
        )

    return active

def _filter_pair(
    pair: tuple[str, str],
    active: set[str] | None,
    mode: str,
) -> bool:
    """Decide whether an unordered heuristic pair should be kept under the"""

    if active is None:
        return True
    a, b = pair
    if mode == "both":
        return a in active and b in active
    if mode == "either":
        return a in active or b in active
    raise ValueError(f"Unknown pair filter mode: {mode!r}. Use 'both' or 'either'.")

def print_hi_summary(
    graph: HIGraph,
    restrict_to: set[str] | None = None,
    pair_mode: str = "both",
) -> None:
    """Human-readable summary of the HI-Graph content."""
    active = set(HEURISTICS) if restrict_to is None else set(restrict_to)
    shown_heuristics = [h for h in HEURISTICS if h in active]

    print("=" * 70)
    print("HI-GRAPH SUMMARY  (conflict-only, no ordering)")
    if restrict_to is not None:
        print(f"  restricted to {len(shown_heuristics)}/{len(HEURISTICS)} "
              f"heuristics (pair_mode={pair_mode!r})")
        print(f"  active: {', '.join(shown_heuristics)}")
    print("=" * 70)

    print("\nImpact Vectors (Devil's Quadrangle):")
    print(f"  {'Heuristic':<22} {'time':<16} {'cost':<16} "
          f"{'quality':<16} {'flex.':<16}")
    print(f"  {'-'*22} {'-'*16} {'-'*16} {'-'*16} {'-'*16}")
    for h in shown_heuristics:
        iv = graph.impact_vectors[h]
        print(
            f"  {h:<22} "
            f"{iv.time:<16} {iv.cost:<16} "
            f"{iv.quality:<16} {iv.flexibility:<16}"
        )

    print("\nType Relations (off-diagonal, filtered):")
    from collections import defaultdict
    by_type: dict[str, list[TypeRelation]] = defaultdict(list)
    kept = 0
    for (a, b), rel in graph.type_relations.items():
        if a == b:
            continue
        if not _filter_pair((a, b), restrict_to, pair_mode):
            continue
        by_type[rel.relation_type].append(rel)
        kept += 1

    for rel_type in RELATION_TYPES:
        rels = by_type.get(rel_type, [])
        if not rels:
            continue
        print(f"\n  [{rel_type}] ({len(rels)} pairs)")
        for rel in rels:
            print(f"    {rel.heuristic_a} × {rel.heuristic_b}")
            print(f"      condition: {rel.condition.text}")

    total_all = sum(1 for (a, b) in graph.type_relations if a != b)
    print(f"\nRelations kept: {kept} / {total_all} off-diagonal "
          f"(reflexive excluded)")

def demo_goal_scoring(
    graph: HIGraph,
    restrict_to: set[str] | None = None,
) -> None:
    """Illustrate goal scoring for each heuristic under several goals."""
    active = set(HEURISTICS) if restrict_to is None else set(restrict_to)
    shown_heuristics = [h for h in HEURISTICS if h in active]

    print("\n" + "=" * 70)
    header = "GOAL-SCORING DEMO"
    if restrict_to is not None:
        header += f"  ({len(shown_heuristics)}/{len(HEURISTICS)} heuristics)"
    else:
        header += "  (all seven heuristics)"
    print(header)
    print("=" * 70)

    goals = {
        "balanced":         Goal.balanced(),
        "optimize_time":    Goal.simple("time"),
        "optimize_cost":    Goal.simple("cost"),
        "optimize_quality": Goal.simple("quality"),
    }

    class _MockCand:
        def __init__(self, h: str) -> None:
            self.heuristic = h

    mocks = [_MockCand(h) for h in shown_heuristics]

    for name, goal in goals.items():
        print(f"\n── Goal: {name} ──────────────────────────────────")
        ev = GoalEvaluator(graph, goal)
        ranked = ev.rank_candidates(mocks)
        print(f"  {'Heuristic':<22} {'score':>8}   violated_constraints")
        print(f"  {'-'*22} {'-'*8}   {'-'*30}")
        for cand, score, viols in ranked:
            v_str = "; ".join(viols) if viols else "(none)"
            print(f"  {cand.heuristic:<22} {score:>+8.3f}   {v_str}")

def main() -> None:
    import sys
    from pathlib import Path

    graph = build_hi_graph()

    script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    candidates = [
        script_dir / "phase1_results.json",
        Path.cwd() / "phase1_results.json",
    ]
    phase1_path = next((p for p in candidates if p.exists()), None)

    if phase1_path is None:
        print("[info] No phase1_results.json found — showing full HI-Graph.\n")
        active: set[str] | None = None
    else:
        active = heuristics_in_phase1(phase1_path, accepted_only=True)
        print(f"[info] Loaded phase1 active heuristics from {phase1_path}")
        print(f"[info] accepted_only=True → {len(active)} heuristic(s): "
              f"{sorted(active)}\n")
        if not active:
            print("[warn] No accepted heuristics in phase1 — showing full "
                  "HI-Graph as fallback.\n")
            active = None

    print_hi_summary(graph, restrict_to=active, pair_mode="both")
    demo_goal_scoring(graph, restrict_to=active)

if __name__ == "__main__":
    main()
