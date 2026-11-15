"""phase2_normalizer.py"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class NormalizedCandidate:
    """Unified representation of a redesign candidate for Phase 2."""

    candidate_id: str
    heuristic: str

    affected_activities: set[str]
    primary_target: Any
    target_role: str

    phase1_accepted: bool
    phase1_confidence: str
    phase1_reason: str

    raw_evidence: dict
    raw_process_fragment: dict | None
    heuristic_context: str
    additional_context: str

    evidence_summary: dict = field(default_factory=dict)

TARGET_ROLES = {
    "single_activity",
    "activity_set",
    "activity_ordering",
    "divergence_point",
    "activity_pair",
}

def _extract_activity_elimination(raw: dict) -> tuple[set[str], Any, str, dict]:
    target = raw.get("target_activity")
    if not target:
        raise ValueError(
            f"Activity Elimination candidate {raw.get('candidate_id')!r} "
            "has no target_activity"
        )
    ev = raw.get("evidence", {})
    summary = {
        "waiting_time_before": ev.get("waiting_time_before"),
        "activity_duration": ev.get("activity_duration"),
        "waiting_to_duration_ratio": ev.get("waiting_to_duration_ratio"),
        "bypass_ratio": ev.get("bypass_ratio"),
        "variant_coverage": ev.get("variant_coverage"),
        "activity_frequency": ev.get("activity_frequency"),
    }
    return ({target}, target, "single_activity", summary)

def _extract_case_based_work(raw: dict) -> tuple[set[str], Any, str, dict]:
    target = raw.get("target_activity")
    if not target:
        raise ValueError(
            f"Case-based Work candidate {raw.get('candidate_id')!r} "
            "has no target_activity"
        )
    ev = raw.get("evidence", {})
    summary = {
        "waiting_time_before": ev.get("waiting_time_before"),
        "batch_size_mean": ev.get("batch_size_mean"),
        "periodicity_strength": ev.get("periodicity_strength"),
        "arrival_rate": ev.get("arrival_rate"),
        "execution_rate": ev.get("execution_rate"),
        "arrival_execution_gap": ev.get("arrival_execution_gap"),
    }
    return ({target}, target, "single_activity", summary)

def _extract_resequencing(raw: dict) -> tuple[set[str], Any, str, dict]:
    target = raw.get("target_activity")
    if not target:
        raise ValueError(
            f"Resequencing candidate {raw.get('candidate_id')!r} "
            "has no target_activity"
        )
    ev = raw.get("evidence", {})
    summary = {
        "waiting_time_before": ev.get("waiting_time_before"),
        "activity_position_ratio": ev.get("activity_position_ratio"),
        "ordering_variance": ev.get("ordering_variance"),
        "position_inconsistency_rate": ev.get("position_inconsistency_rate"),
        "variant_coverage": ev.get("variant_coverage"),
    }
    return ({target}, target, "single_activity", summary)

def _extract_case_assignment(raw: dict) -> tuple[set[str], Any, str, dict]:
    targets = raw.get("target_activities")
    if not targets:
        raise ValueError(
            f"Case Assignment candidate {raw.get('candidate_id')!r} "
            "has no target_activities"
        )
    ev = raw.get("evidence", {})
    summary = {
        "handover_count_mean": ev.get("handover_count_mean"),
        "resource_continuity_rate": ev.get("resource_continuity_rate"),
        "waiting_time_after_handover_mean": ev.get("waiting_time_after_handover_mean"),
        "cross_activity_resource_overlap": ev.get("cross_activity_resource_overlap"),
        "cross_trained_resource_count": ev.get("cross_trained_resource_count"),
        "activity_count": len(targets),
    }
    return (set(targets), list(targets), "activity_set", summary)

def _extract_knockout(raw: dict) -> tuple[set[str], Any, str, dict]:
    targets = raw.get("target_activities")
    if not targets:
        raise ValueError(
            f"Knock-out candidate {raw.get('candidate_id')!r} "
            "has no target_activities"
        )
    ev = raw.get("evidence", {})

    current = set(ev.get("current_ordering", []))
    optimal = set(ev.get("optimal_ordering", []))
    affected = set(targets) | current | optimal
    summary = {
        "current_ordering": ev.get("current_ordering"),
        "optimal_ordering": ev.get("optimal_ordering"),
        "efficiency_gap": ev.get("efficiency_gap"),
        "efficiency_gap_ratio": ev.get("efficiency_gap_ratio"),
        "current_ordering_efficiency": ev.get("current_ordering_efficiency"),
        "optimal_ordering_efficiency": ev.get("optimal_ordering_efficiency"),
    }
    return (affected, list(targets), "activity_ordering", summary)

def _extract_exception(raw: dict) -> tuple[set[str], Any, str, dict]:
    divergence = raw.get("divergence_point")
    if not divergence:
        raise ValueError(
            f"Exception candidate {raw.get('candidate_id')!r} "
            "has no divergence_point"
        )
    variant_diff = raw.get("variant_diff", {}) or {}
    exception_variant = raw.get("exception_variant", {}) or {}
    dominant_variant = raw.get("dominant_variant", {}) or {}

    exception_exclusive = set(variant_diff.get("exception_exclusive_activities", []) or [])

    if not exception_exclusive:
        exception_exclusive = set(exception_variant.get("activity_sequence", []) or [])

    affected = {divergence} | exception_exclusive

    ev = raw.get("evidence", {})
    summary = {
        "exception_variant_frequency": exception_variant.get("frequency"),
        "dominant_variant_frequency": dominant_variant.get("frequency"),
        "variant_frequency_gap": ev.get("variant_frequency_gap"),
        "throughput_time_dominant": ev.get("throughput_time_dominant"),
        "throughput_time_exception": ev.get("throughput_time_exception"),
        "throughput_time_gap": ev.get("throughput_time_gap"),
        "exception_case_count": ev.get("exception_case_count"),
        "exception_exclusive_activities": sorted(exception_exclusive),
    }
    return (affected, divergence, "divergence_point", summary)

def _extract_parallelism(raw: dict) -> tuple[set[str], Any, str, dict]:
    """Parallelism candidate normalization."""

    targets = raw.get("target_activities")
    ev = raw.get("evidence", {})
    summary = {
        "dependency_strength_a_to_b": ev.get("dependency_strength_a_to_b"),
        "dependency_strength_b_to_a": ev.get("dependency_strength_b_to_a"),
        "waiting_time_between":       ev.get("waiting_time_between"),
        "ordering_consistency":       ev.get("ordering_consistency"),
        "resource_conflict_ratio":    ev.get("resource_conflict_ratio"),
    }

    if isinstance(targets, dict):
        a = targets.get("activity_a") or raw.get("activity_a")
        b = targets.get("activity_b") or raw.get("activity_b")
        if a and b:
            return ({a, b}, [a, b], "activity_pair", summary)

    if isinstance(targets, list) and len(targets) >= 2:
        a, b = targets[0], targets[1]
        if a and b:
            return ({a, b}, [a, b], "activity_pair", summary)

    a = raw.get("activity_a")
    b = raw.get("activity_b")
    if a and b:
        return ({a, b}, [a, b], "activity_pair", summary)

    raise ValueError(
        f"Parallelism candidate {raw.get('candidate_id')!r} "
        "has no recognizable activity pair"
    )

_EXTRACTORS = {
    "Activity Elimination": _extract_activity_elimination,
    "Case-based Work":       _extract_case_based_work,
    "Resequencing":          _extract_resequencing,
    "Case Assignment":       _extract_case_assignment,
    "Knock-out":             _extract_knockout,
    "Exception":             _extract_exception,
    "Parallelism":           _extract_parallelism,
}

def _normalize_one(raw: dict, phase1_lookup: dict[str, dict]) -> NormalizedCandidate:
    heuristic = raw.get("heuristic")
    if heuristic not in _EXTRACTORS:
        raise ValueError(
            f"Unknown heuristic {heuristic!r} for candidate "
            f"{raw.get('candidate_id')!r}. Known: {sorted(_EXTRACTORS)}"
        )

    affected, primary, role, summary = _EXTRACTORS[heuristic](raw)

    affected = {a for a in affected if a}
    if not affected:
        raise ValueError(
            f"Candidate {raw.get('candidate_id')!r} yields empty "
            "affected_activities after extraction"
        )

    if role not in TARGET_ROLES:
        raise ValueError(
            f"Extractor for {heuristic!r} produced invalid target_role "
            f"{role!r}. Allowed: {sorted(TARGET_ROLES)}"
        )

    cid = raw.get("candidate_id")
    p1 = phase1_lookup.get(cid)
    if p1 is None:
        warnings.warn(
            f"Candidate {cid!r} has no Phase 1 entry; "
            "defaulting to accepted=True, confidence=unknown.",
            stacklevel=2,
        )
        phase1_accepted = True
        phase1_confidence = "unknown"
        phase1_reason = ""
    else:
        phase1_accepted = bool(p1.get("accepted", True))
        phase1_confidence = str(p1.get("confidence", "unknown"))
        phase1_reason = str(p1.get("reason", ""))

    return NormalizedCandidate(
        candidate_id=cid,
        heuristic=heuristic,
        affected_activities=affected,
        primary_target=primary,
        target_role=role,
        phase1_accepted=phase1_accepted,
        phase1_confidence=phase1_confidence,
        phase1_reason=phase1_reason,
        raw_evidence=raw.get("evidence", {}) or {},
        raw_process_fragment=raw.get("process_fragment"),
        heuristic_context=raw.get("heuristic_context", "") or "",
        additional_context=raw.get("additional_context", "") or "",
        evidence_summary=summary,
    )

def normalize_candidates(
    detector_json_path: str | Path,
    phase1_json_path: str | Path,
    filter_by_phase1: bool = True,
) -> list[NormalizedCandidate]:
    """Load both JSON files and produce a list of NormalizedCandidate objects."""

    with open(detector_json_path, "r", encoding="utf-8") as f:
        raw_candidates = json.load(f)
    with open(phase1_json_path, "r", encoding="utf-8") as f:
        phase1_list = json.load(f)

    if not isinstance(raw_candidates, list):
        raise TypeError(
            f"Expected a JSON list in {detector_json_path}, "
            f"got {type(raw_candidates).__name__}"
        )
    if not isinstance(phase1_list, list):
        raise TypeError(
            f"Expected a JSON list in {phase1_json_path}, "
            f"got {type(phase1_list).__name__}"
        )

    phase1_lookup: dict[str, dict] = {
        p["candidate_id"]: p for p in phase1_list if "candidate_id" in p
    }

    normalized: list[NormalizedCandidate] = []
    n_filtered = 0
    for raw in raw_candidates:
        try:
            cand = _normalize_one(raw, phase1_lookup)
        except ValueError as e:
            warnings.warn(
                f"Skipping candidate {raw.get('candidate_id')!r}: {e}",
                stacklevel=2,
            )
            continue

        if filter_by_phase1 and not cand.phase1_accepted:
            n_filtered += 1
            continue

        normalized.append(cand)

    if filter_by_phase1 and n_filtered > 0:
        print(
            f"  Normalizer: filtered out {n_filtered} Phase 1-rejected "
            f"candidate(s); {len(normalized)} retained."
        )

    return normalized

def print_summary(normalized: list[NormalizedCandidate]) -> None:
    """Print a human-readable summary of the normalized candidate list."""

    from collections import Counter, defaultdict

    print("=" * 70)
    print(f"NORMALIZER SUMMARY  ({len(normalized)} candidates)")
    print("=" * 70)

    by_heuristic = Counter(c.heuristic for c in normalized)
    print("\nBy heuristic:")
    for h, n in by_heuristic.most_common():
        print(f"  {h:<25} {n}")

    accepted = sum(1 for c in normalized if c.phase1_accepted)
    print(f"\nPhase 1 accepted : {accepted} / {len(normalized)}")
    print(f"Phase 1 rejected : {len(normalized) - accepted} / {len(normalized)}")

    by_role = Counter(c.target_role for c in normalized)
    print("\nBy target_role:")
    for r, n in by_role.most_common():
        print(f"  {r:<25} {n}")

    activity_hits: dict[str, list[str]] = defaultdict(list)
    for c in normalized:
        for a in c.affected_activities:
            activity_hits[a].append(c.candidate_id)

    multi_hit = {a: cids for a, cids in activity_hits.items() if len(cids) > 1}
    print(f"\nActivities touched by >1 candidate (potential conflicts): "
          f"{len(multi_hit)}")
    for a, cids in sorted(multi_hit.items(), key=lambda kv: -len(kv[1])):
        print(f"  {a!r}")
        print(f"    → {', '.join(cids)}")

    print("\nPer-candidate detail:")
    for c in normalized:
        acc_mark = "✓" if c.phase1_accepted else "✗"
        targets_str = (
            f"{len(c.affected_activities)} activities"
            if len(c.affected_activities) > 2
            else ", ".join(sorted(c.affected_activities))
        )
        print(
            f"  [{acc_mark}] {c.candidate_id:<8} "
            f"{c.heuristic:<22} "
            f"role={c.target_role:<20} "
            f"affected={{{targets_str}}}"
        )

def save_candidates_json(
    normalized: list[NormalizedCandidate],
    output_path: str | Path,
) -> None:
    """Serialize the normalized candidate list to a JSON file."""

    payload = []
    for c in normalized:
        payload.append({
            "candidate_id":         c.candidate_id,
            "heuristic":            c.heuristic,
            "affected_activities":  sorted(c.affected_activities),
            "primary_target":       c.primary_target,
            "target_role":          c.target_role,
            "phase1_accepted":      c.phase1_accepted,
            "phase1_confidence":    c.phase1_confidence,
            "phase1_reason":        c.phase1_reason,
            "evidence_summary":     c.evidence_summary,
            "raw_evidence":         c.raw_evidence,
            "raw_process_fragment": c.raw_process_fragment,
            "heuristic_context":    c.heuristic_context,
            "additional_context":   c.additional_context,
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(payload)} normalized candidates to {out}")

def main() -> None:
    detector_path = "detector_results.json"
    phase1_path = "phase1_results.json"

    normalized = normalize_candidates(detector_path, phase1_path)
    print_summary(normalized)

if __name__ == "__main__":
    main()
