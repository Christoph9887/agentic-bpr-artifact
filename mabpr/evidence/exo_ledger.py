"""EXO LEDGER — who may claim which exogenous hours, and who may not."""

from __future__ import annotations

from mabpr.evidence.exogenous import RECIPES, exogenous_saving

def _claimed_activities(cand: dict) -> set:
    """The activities whose exogenous hours this candidate is claiming."""

    out = set()
    ta = cand.get("target_activity")
    if isinstance(ta, str) and ta:
        out.add(ta)
    for k in ("target_activities", "contact_activities", "rework_activities"):
        v = cand.get(k) or (cand.get("evidence") or {}).get(k)
        if isinstance(v, list):
            out |= {str(x) for x in v if isinstance(x, str)}
    return out

def build_ledger(candidates: list, *, occurrences_per_case: dict | None = None,
                 baseline_case_hours: float | None = None) -> dict:
    """{candidate_id: entry} + `contested`, for every candidate that injects."""

    occ = occurrences_per_case or {}
    claims, saving = {}, {}
    for c in candidates or []:
        cid = str(c.get("candidate_id") or "")
        if not cid or (c.get("heuristic") or "") not in RECIPES:
            continue
        s = exogenous_saving(c, occ.get(cid), baseline_case_hours)
        if not s:
            continue
        claims[cid] = _claimed_activities(c)
        saving[cid] = s

    owners: dict = {}
    for cid, acts in claims.items():
        for a in acts:
            owners.setdefault(a, set()).add(cid)
    contested = {a: sorted(who) for a, who in owners.items() if len(who) > 1}

    out = {}
    for cid, acts in claims.items():
        mine = sorted(a for a in acts if len(owners[a]) == 1)
        theirs = sorted(a for a in acts if len(owners[a]) > 1)
        s = saving[cid]
        per_case = s.get("hours_per_case_estimate")

        share = (len(mine) / len(acts)) if acts else 0.0
        out[cid] = {
            "heuristic": claims and next(
                (c.get("heuristic") for c in candidates
                 if str(c.get("candidate_id")) == cid), ""),
            "field": s.get("field"),
            "hours_per_occurrence": s.get("hours_per_occurrence"),
            "activities_claimed": sorted(acts),
            "activities_sole_owned": mine,
            "activities_contested": theirs,
            "contested_with": sorted({o for a in theirs for o in owners[a]} - {cid}),
            "hours_per_case_total": per_case,
            "hours_per_case_own": (round(per_case * share, 4)
                                   if isinstance(per_case, (int, float)) else None),
            "note": s.get("note"),
        }
        if theirs and not mine:
            out[cid]["not_established"] = (
                f"every activity this candidate claims exogenous hours on is also "
                f"claimed by {out[cid]['contested_with']}. The hours are NOT zero — "
                f"they are unattributable, and splitting them would be an invention. "
                f"Settle it from the mechanism (which candidate actually removes the "
                f"wait) before crediting either.")
        if s.get("EXCEEDS_WHOLE_CASE"):
            out[cid]["EXCEEDS_WHOLE_CASE"] = s["EXCEEDS_WHOLE_CASE"]
    return {"per_candidate": out, "contested_activities": contested,
            "injecting_heuristics": sorted(RECIPES)}
