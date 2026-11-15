"""CASE TYPES — route a case by what it IS, not by a coin flip."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

MIN_GROUP_CASES = 100
MIN_SEPARATION_Z = 3.0
MIN_SHARE_GAP = 0.05

MAX_ROUTING_MISMATCH = 0.25
CASE_ID_COL = "case:concept:name"

@dataclass
class CaseTypesEvidence:
    """One case TYPE, and every gateway whose branch the log says it decides."""

    attribute: str
    attribute_kind: str
    rule_label: str
    comparison: str
    value: str
    cases_in_group: int
    cases_out_group: int
    routes_conditioned: int
    activities_routed: list
    strongest_activity: str
    strongest_share_in: float
    strongest_share_out: float
    strongest_share_gap: float
    separation_z: float
    min_separation_z: float

    hard_rule_overshoot_pp: float

    events_after_routed_mean: float
    service_h_after_routed_mean: float
    case_events_mean: float
    case_service_h_mean: float
    routes: list

@dataclass
class CaseTypesCandidate:
    candidate_id: str
    heuristic: str
    target_activity: str
    evidence: CaseTypesEvidence
    heuristic_context: str
    additional_context: str
    process_fragment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = asdict(self.evidence)
        return d

def _reachable_tasks(bpmn, start, limit: int = 60) -> set:
    """Task names reachable from `start` without passing through another task."""
    out, seen, q = set(), {start}, deque([start])
    flows = list(bpmn.get_flows())
    while q and len(seen) < limit:
        n = q.popleft()
        for f in flows:
            if f.get_source() is not n:
                continue
            t = f.get_target()
            if "Task" in type(t).__name__ and t.get_name():
                out.add(t.get_name())
            elif t not in seen:
                seen.add(t)
                q.append(t)
    return out

def _returns_to(bpmn, start, target, limit: int = 4000) -> bool:
    """Can control reach `target` again from `start`? (Tasks are traversed.)"""

    flows = list(bpmn.get_flows())
    seen, q, steps = {start}, deque([start]), 0
    while q and steps < limit:
        steps += 1
        n = q.popleft()
        for f in flows:
            if f.get_source() is not n:
                continue
            t = f.get_target()
            if t is target:
                return True
            if t not in seen:
                seen.add(t)
                q.append(t)
    return False

def conditionable_gateways(bpmn_path: str, params: dict,
                           report: list | None = None) -> dict:
    """{gateway_id: (path_id, task_name, other_path_id, pooled_p)} — where a rule can go."""

    import pm4py

    if not bpmn_path or not Path(bpmn_path).exists():
        return {}
    bpmn = pm4py.read_bpmn(str(bpmn_path))
    by_id = {str(n.get_id()): n for n in bpmn.get_nodes()}
    flow_by_id = {str(f.get_id()): f for f in bpmn.get_flows()}

    out = {}
    for g in params.get("gateway_branching_probabilities") or []:
        gid = str(g.get("gateway_id"))
        node = by_id.get(gid)
        probs = g.get("probabilities") or []

        def _no(why: str):
            if report is not None:
                report.append({"gateway": gid, "rejected_because": [why]})

        if node is None:
            _no("the gateway is not a node in the baseline BPMN")
            continue
        if "Exclusive" not in type(node).__name__:
            _no(f"the gateway is {type(node).__name__}, not exclusive — a rule on a "
                f"parallel or inclusive split does not decide anything, both "
                f"branches can still run")
            continue
        if len(probs) != 2:
            _no(f"the gateway has {len(probs)} paths; a case-type rule and its "
                f"complement partition the cases in two, so a wider split would "
                f"leave branches Prosimos falls back to pooled probabilities for")
            continue
        reach = []
        for p in probs:
            f = flow_by_id.get(str(p.get("path_id")))
            if f is None:
                reach.append(None)
                continue
            tgt = f.get_target()
            if "Task" in type(tgt).__name__ and tgt.get_name():
                reach.append({tgt.get_name()})
            else:
                reach.append(_reachable_tasks(bpmn, tgt))
        singles = [i for i, r in enumerate(reach) if r is not None and len(r) == 1]

        if not singles:
            sizes = [len(r) if r is not None else -1 for r in reach]
            _no(f"neither path leads to exactly one task (they reach {sizes} "
                f"activities) — a rule attached here would claim to route a whole "
                f"region of the process, far more than the log evidence about one "
                f"activity supports")
            continue

        if any(f is not None and _returns_to(bpmn, f.get_target(), node)
               for f in (flow_by_id.get(str(p.get("path_id"))) for p in probs)):
            _no(f"the gateway sits on a rework loop (control can return to it), and "
                f"a branch_rules condition is a HARD predicate: the case type that "
                f"satisfies it would take the same branch every time round and never "
                f"leave. Measured — one 3000-case run wrote an 892 MB event log")
            continue
        out[gid] = [(str(probs[i]["path_id"]), next(iter(reach[i])),
                     str(probs[1 - i]["path_id"]), float(probs[i].get("value", 0.0)))
                    for i in singles]
    return out

def _z_proportions(p1: float, n1: int, p2: float, n2: int) -> float:
    """Two-proportion z. Same rule as everywhere else: gap over its own SE."""
    p = (p1 * n1 + p2 * n2) / max(1, n1 + n2)
    se = math.sqrt(max(1e-12, p * (1 - p) * (1 / max(1, n1) + 1 / max(1, n2))))
    return (p1 - p2) / se

def candidate_splits(df: pd.DataFrame, params: dict) -> list:
    """Every (attribute, rule) split the LOG offers and the PARAMS can evaluate."""

    known = {str(c.get("name")): str(c.get("type", "")).lower()
             for c in (params.get("case_attributes") or [])}
    if CASE_ID_COL not in df.columns:
        return []
    splits = []
    for col in df.columns:
        if not col.startswith("case:") or col == CASE_ID_COL:
            continue
        kind = known.get(col)
        if kind is None:
            continue
        per_case = df.groupby(CASE_ID_COL)[col].first().dropna()
        if len(per_case) < 2 * MIN_GROUP_CASES:
            continue
        if kind.startswith("cont"):
            num = pd.to_numeric(per_case, errors="coerce").dropna()
            if len(num) < 2 * MIN_GROUP_CASES or num.nunique() < 8:
                continue
            for q in (0.25, 0.50, 0.75):
                cut = float(num.quantile(q))
                inside = set(num.index[num <= cut])
                outside = set(num.index[num > cut])
                if min(len(inside), len(outside)) < MIN_GROUP_CASES:
                    continue
                splits.append({
                    "attribute": col, "kind": "continuous",
                    "label": f"{col} <= {cut:g}", "comparison": "<=",
                    "value": f"{cut:g}", "in": inside, "out": outside})
        else:
            counts = per_case.astype(str).value_counts()
            for level, n in counts.items():
                if n < MIN_GROUP_CASES or len(per_case) - n < MIN_GROUP_CASES:
                    continue
                s = per_case.astype(str)
                splits.append({
                    "attribute": col, "kind": "discrete",
                    "label": f"{col} = {level!r}", "comparison": "=",
                    "value": str(level), "in": set(s.index[s == level]),
                    "out": set(s.index[s != level])})

    unique, complements = [], []
    for sp in splits:
        if any(sp["in"] == c for c in complements):
            continue
        complements.append(sp["out"])
        unique.append(sp)
    return unique

def detect_case_types(store, df: pd.DataFrame, *, context: dict | None = None,
                      report: list | None = None) -> list:
    """Gateways whose branch the log says is decided by a case attribute."""
    params = (context or {}).get("params") or {}
    gateways = conditionable_gateways((context or {}).get("baseline_bpmn"), params,
                                      report=report)
    if not gateways:
        return []

    splits = candidate_splits(df, params)
    if not splits:
        if report is not None:
            report.append({"gateway": "*", "rejected_because": [
                "no case attribute is both present in the log and declared in "
                "params.case_attributes, so no rule Prosimos can evaluate exists"]})
        return []

    act_col = "concept:name" if "concept:name" in df.columns else "activity"
    visited = df.groupby(CASE_ID_COL)[act_col].apply(set)
    n_cases = len(visited)
    downstream = _work_after(df, act_col)

    live = {}
    for gid, options in gateways.items():

        scored, misses = [], []
        for (path_id, task, other_path, pooled) in options:
            overall = int(visited.map(lambda a: task in a).sum()) / max(1, n_cases)
            gap = abs(pooled - overall)
            if gap > MAX_ROUTING_MISMATCH:
                misses.append((task, pooled, overall))
                continue
            scored.append((gap, path_id, task, other_path, pooled, overall))
        if not scored:
            if report is not None:
                for task, pooled, overall in misses:
                    report.append({
                        "gateway": gid, "task": task,
                        "pooled_probability": round(pooled, 4),
                        "overall_visit_share": round(overall, 4),
                        "rejected_because": [
                            f"the model routes {pooled:.0%} of arrivals down this "
                            f"branch to {task!r}, but only {overall:.0%} of cases "
                            f"ever perform it — this gateway is not what decides "
                            f"whether a case does {task!r}, so a case-type rule "
                            f"here would claim routing power it does not have"]})
            continue
        _, path_id, task, other_path, pooled, overall = min(scored)
        live[gid] = (path_id, task, other_path, pooled, overall)

    best_per_route = {}
    for gid, (path_id, task, other_path, pooled, overall) in live.items():
        seen = {c for c, acts in visited.items() if task in acts}
        for sp in splits:
            n_in, n_out = len(sp["in"]), len(sp["out"])
            if min(n_in, n_out) < MIN_GROUP_CASES:
                continue
            share_in = len(sp["in"] & seen) / n_in
            share_out = len(sp["out"] & seen) / n_out
            gap = abs(share_in - share_out)
            z = _z_proportions(share_in, n_in, share_out, n_out)
            rec = {"gateway": gid, "task": task, "split": sp["label"],
                   "share_in": round(share_in, 4), "share_out": round(share_out, 4),
                   "gap": round(gap, 4), "z": round(z, 2)}
            if gap < MIN_SHARE_GAP:
                rec["rejected_because"] = [
                    f"the two groups reach {task!r} within {gap:.1%} of each other "
                    f"({share_in:.0%} vs {share_out:.0%}) — this attribute does not "
                    f"describe two case types, it describes one"]
            elif abs(z) < MIN_SEPARATION_Z:
                rec["rejected_because"] = [
                    f"the routing gap is z={z:.2f}, inside the noise "
                    f"(|z|>{MIN_SEPARATION_Z} required)"]
            else:

                prev = best_per_route.get(gid)
                if prev is None or abs(z) > abs(prev[1]["z"]):
                    best_per_route[gid] = (sp, rec)
                continue
            if report is not None:
                report.append(rec)

    by_split = {}
    for gid, (sp, rec) in best_per_route.items():
        key = (sp["attribute"], sp["comparison"], sp["value"])
        by_split.setdefault(key, (sp, []))[1].append((gid, rec))

    kept = []
    for key, (sp, members) in sorted(
            by_split.items(),
            key=lambda kv: -max(abs(r["z"]) for _, r in kv[1][1])):
        routes = []
        for gid, rec in sorted(members, key=lambda m: -abs(m[1]["z"])):
            path_id, task, other_path, pooled, overall = live[gid]

            invert = rec["share_in"] < rec["share_out"]
            routes.append({
                "gateway_id": gid, "conditioned_path_id": path_id,
                "complement_path_id": other_path, "target_activity": task,
                "pooled_probability": round(pooled, 4),
                "comparison": (_INVERT[sp["comparison"]] if invert
                               else sp["comparison"]),
                "value": sp["value"],
                "rule_label": (f"NOT ({sp['label']})" if invert else sp["label"]),
                "share_in": round(max(rec["share_in"], rec["share_out"]), 4),
                "share_out": round(min(rec["share_in"], rec["share_out"]), 4),
                "z": round(abs(rec["z"]), 2)})
        top = routes[0]
        acts = sorted({r["target_activity"] for r in routes})
        ev = CaseTypesEvidence(
            attribute=sp["attribute"], attribute_kind=sp["kind"],
            rule_label=sp["label"], comparison=sp["comparison"], value=sp["value"],
            cases_in_group=len(sp["in"]), cases_out_group=len(sp["out"]),
            routes_conditioned=len(routes), activities_routed=acts,
            strongest_activity=top["target_activity"],
            strongest_share_in=top["share_in"], strongest_share_out=top["share_out"],
            strongest_share_gap=round(top["share_in"] - top["share_out"], 4),
            separation_z=top["z"], min_separation_z=min(r["z"] for r in routes),
            hard_rule_overshoot_pp=round(
                100.0 * ((1.0 - top["share_in"]) + top["share_out"]), 1),
            events_after_routed_mean=downstream["after"].get(
                top["target_activity"], (0.0, 0.0))[0],
            service_h_after_routed_mean=downstream["after"].get(
                top["target_activity"], (0.0, 0.0))[1],
            case_events_mean=downstream["case_events"],
            case_service_h_mean=downstream["case_service_h"],
            routes=routes)
        kept.append(CaseTypesCandidate(
            candidate_id="", heuristic="Case Types",
            target_activity=top["target_activity"], evidence=ev,
            heuristic_context=(
                "Target problem types: heterogeneous cases processed in a uniform "
                "way, so simple cases carry the effort of complex ones. Applicable "
                "when cases with clearly different recorded characteristics all "
                "follow the same path."),
            additional_context=(
                f"{sp['label']} splits the log into {len(sp['in'])} and "
                f"{len(sp['out'])} cases that are routed differently at "
                f"{len(routes)} gateway(s), reaching {len(acts)} activities: "
                f"{', '.join(repr(a) for a in acts)}. The mined model has no access "
                f"to the attribute, so each of those gateways uses one pooled "
                f"probability for every case — strongest example, "
                f"{top['rule_label']} reaches {top['target_activity']!r} "
                f"{top['share_in']:.0%} of the time against {top['share_out']:.0%} "
                f"for the rest (z={top['z']:.1f}), while the model sends "
                f"{top['pooled_probability']:.0%} of arrivals there regardless. The "
                f"redesign attaches the condition at every one of those gateways, so "
                f"the case type is routed consistently rather than at a coin flip. "
                f"⚠️ The condition is a HARD predicate, so it sends 100% of this "
                f"type to {top['target_activity']!r} and 0% of the rest, where the "
                f"log shows {top['share_in']:.0%} and {top['share_out']:.0%} — "
                f"{100.0 * ((1.0 - top['share_in']) + top['share_out']):.0f} "
                f"percentage points of standardisation beyond what the log already "
                f"does, which is the intervention rather than a reproduction of it. "
                f"⚠️ This changes WHICH cases go where, not how long anything takes: "
                f"no hours are assumed anywhere, and whether routing by type helps, "
                f"hurts or does nothing is for the simulation to measure. "
                + _direction_note(top["target_activity"], downstream)),
            process_fragment={"attribute": sp["attribute"], "routes": routes}))
    for i, c in enumerate(kept, start=1):
        c.candidate_id = f"CT-{i:03d}"
    return kept

def _work_after(df, act_col: str) -> dict:
    """Per activity, the work a case still records AFTER reaching it."""

    import pandas as pd

    d = df.sort_values([CASE_ID_COL]).copy()
    start = next((c for c in ("start_timestamp", "start_time") if c in d.columns), None)
    end = next((c for c in ("time:timestamp", "end_time", "end_timestamp")
                if c in d.columns), None)
    if start and end:
        d["_svc_h"] = ((pd.to_datetime(d[end], utc=True, errors="coerce")
                        - pd.to_datetime(d[start], utc=True, errors="coerce"))
                       .dt.total_seconds() / 3600.0).clip(lower=0).fillna(0.0)
    else:
        d["_svc_h"] = 0.0
    d["_i"] = d.groupby(CASE_ID_COL).cumcount()
    size = d.groupby(CASE_ID_COL)["_i"].transform("max") + 1
    total_h = d.groupby(CASE_ID_COL)["_svc_h"].transform("sum")
    cum_h = d.groupby(CASE_ID_COL)["_svc_h"].cumsum()
    d["_after_n"] = size - 1 - d["_i"]
    d["_after_h"] = (total_h - cum_h).clip(lower=0)

    last = d.groupby([CASE_ID_COL, act_col]).tail(1)
    after = {a: (round(float(g["_after_n"].mean()), 2),
                 round(float(g["_after_h"].mean()), 4))
             for a, g in last.groupby(act_col)}
    per_case = d.groupby(CASE_ID_COL).agg(n=("_i", "size"), h=("_svc_h", "sum"))
    return {"after": after,
            "case_events": round(float(per_case["n"].mean()), 2),
            "case_service_h": round(float(per_case["h"].mean()), 4)}

def _direction_note(activity: str, downstream: dict) -> str:
    """State whether reaching *activity* leaves work to do, in the log's own terms."""
    n, h = downstream["after"].get(activity, (None, None))
    if n is None:
        return ""
    cn, ch = downstream["case_events"], downstream["case_service_h"]
    tail = (f"⚠️ Direction of the change: in the log a case that reaches "
            f"{activity!r} still records {n:.1f} further events and {h:.3f} h of "
            f"further work, against {cn:.1f} events and {ch:.3f} h for an average "
            f"case. ")
    if n <= 1.0:
        tail += ("Reaching it effectively ENDS the case, so forcing this type "
                 "through it REMOVES downstream work rather than redistributing "
                 "it — a throughput gain here would largely mean fewer cases "
                 "doing the rest of the process, which is a business decision "
                 "and not a measurement this system can settle.")
    elif n >= cn / 2.0:
        tail += ("Most of the case still lies ahead of it, so forcing this type "
                 "through it ADDS work rather than removing it; there is no free "
                 "KPI win in this direction.")
    else:
        tail += ("Part of the case still lies ahead of it, so the rule shifts "
                 "work rather than plainly adding or removing it.")
    return tail

_INVERT = {"<=": ">", ">": "<=", "<": ">=", ">=": "<", "=": "!=", "!=": "="}

def print_candidates(cands: list) -> None:
    print(f"\nCase Types: {len(cands)} candidate(s)")
    for c in cands:
        e = c.evidence
        print(f"  {c.candidate_id}  {e.rule_label}  "
              f"({e.cases_in_group} vs {e.cases_out_group} cases)")
        print(f"      {e.routes_conditioned} gateway(s) -> "
              f"{len(e.activities_routed)} activities, z {e.min_separation_z:.1f}"
              f"..{e.separation_z:.1f}")
        for r in e.routes:
            print(f"        {r['target_activity'][:34]:36} "
                  f"{r['share_in']:>4.0%} vs {r['share_out']:>4.0%}  z={r['z']:>5.1f}"
                  f"  (model pools {r['pooled_probability']:.0%})")

def save_candidates_json(cands: list, path: str, report: list | None = None) -> None:
    Path(path).write_text(json.dumps(
        {"heuristic": "Case Types",
         "candidates": [c.to_dict() for c in cands],
         "rejected": report or []},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
