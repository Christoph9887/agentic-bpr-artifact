"""Where a rework loop lives in the mined model, and how much of it is avoidable."""

from __future__ import annotations

from collections import deque
from pathlib import Path

MIN_CASES = 100
MIN_REPEAT_SHARE = 0.05

def _first_task_after(bpmn, start, limit: int = 60) -> set:
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

def _reaches(bpmn, start, target, limit: int = 4000) -> bool:
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

def rework_gateways(bpmn_path: str, params: dict, report: list | None = None) -> dict:
    """{activity: (gateway_id, loop_path_id, exit_path_id, loop_p)} — the repeat lever."""

    import pm4py

    if not bpmn_path or not Path(bpmn_path).exists():
        return {}
    bpmn = pm4py.read_bpmn(str(bpmn_path))
    by_id = {str(n.get_id()): n for n in bpmn.get_nodes()}
    flow_by_id = {str(f.get_id()): f for f in bpmn.get_flows()}
    task_by_name = {n.get_name(): n for n in bpmn.get_nodes()
                    if "Task" in type(n).__name__ and n.get_name()}

    out = {}
    for g in params.get("gateway_branching_probabilities") or []:
        gid = str(g.get("gateway_id"))
        node = by_id.get(gid)
        probs = g.get("probabilities") or []

        def _no(why: str):
            if report is not None:
                report.append({"gateway": gid, "rejected_because": [why]})

        if node is None or "Exclusive" not in type(node).__name__:
            _no("not an exclusive gateway, so it does not decide whether the work "
                "is repeated")
            continue
        if len(probs) != 2:
            _no(f"the gateway has {len(probs)} paths; a loop and its exit are two")
            continue

        looped = []
        for i, p in enumerate(probs):
            f = flow_by_id.get(str(p.get("path_id")))
            if f is None:
                continue
            tgt = f.get_target()
            reach = ({tgt.get_name()} if "Task" in type(tgt).__name__ and tgt.get_name()
                     else _first_task_after(bpmn, tgt))
            if len(reach) == 1 and _reaches(bpmn, tgt, node):
                looped.append((i, next(iter(reach))))
        if len(looped) != 1:
            _no(f"{len(looped)} of its paths loop back to a single task; a rework "
                f"lever needs exactly one (the other being the exit)")
            continue
        i, task = looped[0]
        exit_flow = flow_by_id.get(str(probs[1 - i].get("path_id")))
        exit_tgt = exit_flow.get_target() if exit_flow is not None else None
        if exit_tgt is None:
            _no("its exit path is not in the model")
            continue

        if _reaches(bpmn, exit_tgt, node):
            _no("its exit path reaches the gateway again, so this is a choice "
                "made inside a cycle rather than the control point of one; "
                "taking the exit does not stop the work being repeated")
            continue
        loop_task = task_by_name.get(task)
        if loop_task is not None and (exit_tgt is loop_task
                                      or _reaches(bpmn, exit_tgt, loop_task)):
            _no(f"its exit path can still reach {task!r}, so taking the exit "
                f"does not stop that activity being done again")
            continue
        out.setdefault(task, (gid, str(probs[i]["path_id"]),
                              str(probs[1 - i]["path_id"]),
                              float(probs[i].get("value", 0.0))))
    return out

def repeat_profile(d, activity: str) -> dict:
    """How often one case does `activity`, and the idle time each repeat costs."""

    sub = d[d["act"] == activity]
    if sub.empty:
        return {}
    per_case = sub.groupby("case").size()
    n_cases = int(per_case.size)
    repeats = per_case[per_case >= 2]

    extra = int((per_case - 1).clip(lower=0).sum())
    gap_mean = float(sub["gap_h"].mean()) if sub["gap_h"].notna().any() else 0.0
    return {
        "occurrences": int(len(sub)),
        "cases": n_cases,
        "cases_repeating": int(repeats.size),
        "repeat_share": round(float(repeats.size) / max(1, n_cases), 4),
        "mean_per_case": round(float(per_case.mean()), 3),
        "max_per_case": int(per_case.max()),
        "extra_occurrences": extra,
        "idle_after_mean_h": round(gap_mean, 4),

        "removable_h_per_repeating_case": round(
            gap_mean * (float(extra) / max(1, repeats.size)), 4) if repeats.size else 0.0,
    }
