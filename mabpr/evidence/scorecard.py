"""SCORECARD — every applied candidate's measured contribution, on four columns."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import pandas as pd

from mabpr.evidence.exogenous import absorbed_by_edit, exogenous_saving

_COLS = ["case_id", "activity", "enable_time", "start_time", "end_time", "resource"]
Z = 3.0

def hours_per_case(sim_csv: str | Path) -> tuple[dict[str, float], dict[str, set]]:
    """{activity: mean hours a case spends there}, and {activity: {resources}}."""

    df = pd.read_csv(sim_csv, usecols=lambda c: c in _COLS)
    for c in ("enable_time", "end_time"):
        df[c] = pd.to_datetime(df[c], format="mixed", utc=True)
    n_cases = max(1, df["case_id"].nunique())
    df["h"] = (df["end_time"] - df["enable_time"]).dt.total_seconds() / 3600.0
    per_act = (df.groupby("activity")["h"].sum() / n_cases).to_dict()
    by_res: dict[str, set] = {}
    if "resource" in df.columns:
        for act, grp in df.groupby("activity"):
            by_res[act] = {str(r) for r in grp["resource"].dropna().unique()}
    del df
    return per_act, by_res

def hours_per_case_runs(sim_csvs: "list[str | Path]"
                        ) -> tuple[dict[str, list], dict[str, set]]:
    """{activity: [hours/case in run 1, run 2, ...]}, and {activity: {resources}}."""

    runs = [hours_per_case(c) for c in sim_csvs]
    acts = {a for per_act, _ in runs for a in per_act}
    per_act_runs = {a: [per_act.get(a, 0.0) for per_act, _ in runs] for a in acts}

    by_res = runs[0][1] if runs else {}
    return per_act_runs, by_res

def occurrences_per_case(sim_csv: "str | Path") -> dict:
    """{activity: how many times a case runs it}. Turns the detector's"""

    df = pd.read_csv(sim_csv, usecols=lambda c: c in ("case_id", "activity"))
    n = max(1, df["case_id"].nunique())
    return (df.groupby("activity")["case_id"].size() / n).to_dict()

def _fingerprint(fp: dict) -> dict:
    """What this candidate's handler STRUCTURALLY did — the discriminator."""

    out = {"heuristic": fp.get("heuristic")}
    for k in ("pool_merged", "gateways_dropped", "paths_renormalized",
              "paths_dropped", "deleted"):
        if fp.get(k):
            out[k] = fp[k]
    return out

def _throughput(acts: list[str], shared: set, runs_b: dict, runs_r: dict,
                n_runs: int) -> dict:
    """Hours-per-case for one candidate, with the same z rule as every other"""

    def _series(names):
        if not names:
            return [0.0]
        k = max(len(v) for v in runs_r.values()) if runs_r else 1
        out = []
        for i in range(k):
            out.append(sum((runs_r.get(a, [0.0] * k)[i] if i < len(runs_r.get(a, [])) else 0.0)
                           for a in names))
        return out

    def _base_series(names):
        if not names:
            return [0.0]
        k = max(len(v) for v in runs_b.values()) if runs_b else 1
        return [sum((runs_b.get(a, [0.0] * k)[i] if i < len(runs_b.get(a, [])) else 0.0)
                    for a in names) for i in range(k)]

    sole = [a for a in acts if a not in shared]
    rb, rr = _base_series(sole), _series(sole)
    mb, mr = mean(rb), mean(rr)
    sb = pstdev(rb) if len(rb) > 1 else 0.0
    sr = pstdev(rr) if len(rr) > 1 else 0.0
    se = math.sqrt((sb ** 2 + sr ** 2) / max(1, n_runs))
    delta = mr - mb
    z = None if se == 0 else delta / se
    shared_b, shared_r = _base_series(list(shared)), _series(list(shared))

    def _per_act(a):
        b = mean(runs_b.get(a) or [0.0])
        r = mean(runs_r.get(a) or [0.0])
        return round(r - b, 4)

    return {
        "attributable": round(delta, 4),
        "attributable_sd": round(math.sqrt(sb ** 2 + sr ** 2), 4),
        "attributable_z": None if z is None else round(z, 2),

        "significant": bool(z is not None and abs(z) > Z),
        "shared_unattributable": round(mean(shared_r) - mean(shared_b), 4),
        "total": round(delta + mean(shared_r) - mean(shared_b), 4),
        "by_activity": {a: v for a in acts if abs(v := _per_act(a)) >= 0.001},

        "n_runs": (max((len(v) for v in runs_r.values()), default=0) if runs_r else 0),
        "note": "hours a case spends AT these activities (enable->end). It "
                "decomposes the MEAN case time, not the MEDIAN the loop optimises: "
                "the mean is additive, the median is not. Measured across the v10 "
                "rounds, the ratio of case-span change to summed activity-time "
                "change is far steadier against the mean (CV 0.65 on 2012, 0.11 on "
                "2017) than against the median (1.75 and 0.22). And activities "
                "overlap, so these hours do not sum to the case span at all: "
                "activity time is 3.2x the case span on BPI 2012 and 2.7x on 2017. "
                "`attributable` excludes hours on activities another applied "
                "candidate also touches, and carries a z on the same |z|>3 rule as "
                "waiting and duration — an insignificant value is a coin flip, not "
                "a small effect.",
    }

def _unattributed(deltas: list[dict], owners: dict, runs_b, runs_r,
                  applied: dict, n_runs: int) -> dict:
    """What changed on activities NO applied candidate touches."""

    acts = sorted(set(runs_b or {}) | set(runs_r or {})) if runs_b is not None else []
    free = [a for a in acts if a not in owners]
    by_activity = {}
    series_b, series_r = [], []
    if free and runs_b is not None:
        k = max((len(v) for v in runs_r.values()), default=0)
        for i in range(k):
            series_r.append(sum((runs_r.get(a) or [0.0] * k)[i] if i < len(runs_r.get(a, []))
                                else 0.0 for a in free))
        kb = max((len(v) for v in runs_b.values()), default=0)
        for i in range(kb):
            series_b.append(sum((runs_b.get(a) or [0.0] * kb)[i] if i < len(runs_b.get(a, []))
                                else 0.0 for a in free))
        for a in free:
            v = round(mean(runs_r.get(a) or [0.0]) - mean(runs_b.get(a) or [0.0]), 4)
            if abs(v) >= 0.001:
                by_activity[a] = v
    mb = mean(series_b) if series_b else 0.0
    mr = mean(series_r) if series_r else 0.0
    sb = pstdev(series_b) if len(series_b) > 1 else 0.0
    sr = pstdev(series_r) if len(series_r) > 1 else 0.0
    se = math.sqrt((sb ** 2 + sr ** 2) / max(1, n_runs))
    z = None if se == 0 else (mr - mb) / se
    by_z = {d["activity"]: d for d in deltas}
    rows = []
    for a in sorted(by_activity, key=lambda x: -abs(by_activity[x])):
        d = by_z.get(a) or {}
        rows.append({"activity": a, "h_per_case": by_activity[a],
                     "waiting_z": (d.get("waiting") or {}).get("z"),
                     "duration_z": (d.get("duration") or {}).get("z")})
    sole = sorted(applied)[0] if len(applied) == 1 else None
    return {
        "activities": rows,
        "h_per_case": round(mr - mb, 4),
        "h_per_case_z": None if z is None else round(z, 2),
        "significant": bool(z is not None and abs(z) > Z),
        "caused_by": sole,
        "note": ("Change on activities no applied candidate touches — queueing "
                 "spillover. " + (f"Only {sole} was applied, so by elimination this "
                                  f"is its doing and is charged to it."
                                  if sole else
                                  "More than one candidate was applied, so this "
                                  "cannot be split between them without inventing "
                                  "a share; it belongs to the plan, not to any one "
                                  "candidate.")),
    }

def _isolation(prior: "list | None", activity: str, claimants: list) -> list:
    """Earlier rounds in which exactly ONE of this activity's claimants was applied."""

    out = []
    for r in (prior or []):
        if r.get("activity") != activity:
            continue
        alone = [c for c in (r.get("claimants") or []) if c in claimants]
        if len(alone) == 1:
            out.append({"round": r.get("round"), "mode": r.get("mode"),
                        "alone": alone[0], "h_per_case": r.get("h_per_case")})
    return out

def _sign_disagrees(measured, exo_hours) -> bool:
    """The measured effect says one direction, the log-evidenced saving the other."""

    return bool(exo_hours and measured is not None and measured > 0)

MAX_REMOVAL_FACTS = 6

def measured_removals(tried_plans) -> list:
    """Every single-candidate removal this run has ALREADY measured."""

    rows = [t for t in (tried_plans or [])
            if t.get("median_h") is not None and t.get("ordering")]
    best = {}
    for a in rows:
        sa = set(a["ordering"])
        for b in rows:
            sb = set(b["ordering"])
            if len(sa - sb) != 1 or (sb - sa):
                continue
            cid = next(iter(sa - sb))
            ka, kb = a.get("kpis") or {}, b.get("kpis") or {}
            if not ka or not kb:
                continue
            moves = {}
            for m in ("throughput_h_median", "throughput_h_mean",
                      "waiting_h_per_case_mean", "duration_h_per_case_mean"):
                pa = (ka.get(m) or {}).get("pct")
                pb = (kb.get(m) or {}).get("pct")
                if pa is None or pb is None:
                    continue
                moves[m] = {"with": pa, "without": pb, "pp": pb - pa}
            if "throughput_h_median" not in moves:
                continue
            rec = {"candidate": cid, "base_size": len(sa), "measures": moves,
                   "with_round": "%s%s" % (a.get("round"),
                                           "c" if a.get("mode") == "confirming" else ""),
                   "without_round": "%s%s" % (b.get("round"),
                                              "c" if b.get("mode") == "confirming" else "")}
            prev = best.get(cid)
            if prev is None or len(sa) > prev["base_size"]:
                best[cid] = rec
    return sorted(best.values(),
                  key=lambda r: -abs(r["measures"]["throughput_h_median"]["pp"]))

def unmeasured_removals(plan, measured) -> list:
    """Candidates in the plan whose removal this run has NOT measured."""

    done = {r["candidate"] for r in (measured or [])}
    return [c for c in (plan or []) if c not in done]

_WAIT_DIRECTION = {
    "timer_h_removed_per_occurrence": "down",
    "batch_component_h_removed": "down",
    "rules_removed": "down",
    "batch_rules_dropped": "down",
    "loop_probability_reduced": "down",
    "task_resources_added": "down",
    "resources_grown": "down",
    "resources_removed": "up",
    "assigned_tasks_removed": "up",
    "concentration": "up",
}

def mechanism_direction(activity: str, measured_h_per_case, claimants, handlers):
    """Which claimants' MECHANISM is compatible with the sign that was measured."""

    try:
        measured = float(measured_h_per_case)
    except (TypeError, ValueError):
        return None
    if not activity or len(claimants or []) < 2 or measured == 0:
        return None
    sign = "up" if measured > 0 else "down"
    rows = []
    for cid in sorted(claimants):
        h = (handlers or {}).get(cid) or {}
        dirs, ev = set(), []
        for k in sorted(h):
            if k in _WAIT_DIRECTION and h.get(k):
                dirs.add(_WAIT_DIRECTION[k])
                ev.append("%s=%s" % (k, h[k]))
        d = next(iter(dirs)) if len(dirs) == 1 else None
        rows.append({"candidate": cid, "direction": d,
                     "evidence": "; ".join(ev) or None})
    known = [r for r in rows if r["direction"]]
    if len(known) < 2:
        return None
    ruled_out = [r["candidate"] for r in known if r["direction"] != sign]
    if not ruled_out:
        return None
    return {
        "activity": activity,
        "measured": "the wait %s (%+.4f h per case)" % (
            "ROSE" if sign == "up" else "FELL", measured),
        "compatible_with_the_measured_sign": [
            r["candidate"] for r in known if r["direction"] == sign],
        "ruled_out_by_mechanism": ruled_out,
        "per_claimant": rows,
        "note": ("the account cannot attribute a shared activity to one "
                 "candidate, but a transformation that can only push this wait "
                 "the OTHER way cannot be the source of what was measured. This "
                 "is the handlers' own reports plus the measured sign -- no "
                 "isolation was run, and nothing is decided here."),
    }

def _accounting_gap(rows: dict, unattributed: dict, runs_b, runs_r,
                    footprints: dict,
                    baseline_case_hours: "float | None",
                    plan_case_hours: "float | None",
                    baseline_case_median: "float | None" = None,
                    plan_case_median: "float | None" = None,
                    activity_rows: "list | None" = None,
                    handler_reports: "dict | None" = None) -> dict:
    """Reconcile the per-candidate account against the KPI the loop ranks on."""

    if not runs_b or not runs_r:
        return {"available": False,
                "reason": "per-activity hours were not available this round"}

    activities = set(runs_b) | set(runs_r)
    activity_hours = 0.0
    for a in activities:
        activity_hours += mean(runs_r.get(a) or [0.0]) - mean(runs_b.get(a) or [0.0])

    explained = 0.0
    for cid, row in rows.items():
        thr = row.get("throughput_h_per_case") or {}
        h = thr.get("attributable")
        if h is not None:
            explained += float(h)
    explained += float((unattributed or {}).get("h_per_case") or 0.0)

    def _activities_of(fp) -> set:
        if fp is None:
            return set()
        acts = fp.get("activities") if isinstance(fp, dict) else getattr(fp, "activities", None)
        return set(acts or ())

    no_activity_row = sorted(cid for cid in rows
                             if not _activities_of(footprints.get(cid)))

    def _routing_of(fp) -> dict:
        if fp is None:
            return {}
        r = (fp.get("routing_change") if isinstance(fp, dict)
             else getattr(fp, "routing_change", None))
        return r or {}

    changed_routing = {cid: _routing_of(footprints.get(cid))
                       for cid in rows if _routing_of(footprints.get(cid))}

    contested: list = []
    unexplained = activity_hours - explained
    applied_ids = set(rows)
    for row in (activity_rows or []):
        claimants = [c for c in (row.get("claimants") or []) if c in applied_ids]
        if len(claimants) > 1 and abs(row.get("h_per_case") or 0.0) > 0:
            _md = mechanism_direction(row.get("activity"), row.get("h_per_case"),
                                       claimants, handler_reports)
            contested.append({"activity": row.get("activity"),
                              "h_per_case": row.get("h_per_case"),
                              "claimants": sorted(claimants),

                              **({"mechanism_direction": _md} if _md else {})})
    contested.sort(key=lambda r: -abs(r["h_per_case"]))

    out = {
        "available": True,
        "activity_hours_h_per_case": round(activity_hours, 4),
        "explained_by_candidates_h_per_case": round(explained, 4),
        "activity_hours_unexplained": round(unexplained, 4),
        "activity_hours_unexplained_share": (
            round(abs(unexplained) / abs(activity_hours), 3)
            if abs(activity_hours) > 1e-9 else None),
        "contested_activities": contested[:8],
        "applied_candidates_owning_no_activity": no_activity_row,
        "applied_candidates_that_changed_routing_or_waiting": changed_routing,
    }

    if baseline_case_hours is not None and plan_case_hours is not None:
        span_delta = plan_case_hours - baseline_case_hours
        out["case_span_h_per_case"] = round(span_delta, 4)
        out["case_span_not_visible_to_any_column"] = round(
            span_delta - activity_hours, 4)
        out["note"] = (
            "The columns measure ACTIVITY HOURS; the loop ranks on CASE SPAN. "
            "`case_span_not_visible_to_any_column` is the part of the span change "
            "that no per-activity column can show — time the case spends with "
            "nothing running (fitted extraneous-delay timers, routing that sends "
            "cases down longer paths). When it is large, a candidate in "
            "`applied_candidates_that_changed_routing_or_waiting` or "
            "`applied_candidates_owning_no_activity` may be driving the KPI while "
            "reading as 'nothing attributable to it alone'. A fact, not an "
            "accusation: settling it is what a confirmation simulation is for.")
    else:
        out["note"] = ("case span was not passed in, so only the activity-hours "
                       "half of the reconciliation is available")

    if None not in (baseline_case_median, plan_case_median,
                    baseline_case_hours, plan_case_hours):
        med_delta = plan_case_median - baseline_case_median
        mean_delta = plan_case_hours - baseline_case_hours
        disagree = (med_delta > 0) != (mean_delta > 0)
        out["case_span_median_delta"] = round(med_delta, 4)
        out["case_span_mean_delta"] = round(mean_delta, 4)
        out["median_and_mean_disagree_in_sign"] = bool(disagree)
        if disagree:
            worse = "median" if med_delta > 0 else "mean"
            out["shape_note"] = (
                f"The median and the mean of case span moved in OPPOSITE "
                f"directions (median {med_delta:+.2f} h, mean {mean_delta:+.2f} h). "
                f"That is a changed case MIX, not a changed speed: cases were "
                f"routed onto different paths. The {worse} — the one that got "
                f"worse — is what the loop ranks on. Only a candidate in "
                f"`applied_candidates_that_changed_routing_or_waiting` can do "
                f"this; a candidate that merely makes an activity faster or "
                f"slower moves both in the same direction. If such a candidate "
                f"reads `covered`, that is the account failing to see it, not "
                f"evidence that it is harmless — spend a confirmation.")
    return out

def build_scorecard(footprints: dict, deltas: list[dict],
                    base_sim_csvs: "list | str | Path | None" = None,
                    redes_sim_csvs: "list | str | Path | None" = None,
                    resource_deltas: list | None = None,
                    n_runs: int = 20,
                    baseline_case_hours: float | None = None,
                    activity_ledger: "list | None" = None,
                    candidates: "list | None" = None,
                    plan_case_hours: float | None = None,
                    baseline_case_median: float | None = None,
                    plan_case_median: float | None = None,
                    handler_reports: "dict | None" = None) -> dict:
    """One row per APPLIED candidate. Facts only — no verdicts."""

    applied = {cid: fp for cid, fp in footprints.items() if fp.get("applied")}
    by_activity = {d["activity"]: d for d in deltas}
    by_id = {c["candidate_id"]: c for c in (candidates or [])}

    noop_owners = {cid for cid, fp in applied.items() if fp.get("structural_noop")}

    owners: dict[str, list[str]] = {}
    for cid, fp in applied.items():
        if cid in noop_owners:
            continue
        for a in fp.get("activities") or []:
            owners.setdefault(a, []).append(cid)

    def _paths(x):
        if x is None:
            return []
        seq = x if isinstance(x, (list, tuple)) else [x]
        return [Path(p) for p in seq if Path(p).exists()]

    runs_b, runs_r, act_res = None, None, None
    occ_b: dict = {}
    bs, rs = _paths(base_sim_csvs), _paths(redes_sim_csvs)
    if bs and rs:
        runs_b, _ = hours_per_case_runs(bs)
        runs_r, act_res = hours_per_case_runs(rs)
        occ_b = occurrences_per_case(bs[0])

    res_owners: dict[str, list] = {}
    for cid, fp in applied.items():
        for a in fp.get("activities") or []:
            for r in (act_res or {}).get(a, set()):
                res_owners.setdefault(r, []).append(cid)

    res_z = {str(r.resource): r.z for r in (resource_deltas or [])
             if getattr(r, "z", None) is not None}

    resolved: dict[str, str] = {}
    if runs_b is not None:
        for a, cs in owners.items():
            if len(cs) < 2:
                continue
            gone = (mean(runs_b.get(a) or [0.0]) > 0.0
                    and mean(runs_r.get(a) or [0.0]) == 0.0)
            if not gone:
                continue
            removers = [c for c in cs if applied[c].get("removes_activities")]
            if len(removers) == 1:
                resolved[a] = removers[0]

    def _owners_of(a: str) -> list:
        return [resolved[a]] if a in resolved else owners.get(a, [])

    rows: dict[str, Any] = {}
    for cid, fp in sorted(applied.items()):
        acts = [] if cid in noop_owners else list(fp.get("activities") or [])

        acts = [a for a in acts if a not in resolved or resolved[a] == cid]
        shared = [a for a in acts if len(_owners_of(a)) > 1]
        row: dict[str, Any] = {
            "heuristic": fp.get("heuristic"),
            "activities": acts,
            "shared_with": sorted({o for a in shared for o in _owners_of(a) if o != cid}),
            "unattributable_activities": shared,
        }
        mine = {a: r for a, r in resolved.items() if r == cid}
        if mine:

            row["resolved_by_mechanism"] = {
                a: (f"another candidate also touches it, but {cid} is the only "
                    f"claimant whose handler REMOVES activities, and it disappeared "
                    f"from the log — so the change is not ambiguous")
                for a in sorted(mine)}
        if fp.get("created"):
            row["created_activities"] = list(fp["created"])

        if shared:
            row["claimants"] = {
                o: _fingerprint(applied[o]) for a in shared for o in _owners_of(a)}

            iso = {a: _isolation(activity_ledger, a, _owners_of(a)) for a in shared}
            iso = {a: v for a, v in iso.items() if v}
            if iso:
                row["isolated_in_earlier_rounds"] = iso

        for col in ("waiting", "duration"):
            row[col] = [
                {"activity": a, "z": by_activity[a][col]["z"],
                 "base": by_activity[a][col].get("base"),
                 "redes": by_activity[a][col].get("redes")}
                for a in acts
                if a in by_activity
                and (by_activity[a].get(col) or {}).get("z") is not None
                and abs(by_activity[a][col]["z"]) > Z]

        if runs_b is not None:

            row["throughput_h_per_case"] = _throughput(
                acts, set(shared), runs_b, runs_r, n_runs)

        if act_res:
            mine = {r for a in acts for r in act_res.get(a, set())}
            sole = {r for r in mine if len(res_owners.get(r, [])) == 1}
            hits = sorted(((r, res_z[r]) for r in mine if r in res_z and abs(res_z[r]) > Z),
                          key=lambda t: -abs(t[1]))
            row["resources"] = {
                "executing_its_activities": len(mine),

                "significant": [{"resource": r, "z": round(z, 1),
                                 "only_this_candidate": r in sole} for r, z in hits[:5]],
                "significant_total": len(hits),
                "significant_listed": min(5, len(hits)),

                "significant_sole_resources": sorted(
                    r for r, _z in hits if r in sole),

                "significant_only_this_candidate": sum(
                    1 for r, _ in hits if r in sole),
                "note": ("A resource here runs at least one of this candidate's "
                         "activities. Most also run other applied candidates' "
                         "activities, so a load change is the PLAN's doing unless "
                         "`only_this_candidate` is true — on BPI 2017 round 1 the "
                         "same five resources were listed under six candidates."),
            }
        if fp.get("pool_merged"):
            row["pool_merged"] = fp["pool_merged"]
        if fp.get("structural_noop"):

            row["structural_noop"] = {
                "aimed_at": sorted(fp.get("activities") or []),
                "note": ("this candidate reported applied=true but its handler made "
                         "NO change to the model — no deletion, no pool merge, no "
                         "subprocess, no gateway or path edit. It therefore owns "
                         "none of the activities it targets, and its four columns "
                         "are empty because nothing was edited, not because the "
                         "edit was ineffective. It occupies a plan slot and cannot "
                         "help; do not spend a confirmation simulation to establish "
                         "that removing it changes nothing. If it carries an "
                         "`exogenous_saving`, THAT is the only thing known about "
                         "it — the log says what it would remove if the simulator "
                         "could represent it."),

                "keeping_it_is_free": (
                    "it changes nothing in the model, so including it cannot "
                    "move any simulated number. Keep or drop it on "
                    "`exogenous_saving` alone, and say which — that choice is "
                    "safe here precisely because there is no measurement for "
                    "it to distort."),
            }

        shared_set = set(shared)
        own = [x for x in row["waiting"] + row["duration"]
               if x["activity"] not in shared_set]
        better = [x for x in own if x["z"] < 0]
        worse = [x for x in own if x["z"] > 0]

        res_sig = ((row.get("resources") or {}).get("significant_sole_resources")
                   or [x["resource"] for x in
                       ((row.get("resources") or {}).get("significant") or [])
                       if x.get("only_this_candidate")])
        thr = row.get("throughput_h_per_case") or {}
        thr_sig = bool(thr.get("significant"))
        row["net"] = {
            "measures_improved": len(better),
            "measures_worsened": len(worse),
            "measures_shared": len(row["waiting"] + row["duration"]) - len(own),
            "throughput_h_per_case": thr.get("attributable"),
            "throughput_h_per_case_sd": thr.get("attributable_sd"),
            "throughput_h_per_case_z": thr.get("attributable_z"),
            "throughput_significant": thr_sig,
            "throughput_h_per_case_shared": thr.get("shared_unattributable"),
            "resources_significant": len(res_sig),

            "any_measurable_effect": bool(better or worse or res_sig or thr_sig),
            "note": ("Counts and hours are SOLE-OWNER only. A candidate whose every "
                     "measure is shared shows zeros here and a non-empty "
                     "`unattributable_activities` — that means 'not established', "
                     "not 'no effect'. `throughput_h_per_case` is only a finding "
                     "when `throughput_significant` is true; otherwise it is noise "
                     "and must not drive a verdict. It is ONE of four columns and "
                     "carries no priority: an effect that shows only on waiting, or "
                     "only on resource utilisation, is a real effect of the "
                     "redesign and must be reported as such."),
        }

        tgt = (fp.get("activities") or []) + ((row.get("structural_noop") or {}).get("aimed_at") or [])
        occ = None
        if runs_b is not None and tgt:
            occ = sum(occ_b.get(a, 0.0) for a in set(tgt)) or None

        _edit = fp.get("routing_change") or {}
        _absorbed = absorbed_by_edit(_edit)
        if _absorbed:

            row["exogenous_absorbed_by_the_edit"] = _absorbed
        exo = exogenous_saving(by_id.get(cid) or {}, occurrences_per_case=occ,
                               baseline_case_hours=baseline_case_hours, edit=_edit)
        if exo:
            row["exogenous_saving"] = dict(exo)
            m = row["net"].get("throughput_h_per_case")
            if _sign_disagrees(m, exo.get("hours_per_occurrence")):
                row["exogenous_saving"]["DISAGREES_WITH_MEASUREMENT"] = (
                    f"the simulator measured this candidate making things WORSE "
                    f"({m:+.3f} h/case attributable) while the log says it also "
                    f"removes {exo['hours_per_occurrence']:.3f} h per occurrence "
                    f"of {exo['field']}. Both are real; they are not additive and "
                    f"neither cancels the other. Decide on the measured column and "
                    f"REPORT the estimate — do not net them off.")

        rows[cid] = row

    un = _unattributed(deltas, owners, runs_b, runs_r, applied, n_runs)
    if un.get("caused_by") and un["caused_by"] in rows:

        cid = un["caused_by"]
        rows[cid]["spillover"] = {
            "h_per_case": un["h_per_case"], "z": un["h_per_case_z"],
            "significant": un["significant"],
            "activities": [r["activity"] for r in un["activities"]],
            "note": ("this candidate was the only one applied, so change on "
                     "activities it does not touch is its doing by elimination"),
        }
        thr = rows[cid].get("throughput_h_per_case") or {}
        rows[cid]["net"]["spillover_h_per_case"] = un["h_per_case"]
        rows[cid]["net"]["h_per_case_including_spillover"] = round(
            (thr.get("attributable") or 0.0) + (un["h_per_case"] or 0.0), 4)

    act_rows = []
    if runs_b is not None:
        for a, cs in sorted(owners.items()):
            act_rows.append({"activity": a, "claimants": sorted(cs),
                             "h_per_case": round(mean(runs_r.get(a) or [0.0])
                                                 - mean(runs_b.get(a) or [0.0]), 4)})

    return {
        "candidates": rows,

        "activity_rows": act_rows,
        "unattributed": un,
        "accounting": _accounting_gap(rows, un, runs_b, runs_r, footprints,
                                      baseline_case_hours, plan_case_hours,
                                      baseline_case_median, plan_case_median,
                                      act_rows, handler_reports),
        "note": ("Every applied candidate appears, including the ones with nothing "
                 "in any column — 'no measurable effect' is a finding, not an "
                 "omission. An activity in `unattributable_activities` is touched "
                 "by more than one applied candidate and its change belongs to "
                 "neither; isolating it needs a confirmation simulation."),
    }

def ledger_rows(round_no: int, plan: "list[str]", scorecard: dict,
                assessment: dict) -> list:
    """One row per candidate measured this round — the candidate ledger's entries."""

    sc = (scorecard or {}).get("candidates") or {}
    verdicts = {e["candidate_id"]: e for e in
                ((assessment or {}).get("assessment", {}).get("per_candidate") or [])}
    applied = set(sc)
    rows = []
    for cid in sorted(set(plan or []) | applied):
        v = verdicts.get(cid, {})
        row = sc.get(cid) or {}
        net = row.get("net") or {}
        entry = {
            "round": round_no,
            "candidate_id": cid,
            "applied": cid in applied,
            "verdict": v.get("verdict"),
            "confidence": v.get("confidence"),

            "dimensions": v.get("dimensions") or {},
            "structural_noop": bool((sc.get(cid) or {}).get("structural_noop")),

            "co_applied": sorted(applied - {cid}),
        }
        if cid in applied:
            entry.update({
                "thr_h_per_case": net.get("throughput_h_per_case"),
                "thr_sd": net.get("throughput_h_per_case_sd"),
                "thr_z": net.get("throughput_h_per_case_z"),
                "thr_significant": bool(net.get("throughput_significant")),
                "measures_improved": net.get("measures_improved"),
                "measures_worsened": net.get("measures_worsened"),
                "measures_shared": net.get("measures_shared"),
                "resources_significant": net.get("resources_significant"),
                "any_measurable_effect": net.get("any_measurable_effect"),
                "shared_with": row.get("shared_with") or [],
            })
        rows.append(entry)
    return rows

def ledger_for(ledger: "list | None", cid: str) -> list:
    return [r for r in (ledger or []) if r.get("candidate_id") == cid]

def ledger_digest(ledger: "list | None", cid: str) -> dict:
    """The compact history of one candidate, plus whether its rounds AGREE."""

    rows = ledger_for(ledger, cid)
    if not rows:
        return {}
    measured = [r for r in rows if r.get("applied")]
    verdicts = [r["verdict"] for r in rows if r.get("verdict")]
    sig = [r for r in measured if r.get("thr_significant")]
    out = {
        "rounds": [r["round"] for r in rows],
        "verdicts": verdicts,
        "history": [
            (f"r{r['round']}: {r.get('verdict') or 'not judged'}"
             + ("" if not r.get("applied") else
                f" | thr {r.get('thr_h_per_case'):+.3f}"
                f"+/-{r.get('thr_sd') or 0:.3f}"
                f" ({'SIGNIFICANT' if r.get('thr_significant') else 'not significant'})"
                f" | {r.get('measures_improved')} better / {r.get('measures_worsened')} worse"

                + (" | dims " + ",".join(f"{k}={v}" for k, v in r["dimensions"].items())
                   if r.get("dimensions") else "")
                + (" | NO EDIT — the handler changed nothing"
                   if r.get("structural_noop") else "")
                + (f" | alongside {r['co_applied']}" if r.get("co_applied") else " | alone")))
            for r in rows],
    }
    if len(set(verdicts)) > 1:
        out["VERDICT_CHANGED"] = {
            "verdicts": sorted(set(verdicts)),
            "rounds_with_a_significant_throughput": [r["round"] for r in sig],
            "note": ("This candidate has been judged differently in different "
                     "rounds. Check `co_applied` before treating that as a "
                     "contradiction: a different plan around it can genuinely "
                     "change what it does. If the contexts are comparable and no "
                     "round shows a SIGNIFICANT measurement, the disagreement is "
                     "noise and neither verdict is established."),
        }
    return out
