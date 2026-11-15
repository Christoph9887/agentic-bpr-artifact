"""EXTRA RESOURCES — add capacity where cases queue for it."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

WAIT_PERCENTILE = 0.75
MIN_SERVICE_H = 0.01
MIN_OCCURRENCES = 30

MIN_SATURATION = 0.70
MIN_CONTENTION = 1.10

_COLS = ["case_id", "activity", "enable_time", "start_time", "end_time", "resource"]

_DAYS = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

def _weekly_hours(calendar: dict) -> float:
    """Hours per week a calendar is open, summed over its time periods."""
    total = 0.0
    for p in calendar.get("time_periods") or []:
        try:
            i, j = _DAYS.index(p["from"].upper()), _DAYS.index(p["to"].upper())
            h0, m0, s0 = (int(x) for x in p["beginTime"].split(":"))
            h1, m1, s1 = (int(x) for x in p["endTime"].split(":"))
        except (KeyError, ValueError, AttributeError):
            continue
        span = (h1 + m1 / 60 + s1 / 3600) - (h0 + m0 / 60 + s0 / 3600)
        ndays = (j - i + 1) if j >= i else (7 - i + j + 1)
        total += max(0.0, span) * ndays
    return total

def normalize_resource_id(rid) -> str:
    """The one spelling of a resource id both artifacts can be looked up by."""

    s = str(rid)
    return s[:-2] if s.endswith(".0") else s

def resource_capacity_hours(params: dict, window_h: float) -> dict:
    """{resource_id: hours it is AVAILABLE across the log window}."""

    cal_hours = {c.get("id"): _weekly_hours(c)
                 for c in (params.get("resource_calendars") or [])}
    weeks = window_h / 168.0
    out = {}
    for prof in params.get("resource_profiles") or []:
        for res in prof.get("resource_list") or []:
            wk = cal_hours.get(res.get("calendar"))
            if wk:
                hours = wk * weeks * int(res.get("amount", 1) or 1)
                out[str(res.get("id"))] = hours
                out[normalize_resource_id(res.get("id"))] = hours
    return out

@dataclass
class ExtraResourcesEvidence:
    """What the log says about the queue at one activity."""

    waiting_time_before: float
    service_time_mean: float
    occurrences: int
    resources_serving: int
    calendar_utilisation: float
    arrival_rate_per_h: float
    service_capacity_per_h: float
    contention_ratio: float
    waiting_share_of_activity_time: float
    queue_length_mean: float

@dataclass
class ExtraResourcesCandidate:
    candidate_id: str
    heuristic: str
    target_activity: str
    evidence: ExtraResourcesEvidence
    heuristic_context: str
    additional_context: str
    process_fragment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = asdict(self.evidence)
        return d

MIN_ARRIVAL_SPAN = 0.10

def _operating_window(df: pd.DataFrame) -> tuple:
    """Cut the tail where the log is only DRAINING, and return (df, hours, note)."""

    full_h = max(1e-9, (df["end_time"].max()
                        - df["start_time"].min()).total_seconds() / 3600.0)
    if "case_id" not in df.columns:
        return df, full_h, "full span (no case_id to find arrivals with)"

    t0 = df["start_time"].min()
    last_arrival = df.groupby("case_id")["start_time"].min().max()
    op_h = max(1e-9, (last_arrival - t0).total_seconds() / 3600.0)

    inside = df[df["start_time"] <= last_arrival].copy()

    win_end = last_arrival
    overlap = ((inside["end_time"].clip(upper=win_end) - inside["start_time"])
               .dt.total_seconds() / 3600.0).clip(lower=0.0)
    inside["busy_in_window_h"] = overlap

    total_busy = float(df["serv_h"].sum())
    retained = float(overlap.sum()) / total_busy if total_busy > 0 else 0.0

    span_share = op_h / full_h if full_h > 0 else 0.0
    if span_share < MIN_ARRIVAL_SPAN or op_h >= full_h:
        return df, full_h, (
            f"full span {full_h:,.0f} h — arrivals occupy only {span_share:.0%} of "
            f"it, so they are batched and there is no operating period to measure "
            f"over")
    return inside, op_h, (
        f"operating window {op_h:,.0f} h of {full_h:,.0f} ({span_share:.0%}): ends "
        f"at the last case arrival, dropping a drain tail with no arrivals in it. "
        f"{retained:.0%} of the work falls inside, and hours are clipped to the "
        f"window so a straddling instance is counted only for the part within it")

def _measure(df: pd.DataFrame, params: dict | None = None,
             window_out: dict | None = None) -> dict:
    """Per-activity queueing facts, from one simulated or real log."""

    df = df.copy()
    for c in ("enable_time", "start_time", "end_time"):
        df[c] = pd.to_datetime(df[c], format="mixed", utc=True)
    df["wait_h"] = (df["start_time"] - df["enable_time"]).dt.total_seconds() / 3600.0
    df["serv_h"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 3600.0
    df, window_h, window_note = _operating_window(df)
    if window_out is not None:
        window_out.update(hours=window_h, note=window_note)

    capacity_h = resource_capacity_hours(params or {}, window_h)

    out: dict[str, ExtraResourcesEvidence] = {}
    for act, g in df.groupby("activity"):
        n = len(g)
        servers = g["resource"].nunique() if "resource" in g.columns else 1
        wait = float(g["wait_h"].mean())
        serv = float(g["serv_h"].mean())

        busy_h = float(g["busy_in_window_h"].sum() if "busy_in_window_h" in g
                       else g["serv_h"].sum())

        avail = sum(capacity_h.get(str(r), 0.0) for r in g["resource"].unique()) \
            if "resource" in g.columns else 0.0
        util = (busy_h / avail) if avail > 0 else busy_h / (window_h * max(1, servers))
        arrival = n / window_h
        capacity = (max(1, servers) / serv) if serv > 0 else float("inf")
        out[act] = ExtraResourcesEvidence(
            waiting_time_before=round(wait, 4),
            service_time_mean=round(serv, 5),
            occurrences=int(n),
            resources_serving=int(servers),
            calendar_utilisation=round(util, 5),
            arrival_rate_per_h=round(arrival, 5),
            service_capacity_per_h=(round(capacity, 5) if capacity != float("inf") else None),
            contention_ratio=(round(arrival / capacity, 4) if capacity not in (0, float("inf")) else 0.0),
            waiting_share_of_activity_time=round(wait / (wait + serv), 4) if (wait + serv) > 0 else 0.0,
            queue_length_mean=round(arrival * wait, 4),
        )
    return out

def detect_extra_resources(df: pd.DataFrame, *, params: dict | None = None,
                           report: list | None = None
                           ) -> list[ExtraResourcesCandidate]:
    """Activities where the queue is CAPACITY, not calendar."""

    window: dict = {}
    ev = _measure(df, params, window_out=window)
    if not ev:
        return []

    waits = sorted(e.waiting_time_before for e in ev.values())
    cut = waits[min(len(waits) - 1, int(len(waits) * WAIT_PERCENTILE))]

    kept: list[ExtraResourcesCandidate] = []
    for act, e in sorted(ev.items(), key=lambda kv: -kv[1].waiting_time_before):
        reasons = []
        if e.occurrences < MIN_OCCURRENCES:
            reasons.append(f"only {e.occurrences} occurrences (<{MIN_OCCURRENCES})")
        if e.waiting_time_before < cut:
            reasons.append(f"waiting {e.waiting_time_before:.3f} h is below the log's "
                           f"own {WAIT_PERCENTILE:.0%} percentile ({cut:.3f} h)")

        if e.service_time_mean < MIN_SERVICE_H:
            reasons.append(f"service time {e.service_time_mean:.5f} h — this is a status "
                           f"event, its wait is the resource CALENDAR, not a queue; "
                           f"adding servers cannot move it")

        saturation = (e.calendar_utilisation if e.calendar_utilisation > 0
                      else e.contention_ratio)
        if saturation < MIN_SATURATION:
            reasons.append(f"saturation {saturation:.3f} of the hours its pool is "
                           f"open — the servers it already has are not the "
                           f"constraint (wall-clock contention {e.contention_ratio:.3f} "
                           f"for reference)")
        if reasons:
            if report is not None:
                report.append({"activity": act, "rejected_because": reasons,
                               "measured_over": window.get("note"),
                               "evidence": asdict(e)})
            continue
        kept.append(ExtraResourcesCandidate(
            candidate_id="", heuristic="Extra Resources", target_activity=act,
            evidence=e,
            heuristic_context=(
                "Target problem types: long queue times and delays caused by "
                "insufficient processing capacity. Applicable when cases wait "
                "before an activity because the resources able to perform it are "
                "saturated."),
            additional_context=(
                f"{e.occurrences} occurrences served by {e.resources_serving} "
                f"resources at {e.calendar_utilisation:.0%} utilisation of the hours "
                f"they are open. Arrivals "
                f"{e.arrival_rate_per_h:.2f}/h against capacity "
                f"{e.service_capacity_per_h:.2f}/h — a contention ratio of "
                f"{e.contention_ratio:.2f}. Mean queue {e.queue_length_mean:.1f} "
                f"cases (Little's law). Waiting is {e.waiting_share_of_activity_time:.0%} "
                f"of the time a case spends here, and the service time of "
                f"{e.service_time_mean:.3f} h confirms real work is being done, so "
                f"the wait is contention rather than calendar."),
        ))
    for i, c in enumerate(kept, start=1):
        c.candidate_id = f"XR-{i:03d}"
    return kept

def _from_real_log(df: pd.DataFrame, store, params: dict) -> pd.DataFrame:
    """Reshape the REAL log into the columns :func:`_measure` expects."""

    act = "concept:name" if "concept:name" in df.columns else "activity"
    res = "org:resource" if "org:resource" in df.columns else "resource"
    case = "case:concept:name" if "case:concept:name" in df.columns else "case_id"
    end = "time:timestamp" if "time:timestamp" in df.columns else "end_time"
    start = "start_timestamp" if "start_timestamp" in df.columns else "start_time"

    out = pd.DataFrame({
        "case_id": df[case], "activity": df[act],
        "resource": df[res] if res in df.columns else "NOT_SET",
        "start_time": pd.to_datetime(df[start], format="mixed", utc=True),
        "end_time": pd.to_datetime(df[end], format="mixed", utc=True),
    })
    out = out.sort_values(["case_id", "start_time"])
    prev_end = out.groupby("case_id")["end_time"].shift(1)
    out["enable_time"] = prev_end.fillna(out["start_time"])
    return out

def detect_extra_resources_from_store(store, df: pd.DataFrame, *,
                                      context: dict | None = None,
                                      report: list | None = None
                                      ) -> list[ExtraResourcesCandidate]:
    """Pipeline entry point: the real log plus the mined resource model."""

    params = (context or {}).get("params") or {}
    return detect_extra_resources(_from_real_log(df, store, params),
                                  params=params, report=report)

def print_candidates(cands: list[ExtraResourcesCandidate]) -> None:
    print(f"\nExtra Resources: {len(cands)} candidate(s)")
    for c in cands:
        e = c.evidence
        print(f"  {c.candidate_id}  {c.target_activity}")
        print(f"      wait {e.waiting_time_before:.3f} h | service {e.service_time_mean:.3f} h "
              f"| {e.resources_serving} servers @ {e.calendar_utilisation:.0%} "
              f"| contention {e.contention_ratio:.2f} | queue {e.queue_length_mean:.1f}")

def save_candidates_json(cands: list[ExtraResourcesCandidate], path: str,
                         report: list | None = None) -> None:
    payload = {
        "heuristic": "Extra Resources",
        "candidates": [c.to_dict() for c in cands],

        "rejected": report or [],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")
