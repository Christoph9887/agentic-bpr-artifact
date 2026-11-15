"""TOOLBOX — the evidence the Critic must ASK for, plus the audited read counter."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mabpr.measure_resources import compare_resources, load_resource_aggregate

CHEAP, EXPENSIVE = "cheap", "expensive"

@dataclass
class ReadEntry:
    """One audited read. ``hypothesis`` is what makes the trail self-explaining."""
    step: int
    tool: str
    args: dict
    cost: str
    hypothesis: str
    cached: bool = False
    error: str = ""

def _hypothesis_param() -> dict:
    return {"type": "string",
            "description": "Which hypothesis id this read serves (e.g. 'H1'). Required."}

class ToolBox:
    """Executes the Critic's tool calls against the on-disk evidence, enforcing"""

    def __init__(self, *, compare_dir: str | Path, phase3_dir: str | Path,
                 detector_results_path: str | Path, n_runs: int = 20,
                 max_expensive: int = 3, sim_dir: str | Path | None = None,
                 ledger: "list | None" = None) -> None:
        self.compare_dir = Path(compare_dir)

        self.sim_dir = Path(sim_dir) if sim_dir else None
        self.phase3_dir = Path(phase3_dir)
        self.detector_results_path = Path(detector_results_path)
        self.n_runs = n_runs
        self.max_expensive = max_expensive

        self.ledger = list(ledger or [])

        self.read_log: list[ReadEntry] = []
        self._cache: dict[str, Any] = {}
        self._expensive_used = 0
        self._step = 0

        self._res_base: dict | None = None
        self._res_redes: dict | None = None
        self._detectors: list[dict] | None = None
        self._phase3: dict | None = None

    def _resources(self) -> tuple[dict, dict]:
        if self._res_base is None:
            self._res_base = load_resource_aggregate(self.compare_dir, "baseline")

            if not (self.sim_dir and sorted(self.sim_dir.glob("stats_*.csv"))):
                raise FileNotFoundError(
                    "no per-round stats_*.csv — get_resource_load needs the stats of "
                    "the round being judged, not another design's")
            self._res_redes = load_resource_aggregate(self.sim_dir, "")
        return self._res_base, self._res_redes  # type: ignore[return-value]

    def _detector_by_id(self) -> dict[str, dict]:
        if self._detectors is None:
            self._detectors = json.loads(
                self.detector_results_path.read_text(encoding="utf-8"))
        return {c["candidate_id"]: c for c in self._detectors}

    def _adapt_by_id(self) -> dict[str, dict]:
        if self._phase3 is None:
            self._phase3 = json.loads(
                (self.phase3_dir / "phase3_log.json").read_text(encoding="utf-8"))
        return {e["candidate_id"]: e for e in self._phase3.get("entries", [])}

    @property
    def specs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "name": "get_candidate_history",
                "description": (
                    "CHEAP. What this candidate was measured and judged to be in EARLIER "
                    "rounds: verdict, hours-per-case with its scatter and significance, and "
                    "which other candidates were applied alongside it. Call it before "
                    "contradicting an earlier verdict, and before treating a small number "
                    "as an effect. If the history disagrees with this round it is flagged "
                    "as VERDICT_CHANGED — check `co_applied` first, because a different "
                    "plan around a candidate genuinely changes what it does; if the "
                    "contexts match and no round is SIGNIFICANT, the disagreement is noise "
                    "and neither reading is established."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hypothesis": _hypothesis_param(),
                        "candidate_id": {"type": "string", "description": "e.g. 'CA-002'."},
                    },
                    "required": ["hypothesis", "candidate_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_resource_load",
                "description": (
                    "Per-resource utilization or task load, baseline vs redesigned, each "
                    "with a z. Use it to find WHO a queue formed against — e.g. which "
                    "members of a merged pool became saturated. Omit 'resource' to get "
                    "the top-k resources by |z|."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hypothesis": _hypothesis_param(),
                        "resource": {"type": ["string", "null"],
                                     "description": "Resource id, or null for the top-k list."},
                        "metric": {"type": "string", "enum": ["utilization", "tasks_allocated"],
                                   "description": "Which metric to compare."},
                        "top_k": {"type": "integer", "description": "How many rows when resource is null."},
                    },
                    "required": ["hypothesis", "resource", "metric", "top_k"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_candidate_evidence",
                "description": (
                    "The detector's ORIGINAL evidence for a candidate — what it CLAIMED "
                    "was inefficient and by how much. Compare that prior against the "
                    "measured result to detect a benefit that never landed."),
                "parameters": {
                    "type": "object",
                    "properties": {"hypothesis": _hypothesis_param(),
                                   "candidate_id": {"type": "string"}},
                    "required": ["hypothesis", "candidate_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_adapt_report",
                "description": (
                    "The full Phase-3 structural report for a candidate: what the edit "
                    "actually changed in the model (pool sizes, gateways dropped, paths "
                    "renormalized, whether it applied at all and why not)."),
                "parameters": {
                    "type": "object",
                    "properties": {"hypothesis": _hypothesis_param(),
                                   "candidate_id": {"type": "string"}},
                    "required": ["hypothesis", "candidate_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_case_examples",
                "description": (
                    "EXPENSIVE (hard-capped per round). Concrete case traces touching an "
                    "activity, showing per-case waiting and the resource that served it, "
                    "baseline vs redesigned. Use only when aggregate evidence cannot "
                    "settle the hypothesis."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hypothesis": _hypothesis_param(),
                        "activity": {"type": "string"},
                        "k": {"type": "integer", "description": "How many example cases per side."},
                    },
                    "required": ["hypothesis", "activity", "k"],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: dict) -> dict:
        """Run one tool call. Never raises: a bad call comes back as {"error": ...}"""

        self._step += 1
        hyp = str(args.get("hypothesis", ""))
        cost = EXPENSIVE if name == "get_case_examples" else CHEAP

        key = json.dumps([name, {k: v for k, v in sorted(args.items()) if k != "hypothesis"}],
                         sort_keys=True, default=str)
        if key in self._cache:
            self.read_log.append(ReadEntry(self._step, name, args, cost, hyp, cached=True))
            return {"cached": True,
                    "note": "You already ran this exact query; here is the same result again.",
                    **self._cache[key]}

        if cost == EXPENSIVE and self._expensive_used >= self.max_expensive:
            msg = (f"budget exhausted: get_case_examples is capped at "
                   f"{self.max_expensive} calls per round")
            self.read_log.append(ReadEntry(self._step, name, args, cost, hyp, error=msg))
            return {"error": msg}

        try:
            result = self._dispatch(name, args)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            self.read_log.append(ReadEntry(self._step, name, args, cost, hyp, error=msg))
            return {"error": msg}

        if cost == EXPENSIVE:
            self._expensive_used += 1
        self._cache[key] = result
        self.read_log.append(ReadEntry(self._step, name, args, cost, hyp))
        return result

    def _dispatch(self, name: str, args: dict) -> dict:
        if name == "get_resource_load":
            return self._get_resource_load(args.get("resource"),
                                           args.get("metric") or "utilization",
                                           int(args.get("top_k") or 10))
        if name == "get_candidate_evidence":
            return self._get_candidate_evidence(str(args["candidate_id"]))
        if name == "get_adapt_report":
            return self._get_adapt_report(str(args["candidate_id"]))
        if name == "get_case_examples":
            return self._get_case_examples(str(args["activity"]), int(args.get("k") or 3))
        if name == "get_candidate_history":
            return self._get_candidate_history(str(args["candidate_id"]))
        raise KeyError(f"unknown tool {name!r}")

    def _get_candidate_history(self, cid: str) -> dict:
        from mabpr.evidence.scorecard import ledger_digest

        if not self.ledger:
            return {"candidate_id": cid, "rounds": [],
                    "note": "this is the first judged round; there is no history yet"}
        d = ledger_digest(self.ledger, cid)
        if not d:
            return {"candidate_id": cid, "rounds": [],
                    "note": "this candidate has not been measured in any earlier round"}
        return {"candidate_id": cid, **d}

    def _get_resource_load(self, resource: str | None, metric: str, top_k: int) -> dict:
        base, redes = self._resources()
        deltas = compare_resources(base, redes, self.n_runs, metric=metric)
        rows = [d for d in deltas if d.resource == resource] if resource else deltas[:max(1, top_k)]
        if resource and not rows:
            raise KeyError(f"resource {resource!r} not found "
                           f"(examples: {[d.resource for d in deltas[:5]]})")
        return {
            "metric": metric,
            "n_resources_total": len(deltas),
            "n_significant": sum(1 for d in deltas if d.significant),
            "rows": [{"resource": d.resource, "pool": d.pool, "base": d.base_mean,
                      "redes": d.redes_mean, "z": d.z, "status": d.status} for d in rows],
        }

    def _get_candidate_evidence(self, cid: str) -> dict:
        det = self._detector_by_id().get(cid)
        if det is None:
            raise KeyError(f"unknown candidate {cid!r}")
        return {
            "candidate_id": cid,
            "heuristic": det.get("heuristic"),
            "evidence": det.get("evidence"),
            "additional_context": (det.get("additional_context") or "")[:1200],
        }

    def _get_adapt_report(self, cid: str) -> dict:
        entry = self._adapt_by_id().get(cid)
        if entry is None:
            raise KeyError(f"unknown candidate {cid!r}")

        return {"candidate_id": cid, "heuristic": entry.get("heuristic"),
                "status": entry.get("status"),
                "adapt_report": entry.get("adapt_report")}

    def _get_case_examples(self, activity: str, k: int) -> dict:
        """Illustrative traces from ONE run per side (not all 20) — these are"""

        k = max(1, min(int(k), 5))
        out: dict[str, Any] = {"activity": activity, "k": k}

        sides = [("baseline", self.compare_dir / "baseline_sim_0.csv")]
        sides.append(("redesigned",
                      (self.sim_dir / "sim_0.csv") if self.sim_dir else None))
        for side, path in sides:
            if path is None:
                out[side] = {"error": "no per-round sim_0.csv — case examples must "
                                      "come from the round being judged"}
                continue
            if not path.exists():
                out[side] = {"error": f"missing {path.name}"}
                continue
            rows = _case_rows(path, activity, k)
            out[side] = {"examples": rows}
            if not rows:
                out[side]["note"] = _why_no_rows(path, activity)
        return out

    def audit(self) -> dict:
        return {
            "total_reads": len(self.read_log),
            "cheap_reads": sum(1 for r in self.read_log if r.cost == CHEAP and not r.cached),
            "expensive_reads": self._expensive_used,
            "expensive_cap": self.max_expensive,
            "cached_hits": sum(1 for r in self.read_log if r.cached),
            "errors": sum(1 for r in self.read_log if r.error),
            "log": [r.__dict__ for r in self.read_log],
        }

    def reads_for(self, hypothesis_id: str) -> list[ReadEntry]:
        """Reads that served a hypothesis — the read-first gate uses this."""
        return [r for r in self.read_log if r.hypothesis == hypothesis_id and not r.error]

def _why_no_rows(sim_csv: Path, activity: str) -> str:
    """⚠️ AN EMPTY EXAMPLE LIST IS NOT "THIS ACTIVITY NEVER RAN"."""

    present = set()
    try:
        with open(sim_csv, encoding="utf-8", newline="") as f:
            for rec in csv.DictReader(f):
                present.add(rec.get("activity"))
    except OSError as exc:
        return f"no rows for {activity!r}, and the log could not be re-read ({exc})"
    holders = sorted(n for n in present
                     if n and activity in [part.strip() for part in n.split("+")])
    if holders:
        return (f"NOT ELIMINATED, RENAMED: {activity!r} has no rows because it was "
                f"MERGED into {holders if len(holders) > 1 else holders[0]!r}. Its work "
                f"is measured there. Read the composite, not this name.")
    return (f"{activity!r} genuinely does not appear in this log "
            f"({len(present)} distinct activities present), and no merged name "
            f"contains it — so it was removed rather than renamed.")

def _case_rows(sim_csv: Path, activity: str, k: int) -> list[dict]:
    """First k rows for `activity`, with waiting = start - enable (hours)."""
    from datetime import datetime

    def _parse(ts: str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None

    rows: list[dict] = []
    with open(sim_csv, encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            if rec.get("activity") != activity:
                continue
            en, st, ed = _parse(rec.get("enable_time", "")), _parse(rec.get("start_time", "")), _parse(rec.get("end_time", ""))
            rows.append({
                "case_id": rec.get("case_id"),
                "resource": rec.get("resource"),
                "waiting_h": round((st - en).total_seconds() / 3600, 4) if en and st else None,
                "duration_h": round((ed - st).total_seconds() / 3600, 4) if st and ed else None,
            })
            if len(rows) >= k:
                break
    return rows
