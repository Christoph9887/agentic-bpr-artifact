"""Two-stage OpenAI client for the closed-loop agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from mabpr import config

_client = None

def _api() -> OpenAI:
    """The OpenAI client, built on first use so that no key is needed to read
    the records or rebuild a report."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.openai_api_key())
    return _client

_TRACE: dict = {"dir": None, "tag": "", "seq": 0}
_REQ_CHARS = 4000

def set_trace_sink(directory: Path | None, tag: str = "") -> None:
    """Send every subsequent LLM call to ``directory/llm_trace.jsonl``."""

    _TRACE["dir"] = Path(directory) if directory is not None else None
    _TRACE["tag"] = tag

def _clip(o: Any, budget: int = _REQ_CHARS) -> Any:
    if isinstance(o, str):
        return o if len(o) <= budget else f"{o[:budget]}… [+{len(o) - budget} chars]"
    if isinstance(o, dict):
        return {k: _clip(v, budget) for k, v in o.items()}
    if isinstance(o, list):
        return [_clip(v, budget) for v in o]
    return o

def _plain(o: Any) -> Any:
    """Best-effort JSON view of an SDK object (they are pydantic models)."""
    for attr in ("model_dump", "dict"):
        f = getattr(o, attr, None)
        if callable(f):
            try:
                return f()
            except Exception:
                pass
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_plain(v) for v in o]
    return str(o)

def _record(kind: str, *, model: str, request: Any, output: Any,
            usage: Any = None) -> None:
    """Append one call to the run's trace. NEVER raises — a failed write must"""

    d = _TRACE.get("dir")
    if d is None:
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        line = {
            "seq": _TRACE["seq"], "node": _TRACE.get("tag", ""), "kind": kind,
            "model": model, "request": _clip(_plain(request)),
            "output": _plain(output),
            "usage": _plain(usage) if usage is not None else None,
        }
        _TRACE["seq"] += 1
        with (d / "llm_trace.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass

def reason(system: str, user: str, *, model: str | None = None,
           max_tokens: int | None = None) -> str:
    """Stage 1 — free-form reasoning. Returns the raw rationale text."""

    resp = _api().chat.completions.create(
        model=model or config.FAST_MODEL,
        max_completion_tokens=max_tokens or config.MAX_REASON_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    out = resp.choices[0].message.content or ""
    _record("reason", model=model or config.FAST_MODEL,
            request={"system": system, "user": user}, output=out,
            usage=getattr(resp, "usage", None))
    return out

def extract(
    reasoning: str,
    schema_hint: str,
    *,
    model: str | None = None,
    retries: int = 2,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Stage 2 — extract structured JSON @0.0 from the reasoning text."""

    system = (
        "You extract a decision as a single JSON object. "
        "Output ONLY valid JSON, no prose. " + schema_hint
    )
    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        resp = _api().chat.completions.create(
            model=model or config.FAST_MODEL,
            max_completion_tokens=max_tokens or config.MAX_EXTRACT_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": reasoning},
            ],
        )
        raw = resp.choices[0].message.content or ""
        _record("extract", model=model or config.FAST_MODEL,
                request={"system": system, "user": reasoning}, output=raw,
                usage=getattr(resp, "usage", None))
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            last_err = e
    raise ValueError(f"extract() failed to produce valid JSON: {last_err}")

def respond(
    input_items: list[Any],
    *,
    model: str | None = None,
    tools: list[dict] | None = None,
    json_format: bool = False,
    max_output_tokens: int = 6000,
):
    """One call to the Responses API. Returns the raw response object."""

    kwargs: dict[str, Any] = {
        "model": model or config.REASONING_MODEL,
        "input": input_items,
        "max_output_tokens": max_output_tokens,
    }
    if tools:
        kwargs["tools"] = tools
    if json_format:
        kwargs["text"] = {"format": {"type": "json_object"}}
    resp = _api().responses.create(**kwargs)

    _record("respond", model=kwargs["model"],
            request={"input": input_items,
                     "tools": [t.get("name") for t in (tools or [])]},
            output=getattr(resp, "output", None),
            usage=getattr(resp, "usage", None))
    return resp

def function_calls(response) -> list[Any]:
    """The ``function_call`` items in a response (empty when the model is done)."""
    return [i for i in response.output if getattr(i, "type", "") == "function_call"]

def truncated(resp) -> bool:
    """Did the model run out of output budget mid-answer?"""

    if getattr(resp, "status", None) == "incomplete":
        return True
    reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)
    return reason == "max_output_tokens"

def respond_json(
    input_items: list[Any],
    *,
    model: str | None = None,
    max_output_tokens: int = 6000,
    retries: int = 2,
) -> dict[str, Any]:
    """Responses-API call constrained to JSON, parsed. Retries on a decode"""

    last: Exception | None = None
    budget = max_output_tokens
    for _ in range(max(1, retries)):
        resp = respond(input_items, model=model, json_format=True,
                       max_output_tokens=budget)
        text = resp.output_text or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:

            if "Extra data" in str(e):
                try:
                    obj, end = json.JSONDecoder().raw_decode(text.lstrip())
                    if isinstance(obj, dict):
                        print(f"      ! JSON had {len(text) - end} trailing char(s); "
                              f"took the first complete object", flush=True)
                        return obj
                except json.JSONDecodeError:
                    pass
            last = e
            if truncated(resp):
                budget *= 2
                print(f"      ! JSON truncated at {budget // 2} tokens, retrying at {budget}",
                      flush=True)
    raise ValueError(f"respond_json() failed to produce valid JSON: {last}")

def two_stage(
    system: str,
    user: str,
    schema_hint: str,
    *,
    model: str | None = None,
    trace_dir: Path | None = None,
    tag: str = "call",
    reason_max: int | None = None,
    extract_max: int | None = None,
) -> dict[str, Any]:
    """reason -> extract, persisting the trace for audit."""

    reasoning = reason(system, user, model=model, max_tokens=reason_max)
    result = extract(reasoning, schema_hint, model=model, max_tokens=extract_max)

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{tag}_full.json").write_text(
            json.dumps(
                {
                    "model": model or config.FAST_MODEL,
                    "system": system,
                    "user": user,
                    "reasoning": reasoning,
                    "extracted": result,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (trace_dir / f"{tag}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return result
