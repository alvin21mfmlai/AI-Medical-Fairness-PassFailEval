"""Tolerant extraction of a JSON object from LLM output, plus light schema checks."""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def _balanced_objects(text: str):
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : i + 1]
                start = None


def extract_json(text: str) -> dict[str, Any]:
    """Return the last top-level JSON object in `text` that parses. Raises ValueError if none."""
    if not text:
        raise ValueError("empty response")
    candidates: list[str] = []
    for m in _FENCE.finditer(text):
        candidates.extend(_balanced_objects(m.group(1)))
    candidates.extend(_balanced_objects(text))
    first_err: json.JSONDecodeError | None = None
    for cand in reversed(candidates):
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError as e:
                first_err = first_err or e
                continue
    if first_err is None and "{" in text:
        # a missing quote leaves no balanced {...}: decode from the first brace to find where it breaks
        try:
            json.loads(text[text.index("{"):])
        except json.JSONDecodeError as e:
            first_err = e
    if first_err is None:
        raise ValueError("no parseable JSON object found")
    # say where the JSON breaks, so a retry can fix it (Phase 0: Nemotron dropped opening quotes on string values)
    e = first_err
    near = e.doc[max(0, e.pos - 60): e.pos + 20].replace("\n", " ")
    raise ValueError(f"no parseable JSON object found: {e.msg} at line {e.lineno} column {e.colno}, near: ...{near}...")


def check_schema(obj: dict, schema: dict[str, Any]) -> list[str]:
    """Very small schema check. schema maps key -> type | tuple of types | list of allowed values."""
    problems = []
    for key, spec in schema.items():
        if key not in obj:
            problems.append(f"missing key '{key}'")
            continue
        val = obj[key]
        if isinstance(spec, list):
            if val not in spec:
                problems.append(f"'{key}' must be one of {spec}, got {val!r}")
        elif isinstance(spec, (type, tuple)):
            if not isinstance(val, spec):
                problems.append(f"'{key}' has type {type(val).__name__}, expected {spec}")
    return problems


def as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "y", "1"}:
            return True
        if s in {"false", "no", "n", "0"}:
            return False
    return None
