"""Answer keys: the structured reference that an answer is graded against.

A key is written for ONE question, before and without seeing any answer. Its sections:

  essential_points  E1..En  required. Atomic, checkable statements that a good answer contains in substance, each
                            "critical" (leaving it out could cause harm or make the answer wrong) or "desirable"
                            (a good answer has it, but leaving it out is not harmful). At least one is critical.
  must_not          M1..Mn  required, may be empty. What a good answer must NOT say, each of type "harmful_advice"
                            or "wrong_claim".
  facts             F1..Fn  optional. Facts of the domain or setting that a good answer relies on (in EquityMedQA-SG:
                            Singapore facts). An answer that contradicts one counts as a contradiction.
  urgency           optional (medical). {where, when, rationale}: the care-seeking advice a good answer gives, on an
                            ordered scale (DEFAULT_SCALE). Reported as a measure; it does not decide pass or fail.
  missing_info      I1..In  optional (medical). What to ask before advising, and how the advice changes. Reported
                            as a measure; it does not decide pass or fail.

Ids are given by position (E1, E2, ..., M1, ...), never by the author, so they are stable and can be checked.
A key file is JSONL, one key per line: {"question_id": ..., "key": <name of the author or key set>, sections...}.
Several keys for the same question (by different authors) share the question_id and differ in "key".
"""
from __future__ import annotations

import re

IMPORTANCE = ["critical", "desirable"]
MUST_NOT_TYPES = ["harmful_advice", "wrong_claim"]
NA = "not_applicable"
# The urgency scale of EquityMedQA-SG (Singapore: "a_and_e" = emergency department, "call_995" = ambulance), each list
# from least to most urgent. Change it in the config for another setting, together with the judge prompt's wording.
DEFAULT_SCALE = {"where": ["self_care", "gp_polyclinic", "specialist", "a_and_e", "call_995"],
                 "when": ["routine", "within_days", "within_24h", "immediately"]}
PREFIX = {"essential_points": "E", "must_not": "M", "missing_info": "I", "facts": "F"}
LIST_FIELDS = ["essential_points", "must_not", "missing_info", "facts"]
REQUIRED = ["essential_points", "must_not"]

# How a key is shown to the judge. "equitymedqa_sg" is the study's wording, byte for byte.
LABELS = {
    "equitymedqa_sg": {"essential_points": "ESSENTIAL POINTS:", "must_not": "MUST NOT SAY:", "urgency": "URGENCY:",
                       "missing_info": "MISSING PATIENT INFORMATION (what to ask, and how the advice changes):",
                       "facts": "SINGAPORE FACTS RELIED ON:"},
    "generic": {"essential_points": "ESSENTIAL POINTS:", "must_not": "MUST NOT SAY:", "urgency": "URGENCY:",
                "missing_info": "MISSING INFORMATION (what to ask, and how the answer changes):",
                "facts": "FACTS RELIED ON:"},
}


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def with_ids(key: dict) -> dict:
    """Copy of a key with ids E1.., M1.., I1.., F1.. by position, tidy whitespace, and must_not = [] if absent."""
    out = {k: v for k, v in key.items() if k not in LIST_FIELDS}
    for f in LIST_FIELDS:
        if f in key or f in REQUIRED:
            out[f] = [{"id": f"{PREFIX[f]}{i + 1}", **{k: (_norm(v) if isinstance(v, str) else v)
                                                        for k, v in it.items() if k != "id"}}
                      for i, it in enumerate(key.get(f) or [])]
    return out


def scale_values(scale: dict) -> tuple[list[str], list[str]]:
    """The allowed values of where and when, including not_applicable."""
    return list(scale["where"]) + [NA], list(scale["when"]) + [NA]


def check_key(key: dict, scale: dict | None = None) -> list[str]:
    """Problems with a key as loaded from a key file (empty list = usable)."""
    scale = scale or DEFAULT_SCALE
    p = []
    for f in ("question_id", "key"):
        if not _norm(key.get(f)):
            p.append(f"'{f}' must be a non-empty string")
    ep = key.get("essential_points")
    if not isinstance(ep, list) or not ep:
        p.append("'essential_points' must be a non-empty list")
    else:
        for i, e in enumerate(ep, 1):
            if not isinstance(e, dict) or not _norm(e.get("text")) or e.get("importance") not in IMPORTANCE:
                p.append(f"essential point {i} needs 'text' and 'importance' in {IMPORTANCE}")
        if not any(isinstance(e, dict) and e.get("importance") == "critical" for e in ep):
            p.append("at least one essential point must be 'critical' (the pass rule needs one)")
    mn = key.get("must_not", [])
    if not isinstance(mn, list):
        p.append("'must_not' must be a list (it may be empty)")
    else:
        for i, m in enumerate(mn, 1):
            if not isinstance(m, dict) or not _norm(m.get("text")) or m.get("type") not in MUST_NOT_TYPES:
                p.append(f"must-not item {i} needs 'text' and 'type' in {MUST_NOT_TYPES}")
    for i, f in enumerate(key.get("facts", []) or [], 1):
        if not isinstance(f, dict) or not _norm(f.get("fact")):
            p.append(f"fact {i} needs 'fact'")
    for i, m in enumerate(key.get("missing_info", []) or [], 1):
        if not isinstance(m, dict) or not _norm(m.get("ask")) or not _norm(m.get("how_advice_changes")):
            p.append(f"missing-information item {i} needs 'ask' and 'how_advice_changes'")
    if key.get("urgency") is not None:
        u = key["urgency"]
        where, when = scale_values(scale)
        if not isinstance(u, dict) or u.get("where") not in where or u.get("when") not in when:
            p.append(f"'urgency' needs where in {where} and when in {when}")
    return p


def key_as_text(key: dict, labels: dict | None = None) -> str:
    """A key (with ids) as plain text for the judge's prompt. Sections absent from the key are left out; a section
    that is present but empty says "(none)". With the study's labels and a full key this is the study's text exactly."""
    lb = labels or LABELS["generic"]
    lines = [lb["essential_points"]]
    lines += [f"  {e['id']} [{e['importance']}] {e['text']}" for e in key["essential_points"]]
    lines.append(lb["must_not"])
    lines += [f"  {m['id']} [{m['type']}] {m['text']}" for m in key.get("must_not", [])] or ["  (none)"]
    if key.get("urgency") is not None:
        u = key["urgency"]
        lines.append(f"{lb['urgency']} where = {u['where']}; when = {u['when']}. {u.get('rationale', '')}".rstrip())
    if "missing_info" in key:
        lines.append(lb["missing_info"])
        lines += [f"  {i['id']} {i['ask']} -> {i['how_advice_changes']}" for i in key["missing_info"]] or ["  (none)"]
    if "facts" in key:
        lines.append(lb["facts"])
        lines += [f"  {f['id']} [{f['kind']}] {f['fact']}" if f.get("kind") else f"  {f['id']} {f['fact']}"
                  for f in key["facts"]] or ["  (none)"]
    return "\n".join(lines)


def load_keys(rows: list[dict], scale: dict | None = None) -> tuple[dict[tuple[str, str], dict], list[str]]:
    """{(key name, question_id): key with ids} and the problems found (each prefixed with the key it is in).
    A key with problems is left out; two keys with the same name for one question are an error."""
    out, problems = {}, []
    for n, k in enumerate(rows, 1):
        where = f"key line {n} ({k.get('key')!r}, {k.get('question_id')!r})"
        p = check_key(k, scale)
        if (str(k.get("key")), str(k.get("question_id"))) in out:
            p.append("a second key with the same name for this question")
        if p:
            problems += [f"{where}: {x}" for x in p]
            continue
        out[(str(k["key"]), str(k["question_id"]))] = with_ids(k)
    return out, problems
