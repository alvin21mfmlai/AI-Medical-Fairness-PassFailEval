"""Optional: draft answer keys with an LLM, one author at a time, from the question alone.

The author sees only the question, never an answer, another author's key or any label. Its reply is checked (the
problems go back to it, as for the judge) and saved as a key file that `passfail grade` reads.

Read this before relying on AI-drafted keys: in EquityMedQA-SG the author of the key changed far more verdicts than
anything else that was varied (see README). Draft keys with at least two independent authors, report the "key" rung
of the ladder, and have experts review the keys before the verdicts are used for anything that matters.

Questions file (JSONL): {"question_id": ..., "question": <the question with no personal details>}.
"""
from __future__ import annotations

import re
import sys
from importlib import resources
from pathlib import Path

from .io import latest_by_key, read_jsonl, sha1_text, write_jsonl
from .keys import DEFAULT_SCALE, IMPORTANCE, MUST_NOT_TYPES, NA, with_ids
from .llm import Request, fill_prompt, json_validator, run_requests, user_message

FACT_KINDS = ["subsidy_scheme", "screening_programme", "clinical_cutoff", "service", "guideline", "regulation", "other"]
VALIDITY = ["yes", "needs_edit", "no"]
CONFIDENCE = ["high", "medium", "low"]
LIMITS = {"essential_points": (3, 10), "must_not": (1, 6), "missing_info": (0, 5), "sg_facts": (0, 6), "facts": (0, 6)}
PREFIX = {"essential_points": "E", "must_not": "M", "missing_info": "I", "sg_facts": "F", "facts": "F"}
MAX_ITEM_CHARS = 400
FIELDS = {"equitymedqa_sg": ["essential_points", "must_not", "missing_info", "sg_facts"],
          "generic": ["essential_points", "must_not", "facts"]}


def load_draft_prompt(name: str) -> str:
    if name in FIELDS:
        return resources.files("passfail_eval").joinpath(f"prompts/draft_key_{name}.txt").read_text(encoding="utf-8")
    return Path(name).read_text(encoding="utf-8")


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _enum(obj: dict, key: str, allowed: list[str], where: str) -> list[str]:
    v = obj.get(key)
    return [] if v in allowed else [f"{where}{key} must be one of {allowed}, got {v!r}"]


def _text(obj: dict, key: str, where: str, required: bool = True) -> list[str]:
    v = obj.get(key)
    if not isinstance(v, str) or (required and not v.strip()):
        return [f"{where}{key} must be a non-empty string"]
    if len(v) > MAX_ITEM_CHARS * 2:
        return [f"{where}{key} is too long ({len(v)} characters; keep it under {MAX_ITEM_CHARS * 2})"]
    return []


def check_draft(obj: dict, preset: str = "generic", scale: dict | None = None) -> list[str]:
    """Problems with a drafted key (empty list = valid). With preset "equitymedqa_sg" this is the study's check."""
    scale = scale or DEFAULT_SCALE
    p: list[str] = []
    for f in FIELDS[preset]:
        v = obj.get(f)
        if not isinstance(v, list):
            p.append(f"'{f}' must be a list")
            continue
        lo, hi = LIMITS[f]
        if not (lo <= len(v) <= hi):
            p.append(f"'{f}' must have {lo}-{hi} items, got {len(v)}")
        elif not v and lo == 0 and not _norm(obj.get(f"{f}_none_reason")):
            p.append(f"'{f}' is empty, so give '{f}_none_reason'")
        for i, it in enumerate(v):
            w = f"{PREFIX[f]}{i + 1}."
            if not isinstance(it, dict):
                p.append(f"{w} must be an object")
                continue
            if f == "essential_points":
                p += _text(it, "text", w) + _enum(it, "importance", IMPORTANCE, w)
            elif f == "must_not":
                p += _text(it, "text", w) + _enum(it, "type", MUST_NOT_TYPES, w)
            elif f == "missing_info":
                p += _text(it, "ask", w) + _text(it, "how_advice_changes", w)
            elif f == "sg_facts":
                p += _text(it, "fact", w) + _enum(it, "kind", FACT_KINDS, w) + _text(it, "source_hint", w)
            elif f == "facts":
                p += _text(it, "fact", w) + _text(it, "source_hint", w)
    ep = obj.get("essential_points")
    if isinstance(ep, list) and ep and not any(isinstance(e, dict) and e.get("importance") == "critical" for e in ep):
        p.append("at least one essential point must be 'critical'")
    if preset == "equitymedqa_sg":
        u = obj.get("urgency")
        where, when = list(scale["where"]) + [NA], list(reversed(scale["when"])) + [NA]
        if not isinstance(u, dict):
            p.append("'urgency' must be an object with where, when, rationale")
        else:
            p += _enum(u, "where", where, "urgency.") + _enum(u, "when", when, "urgency.") + _text(u, "rationale", "urgency.")
            if (u.get("where") == NA) != (u.get("when") == NA):
                p.append("urgency.where and urgency.when must both be 'not_applicable' or neither")
            # the two most urgent places (A&E, 995) need one of the two most urgent times
            if u.get("where") in scale["where"][-2:] and u.get("when") not in scale["when"][-2:][::-1]:
                p.append("urgency: A&E or 995 needs when = 'immediately' or 'within_24h'")
    qv = obj.get("question_validity")
    if not isinstance(qv, dict):
        p.append("'question_validity' must be an object with valid, biased_premise, note")
    else:
        p += _enum(qv, "valid", VALIDITY, "question_validity.")
        if not isinstance(qv.get("biased_premise"), bool):
            p.append("question_validity.biased_premise must be true or false")
        if qv.get("valid") != "yes" or qv.get("biased_premise"):
            p += _text(qv, "note", "question_validity.")
    p += _enum(obj, "confidence", CONFIDENCE, "")
    return p


def to_key(obj: dict, question_id: str, author: str) -> dict:
    """A valid draft as a key-file line (the study's sg_facts become facts)."""
    k = dict(obj)
    if "sg_facts" in k:
        k["facts"] = k.pop("sg_facts")
    return {"question_id": question_id, "key": author, **with_ids(k)}


def run_draft(cfg: dict, author_cfg: dict, questions: str | Path, out_keys: str | Path, limit: int | None = None) -> Path:
    preset = cfg.get("draft_prompt", "generic")
    tmpl = load_draft_prompt(preset)
    fields_preset = preset if preset in FIELDS else ("equitymedqa_sg" if '"sg_facts"' in tmpl else "generic")
    qs = read_jsonl(questions)
    reqs = []
    for q in qs:
        user = fill_prompt(tmpl, question=q["question"])
        reqs.append(Request(key=f"{q['question_id']}|draft|{sha1_text(user)[:10]}", task="draft_key",
                            messages=user_message(user), temperature=float(author_cfg.get("temperature", 0.0)),
                            max_tokens=int(author_cfg.get("max_tokens", 8192)), seed=author_cfg.get("seed"),
                            meta={"question_id": str(q["question_id"]), "author": author_cfg["name"]},
                            validator=json_validator(lambda o: check_draft(o, fields_preset, cfg.get("urgency_scale")))))
    log = Path(cfg["out"]) / "draft" / author_cfg["name"] / "draft.jsonl"
    run_requests(author_cfg, reqs, log, parse_retries=int(cfg.get("parse_retries", 3)),
                 desc=f"draft {author_cfg['name']}", limit=limit)
    keys, failed = [], 0
    for r in sorted(latest_by_key(read_jsonl(log)).values(), key=lambda r: r["question_id"]):
        if r.get("error") or r.get("parse_error") or r.get("parsed") is None:
            failed += 1
            continue
        keys.append(to_key(r["parsed"], r["question_id"], author_cfg["name"]))
    write_jsonl(out_keys, keys)
    print(f"[draft] {len(keys)} keys by {author_cfg['name']} ({failed} questions without a valid key) -> {out_keys}",
          file=sys.stderr)
    return Path(out_keys)
