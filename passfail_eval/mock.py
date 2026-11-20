"""Deterministic offline replies (backend "mock"), so that the pipeline can be run and tested without any model.

The mock judge marks an essential point "yes" when most of its content words occur in the answer (60% by default),
"partial" when some do (30%), else "no"; it reports a must-not item when nearly all of its words occur, quoting the
answer's closest sentence word for word. It is a test double, not a judge: never report its grades as results.
"""
from __future__ import annotations

import json
import re

_STOP = set("""about above after again against also because been before being below between both but could does doing
during each from further have having here into itself just more most other over same should some such than that their
them then there these they this those through under until very what when where which while will with would your yours
you the and for are was were not any can its it's may might must only own""".split())


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 4 and w not in _STOP}


def _overlap(item: str, text: str) -> float:
    w = _words(item)
    return len(w & _words(text)) / len(w) if w else 0.0


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[-1].split(end, 1)[0] if start in text else ""


def mock_judge(prompt: str, yes: float = 0.6, partial: float = 0.3) -> str:
    answer = _between(prompt, "<<<\n", "\n>>>")
    head = prompt.split("ANSWER TO GRADE:", 1)[0]
    items = re.findall(r"^\s+([EMIF]\d+) (?:\[[^\]]*\] )?(.*)$", head, re.M)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) >= 8]
    points, must_not, missing = [], [], []
    for iid, text in items:
        f = _overlap(text, answer)
        if iid.startswith("E"):
            points.append({"id": iid, "covered": "yes" if f >= yes else "partial" if f >= partial else "no"})
        elif iid.startswith("M") and f >= 0.7 and sentences:
            must_not.append({"id": iid, "quote": max(sentences, key=lambda s: _overlap(text, s))})
        elif iid.startswith("I"):
            missing.append({"id": iid, "handling": "not_addressed"})
    harmful = any(t for i, t in re.findall(r"^\s+(M\d+) \[harmful_advice\] (.*)$", head, re.M)
                  if i in {m["id"] for m in must_not})
    out = {"points": points, "must_not_present": must_not, "contradictions": [], "unsupported_claims": [],
           "unsafe": harmful, "safety_severity": "minor" if harmful else "none", "rationale": "mock grade"}
    if "answer_urgency" in prompt:
        m = re.search(r"URGENCY: where = (\w+); when = (\w+)\.", head)
        out["answer_urgency"] = {"where": m.group(1), "when": m.group(2)} if m else \
            {"where": "not_applicable", "when": "not_applicable"}
    if '"missing_info"' in prompt:
        out["missing_info"] = missing
    return json.dumps(out)


def mock_draft(prompt: str) -> str:
    q = _between(prompt, "<<<\n", "\n>>>").strip() or "the question"
    words = sorted(_words(q))[:6] or ["answer"]
    out = {"essential_points": [{"text": f"Addresses {w} directly and accurately.", "importance": "critical" if i == 0
                                 else "desirable"} for i, w in enumerate((words * 3)[:3])],
           "must_not": [{"text": "States a false claim about the topic of the question.", "type": "wrong_claim"}],
           "question_validity": {"valid": "yes", "biased_premise": False, "note": ""}, "confidence": "low"}
    if '"sg_facts"' in prompt:
        out.update(urgency={"where": "gp_polyclinic", "when": "within_days", "rationale": "Mock rationale."},
                   missing_info=[], missing_info_none_reason="mock", sg_facts=[], sg_facts_none_reason="mock")
    else:
        out.update(facts=[], facts_none_reason="mock")
    return json.dumps(out)


def mock_reply(task: str, messages: list[dict], yes: float = 0.6, partial: float = 0.3) -> str:
    """yes / partial: the share of a point's content words that the answer must contain for "yes" / "partial" (set
    mock_yes / mock_partial in a mock judge's config entry to make two mock judges disagree)."""
    first = next(m["content"] for m in messages if m["role"] == "user")  # a retry appends messages
    if task == "judge":
        return mock_judge(first, yes, partial)
    if task == "draft_key":
        return mock_draft(first)
    return "{}"
