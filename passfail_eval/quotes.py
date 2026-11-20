"""Verbatim-quote checks and removal of thinking blocks from answers.

Every quote a judge gives as evidence (a must-not item the answer says, a contradiction, a false claim) must appear word
for word in the answer. A grade with a quote that does not is sent back to the judge, and is rejected if the judge
cannot correct it. This is what stops a judge from failing an answer for something the answer never said.
"""
from __future__ import annotations

import re
import unicodedata

_DASHES = re.compile(r"[‐‑‒–—―−]")
_MARKDOWN = re.compile(r"[*_`#]+")
_BULLETS = re.compile(r"[•▪◦●‣‡†§¶]")


def quote_norm(s: str) -> str:
    """Case, curly quotes, every dash variant (some models write U+2011 non-breaking hyphens), markdown emphasis
    (answers say "**SOS:** call", judges quote "SOS: call"), bullet and footnote marks and whitespace are normalised;
    nothing else is."""
    s = unicodedata.normalize("NFKC", str(s or "")).replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = _BULLETS.sub(" ", _MARKDOWN.sub("", _DASHES.sub("-", s)))
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_found(quote: str, passage: str, min_chars: int = 20) -> bool:
    """A quote counts only if it is at least `min_chars` long and appears word for word (after quote_norm) in the
    passage. Whether the quote SUPPORTS the judge's point is a separate question, which this does not decide."""
    q = quote_norm(quote)
    return len(q) >= min_chars and q in quote_norm(passage)


# MedGemma 1.5 sometimes emits "<unused94>thought ... <unused95>answer"; many reasoning models emit <think>...</think>.
_MEDGEMMA_OPEN, _MEDGEMMA_CLOSE = "<unused94>", "<unused95>"
_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)


def strip_thinking(text: str) -> tuple[str, str | None]:
    """Split a thinking block off an answer. Returns (answer, thought); thought is None when there was no block.
    A block that never closes (the token budget ran out while thinking) leaves no answer, so the answer is not graded."""
    text = text or ""
    if _MEDGEMMA_OPEN in text:
        head, _, rest = text.partition(_MEDGEMMA_OPEN)
        thought, closed, answer = rest.partition(_MEDGEMMA_CLOSE)
        thought = re.sub(r"^thought\s*", "", thought)
        return (head + (answer if closed else "")).strip(), thought.strip()
    if "<think>" in text:
        thoughts = [m.group(0) for m in _THINK.finditer(text)]
        return _THINK.sub("", text).strip(), "\n".join(thoughts)
    return text, None
