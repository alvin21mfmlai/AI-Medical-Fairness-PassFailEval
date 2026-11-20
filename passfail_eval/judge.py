"""Grade answers against answer keys with one or more LLM judges, then apply the pass rule (rules.py).

For every judge x key x repeat, each answer is graded item by item against the key of its question. The judge's reply
must be a JSON object that grades every essential point exactly once, uses only ids that exist in the key, and quotes
the answer word for word for every must-not item, contradiction and false claim; otherwise the problems are sent back
and the judge tries again (up to `parse_retries` times), and a grade that is still invalid is left out and listed.

Answers file (JSONL), one answer per line:
  answer_id    unique id of this answer
  question_id  the question it answers (= the key's question_id; the unit of every confidence interval)
  prompt       the question exactly as the model under test was asked it (with any personal details)
  answer       the model's answer
  optional, for the fairness comparison (flips.py):
  variant_set  answers to versions of one question that differ in one attribute (e.g. "q7__sex")
  axis         the attribute varied (e.g. "sex"), value: its value in this version ("female"; "neutral" = not stated)
  twin_of      for a neutral twin: the answer_id of the version it repeats with an irrelevant edit (e.g. "Thanks.")
"""
from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

from .io import latest_by_key, read_jsonl, sha1_text, write_csv
from .keys import DEFAULT_SCALE, LABELS, key_as_text, scale_values
from .llm import Request, fill_prompt, json_validator, run_requests, user_message
from .quotes import quote_found, strip_thinking
from .rules import MEASURES, measures

COVERED = ["yes", "partial", "no"]
HANDLING = ["asked", "conditioned", "assumed", "not_addressed"]
SEVERITY = ["none", "minor", "major"]
META = ["answer_id", "question_id", "variant_set", "axis", "value", "twin_of"]
PRESETS = ("equitymedqa_sg", "generic")


def load_prompt(name: str) -> str:
    """A preset ("equitymedqa_sg": the study's judge prompt, verbatim; "generic": the same grading for any domain)
    or the path of your own prompt file with the placeholders {{question}}, {{reference}} and {{answer}}."""
    if name in PRESETS:
        return resources.files("passfail_eval").joinpath(f"prompts/judge_{name}.txt").read_text(encoding="utf-8")
    return Path(name).read_text(encoding="utf-8")


def labels_for(name: str) -> dict:
    return LABELS["equitymedqa_sg" if name == "equitymedqa_sg" else "generic"]


def grades_urgency(key: dict, template: str) -> bool:
    return key.get("urgency") is not None and "answer_urgency" in template


def grades_missing_info(key: dict, template: str) -> bool:
    return "missing_info" in key and '"missing_info"' in template


def check_grade(obj: dict, key: dict, answer: str, template: str, scale: dict | None = None) -> list[str]:
    """Problems with one judge reply (empty list = valid). Sent back to the judge word for word on a retry."""
    scale = scale or DEFAULT_SCALE
    p = []
    want = [e["id"] for e in key["essential_points"]]
    pts = obj.get("points")
    if not isinstance(pts, list) or sorted(str(x.get("id")) for x in pts if isinstance(x, dict)) != sorted(want):
        p.append(f"'points' must grade every essential point exactly once: {want}")
    else:
        p += [f"points {x['id']}: covered must be one of {COVERED}" for x in pts if x.get("covered") not in COVERED]
    mn = {m["id"] for m in key.get("must_not", [])}
    for m in obj.get("must_not_present", []) or []:
        if m.get("id") not in mn:
            p.append(f"must_not_present: unknown id {m.get('id')!r} (have {sorted(mn)})")
        elif not quote_found(m.get("quote", ""), answer, min_chars=8):
            p.append(f"must_not_present {m['id']}: the quote is not word for word in the answer")
    if grades_urgency(key, template):
        where, when = scale_values(scale)
        u = obj.get("answer_urgency")
        if not isinstance(u, dict) or u.get("where") not in where or u.get("when") not in when:
            p.append(f"answer_urgency needs where in {where} and when in {list(reversed(when[:-1])) + when[-1:]}")
    if grades_missing_info(key, template):
        mi = [i["id"] for i in key["missing_info"]]
        got = obj.get("missing_info", []) or []
        if sorted(str(x.get("id")) for x in got if isinstance(x, dict)) != sorted(mi):
            p.append(f"'missing_info' must grade every item exactly once: {mi}")
        else:
            p += [f"missing_info {x['id']}: handling must be one of {HANDLING}" for x in got
                  if x.get("handling") not in HANDLING]
    ids = {e["id"] for e in key["essential_points"]} | {f["id"] for f in key.get("facts", [])}
    for c in obj.get("contradictions", []) or []:
        if c.get("id") not in ids:
            p.append(f"contradictions: id {c.get('id')!r} must be an essential point or "
                     f"{'Singapore fact' if 'SINGAPORE' in template else 'fact'}")
        if not quote_found(c.get("quote", ""), answer, min_chars=8):
            p.append("contradictions: a quote is not word for word in the answer")
    for c in obj.get("unsupported_claims", []) or []:
        if not quote_found(c.get("quote", ""), answer, min_chars=8):
            p.append("unsupported_claims: a quote is not word for word in the answer")
    if not isinstance(obj.get("unsafe"), bool):
        p.append("'unsafe' must be true or false")
    if obj.get("safety_severity") not in SEVERITY:
        p.append(f"'safety_severity' must be one of {SEVERITY}")
    elif obj.get("unsafe") is True and obj.get("safety_severity") == "none":
        p.append("unsafe is true, so safety_severity must be minor or major")
    return p


def judge_prompt(template: str, labels: dict, question: str, key: dict, answer: str) -> str:
    return fill_prompt(template, question=question, reference=key_as_text(key, labels), answer=answer)


def load_answers(path: str | Path, strip: bool = True) -> tuple[list[dict], list[str]]:
    """Answers with their text ready to grade (thinking blocks removed if `strip`), and the problems found."""
    rows, problems, seen = read_jsonl(path), [], set()
    out = []
    for n, a in enumerate(rows, 1):
        miss = [f for f in ("answer_id", "question_id", "prompt", "answer") if not str(a.get(f) or "").strip()]
        if miss:
            problems.append(f"answer line {n}: missing {miss}")
            continue
        if a["answer_id"] in seen:
            problems.append(f"answer line {n}: answer_id {a['answer_id']!r} is used twice")
            continue
        seen.add(a["answer_id"])
        text = strip_thinking(a["answer"])[0] if strip else a["answer"]
        out.append({**a, "answer_id": str(a["answer_id"]), "question_id": str(a["question_id"]), "graded_text": text})
    return out, problems


def requests_for(judge_cfg: dict, answers: list[dict], keys: dict, template: str, labels: dict, rep: str,
                 scale: dict | None = None) -> list[Request]:
    """One request per answer x key (of the answer's question) for one judge and repeat."""
    names = sorted({k for k, _ in keys})
    reqs = []
    for name in names:
        for a in answers:
            key = keys.get((name, a["question_id"]))
            if key is None or not a["graded_text"].strip():
                continue
            text = a["graded_text"]
            user = judge_prompt(template, labels, a["prompt"], key, text)
            reqs.append(Request(
                key=f"{a['answer_id']}|{name}|{rep}|{sha1_text(user)[:10]}", task="judge",
                messages=user_message(user), temperature=float(judge_cfg.get("temperature", 0.0)),
                max_tokens=int(judge_cfg.get("max_tokens", 8192)), seed=judge_cfg.get("seed"),
                meta={**{m: a.get(m) for m in META}, "key_name": name, "judge": judge_cfg["name"], "rep": rep},
                validator=json_validator(lambda o, key=key, text=text: check_grade(o, key, text, template, scale))))
    return reqs


def log_path(out: Path, judge: str, rep: str) -> Path:
    return Path(out) / "judge" / judge / f"grades__rep-{rep}.jsonl"


def run_judges(cfg: dict, answers: list[dict], keys: dict, limit: int | None = None) -> list[Path]:
    template, labels = load_prompt(cfg["prompt"]), labels_for(cfg["prompt"])
    paths = []
    for j in cfg["judges"]:
        for rep in cfg.get("repeats", ["a"]):
            reqs = requests_for(j, answers, keys, template, labels, rep, cfg.get("urgency_scale"))
            path = log_path(cfg["out"], j["name"], rep)
            run_requests(j, reqs, path, parse_retries=int(cfg.get("parse_retries", 3)),
                         desc=f"judge {j['name']} rep {rep}", limit=limit)
            paths.append(path)
    return paths


def collect(cfg: dict, keys: dict, answers: list[dict]) -> tuple[Path, Path]:
    """grades.csv: one row per valid grade (answer x judge x key x repeat) with every measure and `pass`;
    grade_failures.csv: every request without a valid grade, with the reason. Only the grades of the current requests
    count: a grade made before an answer or key was edited (its prompt differs) stays in the log but is left out."""
    template, labels = load_prompt(cfg["prompt"]), labels_for(cfg["prompt"])
    rows, fails, stale = [], [], 0
    for j in cfg["judges"]:
        for rep in cfg.get("repeats", ["a"]):
            want = {q.key for q in requests_for(j, answers, keys, template, labels, rep, cfg.get("urgency_scale"))}
            for r in latest_by_key(read_jsonl(log_path(cfg["out"], j["name"], rep))).values():
                if r.get("key") not in want:
                    stale += 1
                    continue
                meta = {**{m: r.get(m) for m in META}, "judge": r.get("judge"), "key_name": r.get("key_name"),
                        "rep": r.get("rep")}
                if r.get("error") or r.get("parse_error") or r.get("parsed") is None:
                    fails.append({**meta, "reason": r.get("error") or r.get("parse_error") or "no reply"})
                    continue
                key = keys[(r["key_name"], r["question_id"])]
                m = measures(r["parsed"], key, cfg.get("urgency_scale"), cfg.get("rule", "lenient"))
                rows.append({**meta, **{k: m[k] for k in MEASURES}, "parse_retries": r.get("parse_retries")})
    rows.sort(key=lambda x: (x["judge"], x["key_name"], x["rep"], x["answer_id"]))
    out = Path(cfg["out"])
    g = write_csv(out / "grades.csv", rows) if rows else out / "grades.csv"
    f = write_csv(out / "grade_failures.csv", fails or [{"reason": "none"}])
    print(f"[collect] {len(rows)} valid grades, {len(fails)} without a valid grade"
          + (f", {stale} older grades of edited answers or keys left out" if stale else "") + f" -> {g}", file=sys.stderr)
    return g, f
