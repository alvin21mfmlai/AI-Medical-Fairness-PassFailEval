"""Command line: passfail-eval {check, draft-keys, grade, flips, run} --config <file.yaml>   (see README.md)."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .config import load_config
from .flips import (SCALES, axes_of, cells, complete_only, fmt, group_twin_pairs, ladder, load_grades,
                    questions_table, rungs_all_judges, rungs_for_judge, _dims)
from .io import read_jsonl, write_csv
from .judge import collect, load_answers, load_prompt, run_judges
from .keys import load_keys


def _keys(cfg: dict):
    files = cfg["keys"] if isinstance(cfg["keys"], list) else [cfg["keys"]]
    rows = [r for f in files for r in read_jsonl(f)]
    return load_keys(rows, cfg["urgency_scale"])


def cmd_check(cfg: dict) -> bool:
    """Read the answers and keys, report problems and what each comparison will have to work with."""
    answers, pa = load_answers(cfg["answers"], cfg["strip_thinking"])
    keys, pk = _keys(cfg)
    for p in pa + pk:
        print("PROBLEM", p)
    names = sorted({k for k, _ in keys})
    qs = sorted({a["question_id"] for a in answers})
    print(f"answers: {len(answers)} for {len(qs)} questions; keys: "
          + ", ".join(f"{n} ({sum(1 for k, _ in keys if k == n)} questions)" for n in names))
    for n in names:
        miss = [q for q in qs if (n, q) not in keys]
        if miss:
            print(f"  key {n!r} has no key for {len(miss)} questions (their answers are not graded against it), "
                  f"e.g. {miss[:5]}")
    ids = {a["answer_id"] for a in answers}
    bad_twin = [a["answer_id"] for a in answers if a.get("twin_of") and a["twin_of"] not in ids]
    if bad_twin:
        print(f"PROBLEM twin_of names an unknown answer_id in {len(bad_twin)} answers, e.g. {bad_twin[:5]}")
    empty = [a["answer_id"] for a in answers if not a["graded_text"].strip()]
    if empty:
        print(f"  {len(empty)} answers are empty after removing thinking blocks (not graded), e.g. {empty[:5]}")
    vs = Counter(a.get("axis") for a in answers if a.get("variant_set") and not a.get("twin_of")
                 and a.get("value") not in cfg["neutral_values"])
    tw = sum(1 for a in answers if a.get("twin_of"))
    print(f"group versions by axis: {dict(vs) or 'none (no group or twin comparison)'}; neutral twins: {tw}")
    print(f"judges: {[j['name'] for j in cfg['judges']]} x repeats {cfg['repeats']}; prompt: {cfg['prompt']}; "
          f"rule: {cfg['rule']}")
    tmpl = load_prompt(cfg["prompt"])
    if any(k.get("urgency") is not None for k in keys.values()) and "answer_urgency" not in tmpl:
        print("  note: the keys have urgency, but this prompt does not grade it (no urgency measure)")
    return not (pa or pk or bad_twin)


def cmd_grade(cfg: dict, limit: int | None = None) -> Path:
    answers, pa = load_answers(cfg["answers"], cfg["strip_thinking"])
    keys, pk = _keys(cfg)
    if pa or pk:
        raise SystemExit("fix the problems reported by `passfail-eval check` first")
    if not cfg["judges"]:
        raise SystemExit("no judges in the config")
    run_judges(cfg, answers, keys, limit)
    return collect(cfg, keys, answers)[0]


def cmd_flips(cfg: dict, grades: str | None = None) -> Path:
    rows = load_grades(grades or Path(cfg["out"]) / "grades.csv")
    c_all = cells(rows)
    c = complete_only(c_all)
    neutral = tuple(cfg["neutral_values"])
    nb, seed = int(cfg["bootstrap"]["samples"]), int(cfg["bootstrap"]["seed"])
    out = Path(cfg["out"]) / "flips"
    judges, keys, reps = _dims(c)
    res, md = [], ["# Flip ladder", "",
                   f"{len(c)} answers graded in every one of the {len(judges) * len(keys) * len(reps)} judge x key x "
                   f"repeat cells (of {len(c_all)} answers with any valid grade). Flips per 100 answers or pairs, "
                   f"95% interval by bootstrap over questions ({nb} resamples). Within a ladder, every rung uses the "
                   f"same questions.", "",
                   "regrade = same answer, judge and key graded again; judge = another judge; key = another key; "
                   "group = two versions differing only in a personal attribute; twin = a version and its neutral twin; "
                   "excess = group - twin (above 0: a difference between groups that re-asking does not explain).", ""]
    for scale in ("pass", "any_part"):
        diff = SCALES[scale]
        md += [f"## Scale: {scale}", "", "| ladder | " + " | ".join(
            ["regrade", "judge", "key", "group", "twin", "excess", "key - group", "key / group", "questions"]) + " |",
               "|---|" + "---|" * 9]
        ladders = [(f"judge {j}", rungs_for_judge(c, j, diff, neutral)) for j in judges]
        ladders.append(("all judges", rungs_all_judges(c, diff, neutral)))
        for ax in axes_of(c):
            ladders.append((f"all judges, axis {ax}", {k: v for k, v in rungs_all_judges(c, diff, neutral, ax).items()
                                                       if k in ("group", "twin")}))
        for name, rungs in ladders:
            if not rungs:
                continue
            rows_ = ladder(rungs, nb, seed)
            by = {r["what"]: r for r in rows_}
            for r in rows_:
                res.append({"scale": scale, "ladder": name, **r})
            cell = [fmt(by[w]) if w in by else "" for w in
                    ("regrade", "judge", "key", "group", "twin", "excess", "key_minus_group", "key_over_group")]
            n = max((r["n_questions"] for r in rows_), default=0)
            md.append(f"| {name} | " + " | ".join(cell) + f" | {n} |")
        md.append("")
    write_csv(out / "ladder.csv", res)
    q = questions_table(c, SCALES["pass"], neutral)
    write_csv(out / "questions.csv", q)
    md += ["## Questions", "", f"`questions.csv`: one row per question (scale pass), sorted by excess. A question's "
           f"own numbers rest on few pairs: use the list to choose questions for expert review, not as findings.", ""]
    (out / "ladder.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[flips] -> {out / 'ladder.md'}, ladder.csv, questions.csv", file=sys.stderr)
    return out / "ladder.md"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="passfail-eval", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "draft-keys", "grade", "flips", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        if name in ("draft-keys", "grade", "run"):
            sp.add_argument("--limit", type=int, default=None, help="send at most N new requests (a trial run)")
        if name == "draft-keys":
            sp.add_argument("--author", required=True, help="name of an entry under 'authors' in the config")
            sp.add_argument("--keys-out", default=None)
        if name == "flips":
            sp.add_argument("--grades", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "check":
        sys.exit(0 if cmd_check(cfg) else 1)
    if a.cmd == "draft-keys":
        from .draft import run_draft

        author = next((m for m in cfg["authors"] if m["name"] == a.author), None)
        if author is None:
            raise SystemExit(f"no author {a.author!r} under 'authors' in the config")
        run_draft(cfg, author, cfg["questions"], a.keys_out or Path(cfg["out"]) / f"keys_{a.author}.jsonl", a.limit)
    elif a.cmd == "grade":
        cmd_grade(cfg, a.limit)
    elif a.cmd == "flips":
        cmd_flips(cfg, a.grades)
    elif a.cmd == "run":
        if not cmd_check(cfg):
            raise SystemExit("fix the problems above first")
        cmd_grade(cfg, a.limit)
        cmd_flips(cfg)


if __name__ == "__main__":
    main()
