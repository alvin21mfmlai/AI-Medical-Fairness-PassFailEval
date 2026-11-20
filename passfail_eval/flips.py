"""Count how often the verdict flips when exactly one thing changes, on one scale: flips per 100 answers (or pairs).

The idea: keep everything fixed, change one thing, and count how often pass becomes fail or fail becomes pass.

  regrade  the same answer, judge and key, graded twice              -> the judge's own noise (needs 2 repeats)
  judge    the same answer and key, another judge                    -> judge disagreement (needs 2 judges)
  key      the same answer and judge, another author's key           -> key disagreement (needs 2 keys per question)
  group    two versions of a question that differ only in a personal attribute (same variant_set and axis, neither
           "neutral"), each answered afresh                          -> what a demographic edit changes
  twin     a version and its neutral twin (the same prompt with an irrelevant edit, answered afresh)
                                                                     -> what any re-asking changes (the noise floor
                                                                        for "group")
  excess   group - twin: the part of the demographic flips that re-asking alone does not explain. An interval above 0
           points to a difference between groups; one that includes 0 does not.

Two scales: "pass" (the verdict, the study's headline scale) and "any_part" (the verdict's parts: every critical point
fully covered, a must-not item, a contradiction, or the urgency level; the study's "grade_differs").

The unit of every 95% interval is the question: per question, the flips of its pairs are averaged, and the questions
are resampled with replacement (percentile bootstrap). Within one judge, every rung uses the same questions (those that
have all the rungs), so the rungs can be compared directly.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np

from .io import read_csv

ANY_PART = ("critical_covered_strict", "must_not_hit", "hallucination_1", "urgency_where")
SCALES = {
    "pass": lambda a, b: a["pass"] != b["pass"],
    "any_part": lambda a, b: any(a[k] != b[k] for k in ANY_PART),
}


# ------------------------------------------------------------------------------------------------ inputs
def load_grades(path) -> list[dict]:
    """grades.csv as written by `passfail grade` (values stay strings: "True", "False", "match", "" ...)."""
    rows = read_csv(path)
    if not rows:
        raise SystemExit(f"{path}: no grades")
    for r in rows:
        r["rep"] = r.get("rep") or "a"
    return rows


def cells(rows: list[dict]) -> dict[str, dict[tuple[str, str, str], dict]]:
    """answer_id -> (judge, key_name, rep) -> grade row."""
    out: dict = defaultdict(dict)
    for r in rows:
        out[r["answer_id"]][(r["judge"], r["key_name"], r["rep"])] = r
    return dict(out)


def complete_only(c: dict) -> dict:
    """Only the answers graded in every judge x key x repeat cell, so that every comparison uses the same answers."""
    every = set().union(*(set(v) for v in c.values())) if c else set()
    return {a: v for a, v in c.items() if set(v) == every}


# ------------------------------------------------------------------------------------------------ pairs
def group_twin_pairs(cell: dict[str, dict], diff, axis: str | None = None,
                     neutral: tuple[str, ...] = ("neutral",)) -> tuple[dict, dict]:
    """For one judge x key x repeat cell ({answer_id: row}): per question, the flip indicators of the group pairs
    (two versions in one variant_set, same axis, neither neutral nor a twin) and of the twin pairs (a version and its
    neutral twin; only versions that are themselves group versions). axis None = every axis together."""
    sets: dict = defaultdict(dict)
    for aid, r in cell.items():
        if r.get("variant_set"):
            sets[r["variant_set"]][aid] = r
    group, twin = defaultdict(list), defaultdict(list)
    twins = {r["twin_of"]: r for r in cell.values() if r.get("twin_of")}
    for its in sets.values():
        demo = {k: v for k, v in its.items() if not v.get("twin_of") and v.get("value") not in neutral
                and (axis is None or v.get("axis") == axis)}
        for a, b in itertools.combinations(sorted(demo), 2):
            if demo[a].get("axis") == demo[b].get("axis"):
                group[demo[a]["question_id"]].append(float(diff(demo[a], demo[b])))
        for k, v in demo.items():
            tw = twins.get(k)
            if tw is not None:
                twin[v["question_id"]].append(float(diff(v, tw)))
    return dict(group), dict(twin)


def cross_pairs(c: dict, cell_a: tuple, cell_b: tuple, diff, answers: list[str] | None = None) -> dict:
    """Per question: the flip indicators of the same answers in two cells (another judge, key or repeat)."""
    out = defaultdict(list)
    for aid in (answers if answers is not None else c):
        v = c[aid]
        if cell_a in v and cell_b in v:
            out[v[cell_a]["question_id"]].append(float(diff(v[cell_a], v[cell_b])))
    return dict(out)


def per_question(pairs: dict) -> dict[str, float]:
    return {q: float(np.mean(v)) for q, v in pairs.items() if v}


def pooled(per_q_list: list[dict[str, float]]) -> dict[str, float]:
    """Average, per question, over several cells (e.g. the two keys): each cell counts once per question."""
    by = defaultdict(list)
    for d in per_q_list:
        for q, x in d.items():
            by[q].append(x)
    return {q: float(np.mean(v)) for q, v in by.items()}


# ------------------------------------------------------------------------------------------------ intervals
def bootstrap(values: dict[str, float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, 2.5th, 97.5th percentile) over questions resampled with replacement, in the order of `values`."""
    keys = [k for k, v in values.items() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not keys:
        return (math.nan, math.nan, math.nan)
    x = np.array([values[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    bs = [np.mean(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return float(np.mean(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def bootstrap_ratio(num: dict[str, float], den: dict[str, float], n_boot: int = 2000,
                    seed: int = 0) -> tuple[float, float, float]:
    """Ratio of two means over the same questions (sorted), with a percentile interval from paired resamples."""
    qs = sorted(set(num) & set(den))
    a, b = np.array([num[q] for q in qs]), np.array([den[q] for q in qs])
    rng = np.random.default_rng(seed)
    sel = [rng.integers(0, len(qs), len(qs)) for _ in range(n_boot)]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.array([a[x].mean() / b[x].mean() for x in sel])
        est = a.mean() / b.mean() if b.mean() else math.nan
    r = r[np.isfinite(r)]  # a resample without any group flip has no ratio; it is left out
    if not len(r):
        return float(est), math.nan, math.nan
    return float(est), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


# ------------------------------------------------------------------------------------------------ the ladder
def _dims(c: dict) -> tuple[list[str], list[str], list[str]]:
    js, ks, rs = set(), set(), set()
    for v in c.values():
        for j, k, r in v:
            js.add(j), ks.add(k), rs.add(r)
    return sorted(js), sorted(ks), sorted(rs)


def rungs_for_judge(c: dict, judge: str, diff, neutral=("neutral",), axis: str | None = None) -> dict[str, dict]:
    """Per-question flip rates of each rung available for one judge, pooled over its keys (and key pairs)."""
    _, keys, reps = _dims(c)
    r0 = reps[0]
    out = {}
    if len(reps) > 1:
        out["regrade"] = pooled([per_question(cross_pairs(c, (judge, k, r0), (judge, k, r), diff))
                                 for k in keys for r in reps[1:]])
    if len(keys) > 1:
        out["key"] = pooled([per_question(cross_pairs(c, (judge, x, r0), (judge, y, r0), diff))
                             for x, y in itertools.combinations(keys, 2)])
    g, t = [], []
    for k in keys:
        cell = {a: v[(judge, k, r0)] for a, v in c.items() if (judge, k, r0) in v}
        gp, tp = group_twin_pairs(cell, diff, axis, neutral)
        g.append(per_question(gp)), t.append(per_question(tp))
    if any(g):
        out["group"] = pooled(g)
    if any(t):
        out["twin"] = pooled(t)
    return {k: v for k, v in out.items() if v}


def rungs_all_judges(c: dict, diff, neutral=("neutral",), axis: str | None = None) -> dict[str, dict]:
    """The same pooled over every judge, plus the judge rung (pairs of judges, same key)."""
    judges, keys, reps = _dims(c)
    per = [rungs_for_judge(c, j, diff, neutral, axis) for j in judges]
    out = {name: pooled([p[name] for p in per if name in p]) for name in ("regrade", "key", "group", "twin")}
    if len(judges) > 1:
        out["judge"] = pooled([per_question(cross_pairs(c, (x, k, reps[0]), (y, k, reps[0]), diff))
                               for x, y in itertools.combinations(judges, 2) for k in keys])
    return {k: v for k, v in out.items() if v}


ORDER = ["regrade", "judge", "key", "group", "twin"]
DIFFS = [("excess", "group", "twin"), ("key_minus_group", "key", "group"), ("key_minus_twin", "key", "twin"),
         ("key_minus_regrade", "key", "regrade"), ("group_minus_regrade", "group", "regrade")]


def ladder(rungs: dict[str, dict], n_boot: int = 2000, seed: int = 0, ratio_seed: int | None = None,
           common: bool = True) -> list[dict]:
    """Rows {what, est, lo, hi, n_questions} per 100, for the rungs, their differences and the key/group ratio.
    common: restrict every rung to the questions that have all of them (sorted), so the rungs are comparable."""
    qs = sorted(set.intersection(*(set(v) for v in rungs.values()))) if rungs and common else None
    use = {k: ({q: v[q] for q in qs} if qs is not None else dict(sorted(v.items()))) for k, v in rungs.items()}
    rows = []
    for name in ORDER:
        if name in use:
            t = bootstrap(use[name], n_boot, seed)
            rows.append({"what": name, "est": t[0] * 100, "lo": t[1] * 100, "hi": t[2] * 100, "n_questions": len(use[name])})
    for name, a, b in DIFFS:
        if a in use and b in use:
            both = sorted(set(use[a]) & set(use[b]))
            t = bootstrap({q: use[a][q] - use[b][q] for q in both}, n_boot, seed)
            rows.append({"what": name, "est": t[0] * 100, "lo": t[1] * 100, "hi": t[2] * 100, "n_questions": len(both)})
    if "key" in use and "group" in use:
        t = bootstrap_ratio(use["key"], use["group"], n_boot, seed if ratio_seed is None else ratio_seed)
        rows.append({"what": "key_over_group", "est": t[0], "lo": t[1], "hi": t[2],
                     "n_questions": len(set(use["key"]) & set(use["group"]))})
    return rows


# ------------------------------------------------------------------------------------------------ per question
def questions_table(c: dict, diff, neutral=("neutral",)) -> list[dict]:
    """One row per question: pass rate per judge x key, and the pooled rungs. Sorted by excess (group - twin), then by
    the number of judge x key cells in which group flips exceed twin flips. Use it to choose questions for expert
    review; one question has few pairs, so its numbers are noisy and are not a finding on their own."""
    judges, keys, reps = _dims(c)
    r0 = reps[0]
    allr = rungs_all_judges(c, diff, neutral)
    rate = defaultdict(lambda: defaultdict(list))
    for v in c.values():
        for (j, k, r), row in v.items():
            if r == r0:
                rate[row["question_id"]][f"pass_rate[{j}|{k}]"].append(row["pass"] == "True" or row["pass"] is True)
    above = defaultdict(int)
    for j in judges:
        for k in keys:
            cell = {a: v[(j, k, r0)] for a, v in c.items() if (j, k, r0) in v}
            gp, tp = group_twin_pairs(cell, diff, None, neutral)
            g, t = per_question(gp), per_question(tp)
            for q in set(g) & set(t):
                above[q] += g[q] > t[q]
    rows = []
    for q in sorted(rate):
        row = {"question_id": q}
        for name in ORDER:
            row[f"{name}_flips"] = round(allr[name][q] * 100, 1) if name in allr and q in allr[name] else None
        g, t = row.get("group_flips"), row.get("twin_flips")
        row["excess"] = round(g - t, 1) if g is not None and t is not None else None
        row["cells_group_above_twin"] = above.get(q, 0)
        row.update({k: round(100 * float(np.mean(v)), 1) for k, v in sorted(rate[q].items())})
        rows.append(row)
    rows.sort(key=lambda r: (-(r["excess"] if r["excess"] is not None else -1e9), -r["cells_group_above_twin"],
                             r["question_id"]))
    return rows


def axes_of(c: dict) -> list[str]:
    return sorted({r.get("axis") for v in c.values() for r in v.values() if r.get("axis") and r.get("variant_set")})


def fmt(row: dict) -> str:
    if any(isinstance(row[k], float) and math.isnan(row[k]) for k in ("est", "lo", "hi")):
        return "n/a"
    if row["what"] == "key_over_group":
        return f"{row['est']:.2f} ({row['lo']:.2f} to {row['hi']:.2f})"
    sign = "+" if row["what"] in {d[0] for d in DIFFS} else ""
    return f"{row['est']:{sign}.1f} ({row['lo']:{sign}.1f} to {row['hi']:{sign}.1f})"
