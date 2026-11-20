"""Flip counting: which pairs are compared, the per-question unit, and the intervals (offline)."""
import math

from passfail_eval.flips import (SCALES, bootstrap, bootstrap_ratio, cells, complete_only, cross_pairs,
                                 group_twin_pairs, ladder, per_question, questions_table, rungs_for_judge)


def row(aid, q, passed, value=None, twin_of="", judge="J", key="K", rep="a", axis="sex"):
    return {"answer_id": aid, "question_id": q, "variant_set": f"{q}__{axis}" if value else "", "axis": axis if value else "",
            "value": value or "", "twin_of": twin_of, "judge": judge, "key_name": key, "rep": rep,
            "pass": str(passed), "critical_covered_strict": str(passed), "must_not_hit": "False",
            "hallucination_1": "False", "urgency_where": ""}


def one_question(q="q1", key="K", f=True, m=False, ft=True, mt=True, n=True, nt=False):
    return [row(f"{q}__f", q, f, "female", key=key), row(f"{q}__m", q, m, "male", key=key),
            row(f"{q}__f__t", q, ft, "female", f"{q}__f", key=key), row(f"{q}__m__t", q, mt, "male", f"{q}__m", key=key),
            row(f"{q}__n", q, n, "neutral", key=key), row(f"{q}__n__t", q, nt, "neutral", f"{q}__n", key=key)]


def test_group_and_twin_pairs():
    cell = {r["answer_id"]: r for r in one_question()}
    g, t = group_twin_pairs(cell, SCALES["pass"])
    assert g == {"q1": [1.0]}                 # female vs male: one pair, it flips; neutral is not a group version
    assert sorted(t["q1"]) == [0.0, 1.0]      # female vs its twin: no flip; male vs its twin: flip; neutral's twin unused
    g2, _ = group_twin_pairs(cell, SCALES["pass"], axis="age")
    assert g2 == {}


def test_cross_pairs_and_ladder():
    rows = one_question(key="K1") + one_question(key="K2", f=False)  # key K2 fails the female version
    c = cells(rows)
    kp = cross_pairs(c, ("J", "K1", "a"), ("J", "K2", "a"), SCALES["pass"])
    assert per_question(kp) == {"q1": 1 / 6}
    rungs = rungs_for_judge(c, "J", SCALES["pass"])
    assert set(rungs) == {"key", "group", "twin"}
    assert rungs["group"]["q1"] == (1.0 + 0.0) / 2     # pooled over the two keys
    lad = {r["what"]: r for r in ladder(rungs)}
    assert math.isclose(lad["key"]["est"], 100 / 6) and lad["key"]["n_questions"] == 1
    assert {"excess", "key_minus_group", "key_over_group"} <= set(lad)


def test_complete_only_and_common_questions():
    rows = one_question("q1", key="K1") + one_question("q1", key="K2") + one_question("q2", key="K1")
    c = cells(rows)
    assert len(c) == 12 and len(complete_only(c)) == 6       # q2 was not graded against K2
    lad = ladder(rungs_for_judge(c, "J", SCALES["pass"]))
    assert all(r["n_questions"] == 1 for r in lad)           # key flips exist only for q1, so every rung uses q1


def test_bootstrap_is_deterministic_and_centred():
    v = {f"q{i}": i % 3 / 2 for i in range(30)}
    a, b = bootstrap(v, 500, 0), bootstrap(v, 500, 0)
    assert a == b and a[1] <= a[0] <= a[2]
    assert math.isclose(a[0], sum(v.values()) / 30)
    est, lo, hi = bootstrap_ratio({"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.0}, 200, 0)
    assert est == 2.0 and not math.isnan(lo)                  # resamples with no group flip are left out
    assert all(math.isnan(x) for x in bootstrap({}, 10))


def test_questions_table_ranks_by_excess():
    rows = one_question("q1") + one_question("q2", m=True)   # q2: no group flip
    q = questions_table(cells(rows), SCALES["pass"])
    assert [r["question_id"] for r in q] == ["q1", "q2"]
    assert q[0]["group_flips"] == 100.0 and q[1]["group_flips"] == 0.0
