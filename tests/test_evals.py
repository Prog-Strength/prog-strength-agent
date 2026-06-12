"""Unit tests for the macro-accuracy eval harness — scorer math,
dataset schema validation (run against the real checked-in dataset),
verdict thresholds, and markdown rendering. No LLM in the loop; the
runner's network path is exercised by the eval itself in CI.
"""

from pathlib import Path

import pytest

from evals import compare
from evals.scoring import (
    Case,
    CaseResult,
    DatasetError,
    aggregate,
    load_dataset,
    results_to_json,
    score_trial,
)

DATASET_PATH = Path(__file__).parent.parent / "evals" / "dataset" / "custom_meals.json"


def _case(
    case_id: str = "c1",
    category: str = "chain",
    tolerance: float = 15.0,
    expect: dict | None = None,
) -> Case:
    return Case(
        id=case_id,
        category=category,
        message="log it",
        expect=expect or {"calories": 100.0, "protein_g": 10.0, "fat_g": 5.0, "carbs_g": 20.0},
        tolerance_pct=tolerance,
        source="test",
    )


def _result(case_id: str, category: str, cal_apes: list[float | None]) -> CaseResult:
    """Build a CaseResult whose trials have the given calorie APEs
    (None = no_log trial). Non-calorie macros get the same APE."""
    result = CaseResult(case=_case(case_id, category))
    for ape in cal_apes:
        if ape is None:
            result.trials.append(score_trial(None, result.case.expect))
        else:
            logged = {
                m: v * (1 + ape / 100) for m, v in result.case.expect.items()
            }
            result.trials.append(score_trial(logged, result.case.expect))
    return result


# --- scorer math --------------------------------------------------------


def test_score_trial_exact_match_zero_ape():
    case = _case()
    trial = score_trial(dict(case.expect), case.expect)
    assert trial.ape == {"calories": 0, "protein_g": 0, "fat_g": 0, "carbs_g": 0}
    assert not trial.no_log


def test_score_trial_no_log():
    trial = score_trial(None, _case().expect)
    assert trial.no_log
    assert trial.ape is None


def test_score_trial_zero_expected_macro_uses_floor():
    # Plain chicken: carbs expected 0; logging 2g must not divide by zero.
    expect = {"calories": 375.0, "protein_g": 70.0, "fat_g": 8.0, "carbs_g": 0.0}
    trial = score_trial({**expect, "carbs_g": 2.0}, expect)
    assert trial.ape["carbs_g"] == pytest.approx(200.0)


def test_case_passes_within_tolerance_and_fails_outside():
    assert _result("ok", "chain", [10.0, 12.0, 14.0]).passed
    assert not _result("miss", "chain", [30.0, 35.0, 40.0]).passed


def test_case_median_damps_one_outlier_trial():
    # One wild trial out of three shouldn't fail the case.
    assert _result("outlier", "chain", [5.0, 8.0, 90.0]).passed


def test_case_with_majority_no_log_fails_even_if_accurate():
    result = _result("silent", "chain", [None, None, 2.0])
    assert result.no_log_rate == pytest.approx(2 / 3)
    assert not result.passed


# --- aggregation --------------------------------------------------------


def test_composite_is_mean_of_category_pass_rates():
    results = [
        _result("a", "chain", [5.0]),
        _result("b", "chain", [50.0]),  # chain: 50%
        _result("c", "generic", [5.0]),  # generic: 100%
    ]
    agg = aggregate(results)
    assert agg["categories"]["chain"]["pass_rate_pct"] == 50.0
    assert agg["categories"]["generic"]["pass_rate_pct"] == 100.0
    # Mean of (50, 100) — the 2-case chain bucket doesn't outvote
    # the 1-case generic bucket.
    assert agg["composite"] == 75.0


def test_aggregate_no_log_rate_surfaces():
    agg = aggregate([_result("a", "chain", [None, None, None])])
    assert agg["categories"]["chain"]["no_log_rate_pct"] == 100.0
    assert agg["categories"]["chain"]["median_ape"] is None


# --- dataset validation (the real one) ----------------------------------


def test_checked_in_dataset_is_valid_and_covers_all_categories():
    cases = load_dataset(DATASET_PATH)
    assert len(cases) >= 60
    categories = {c.category for c in cases}
    assert categories == {"chain", "packaged", "generic"}
    # Every case must be auditable.
    assert all(c.source for c in cases)
    # Quantity-scaling coverage — the failure mode lookup exists to fix.
    assert any("10 chicken minis" in c.message for c in cases)


def test_smoke_subset_is_small_and_stratified():
    """The smoke subset is the cost-conscious default for CI runs:
    small enough to be cheap (every trial spends tokens), and covering
    every category so a smoke verdict means something."""
    cases = load_dataset(DATASET_PATH)
    smoke = [c for c in cases if c.smoke]
    assert 18 <= len(smoke) <= 30
    per_category = {cat: sum(1 for c in smoke if c.category == cat) for cat in
                    ("chain", "packaged", "generic")}
    assert all(n >= 5 for n in per_category.values()), per_category
    # The headline failure mode stays in the cheap loop.
    assert any("10 chicken minis" in c.message for c in smoke)


def test_dataset_rejects_bad_category(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        '[{"id":"x","category":"restaurant","message":"m",'
        '"expect":{"calories":1,"protein_g":0,"fat_g":0,"carbs_g":0},"source":"s"}]'
    )
    with pytest.raises(DatasetError, match="category"):
        load_dataset(bad)


def test_dataset_rejects_duplicate_ids(tmp_path):
    case = (
        '{"id":"dup","category":"chain","message":"m",'
        '"expect":{"calories":1,"protein_g":0,"fat_g":0,"carbs_g":0},"source":"s"}'
    )
    bad = tmp_path / "bad.json"
    bad.write_text(f"[{case},{case}]")
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(bad)


def test_dataset_rejects_missing_macro(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        '[{"id":"x","category":"chain","message":"m",'
        '"expect":{"calories":1,"protein_g":0,"fat_g":0},"source":"s"}]'
    )
    with pytest.raises(DatasetError, match="expect"):
        load_dataset(bad)


# --- comparison + rendering ----------------------------------------------


def _doc(composite_shift: float = 0.0) -> dict:
    results = [
        _result("a", "chain", [5.0]),
        _result("b", "packaged", [5.0]),
        _result("c", "generic", [50.0]),
    ]
    doc = results_to_json(results, meta={"git_sha": "abc1234", "trials": 1})
    doc["aggregates"]["composite"] += composite_shift
    return doc


def test_verdict_thresholds():
    base = _doc()
    assert compare.verdict(_doc(composite_shift=5.0), base) == "improved"
    assert compare.verdict(_doc(composite_shift=-5.0), base) == "regressed"
    assert compare.verdict(_doc(composite_shift=1.0), base) == "neutral"
    assert compare.verdict(base, None) == "no_baseline"


def test_render_markdown_has_marker_verdict_and_failing_cases():
    current = _doc(composite_shift=5.0)
    out = compare.render_markdown(current=current, baseline=_doc())
    assert out.startswith(compare.MARKER)
    assert "✅ improved" in out
    assert "| chain |" in out
    # The generic case failed (50% APE) and must appear in the details.
    assert "failing cases" in out
    assert "| c | generic |" in out


def test_restrict_baseline_recomputes_over_subset():
    """A smoke run must compare against the SAME cases in the full
    baseline, not the whole-dataset composite."""
    baseline = _doc()  # cases a (chain, pass), b (packaged, pass), c (generic, fail)
    restricted = compare.restrict_baseline(baseline, {"a", "c"})
    assert restricted["restricted_to"] == 2
    agg = restricted["aggregates"]
    assert set(agg["categories"]) == {"chain", "generic"}
    # chain 100%, generic 0% → composite 50.
    assert agg["composite"] == 50.0
    # Full overlap returns the baseline untouched.
    assert compare.restrict_baseline(baseline, {"a", "b", "c"}) is baseline
    # Zero overlap → no comparison.
    assert compare.restrict_baseline(baseline, {"zzz"}) is None


def test_render_markdown_notes_subset_restriction():
    current = _doc()
    current["cases"] = [c for c in current["cases"] if c["id"] in ("a", "c")]
    out = compare.render_markdown(current=current, baseline=_doc())
    assert "baseline restricted to the 2 cases" in out


def test_render_markdown_without_baseline():
    out = compare.render_markdown(current=_doc(), baseline=None)
    assert "no baseline available" in out
    # Delta column renders as em-dash, not a crash.
    assert "| — |" in out.replace("| — |", "| — |")


def test_history_line_is_compact_json():
    import json

    line = compare.history_line(_doc())
    parsed = json.loads(line)
    assert parsed["git_sha"] == "abc1234"
    assert "composite" in parsed
    assert set(parsed["pass_rates"]) == {"chain", "packaged", "generic"}
    assert "\n" not in line


# --- real-API plumbing: JWT mint + SQLite readback -------------------------


def test_mint_jwt_round_trips_subject():
    import jwt as pyjwt

    from evals.eval_api import SIGNING_KEY, mint_jwt

    token = mint_jwt("eval-case-t0")
    claims = pyjwt.decode(token, SIGNING_KEY, algorithms=["HS256"])
    assert claims["sub"] == "eval-case-t0"
    assert claims["exp"] > claims["iat"]


def test_read_logged_macros_sums_per_user(tmp_path):
    import sqlite3

    from evals.eval_api import read_logged_macros

    db = tmp_path / "eval.db"
    conn = sqlite3.connect(db)
    # Minimal slice of the real nutrition_log_entries schema
    # (migration 012) — just the columns the readback touches.
    conn.execute(
        """CREATE TABLE nutrition_log_entries (
            user_id TEXT, calories REAL, protein_g REAL,
            fat_g REAL, carbs_g REAL, deleted_at DATETIME
        )"""
    )
    rows = [
        ("u1", 500.0, 25.0, 30.0, 40.0, None),
        ("u1", 100.0, 5.0, 2.0, 10.0, None),
        ("u1", 999.0, 9.0, 9.0, 9.0, "2026-06-11"),  # soft-deleted: excluded
        ("u2", 200.0, 10.0, 5.0, 20.0, None),
    ]
    conn.executemany("INSERT INTO nutrition_log_entries VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    assert read_logged_macros(db, "u1") == {
        "calories": 600.0,
        "protein_g": 30.0,
        "fat_g": 32.0,
        "carbs_g": 50.0,
    }
    assert read_logged_macros(db, "u2")["calories"] == 200.0
    assert read_logged_macros(db, "u3") is None
