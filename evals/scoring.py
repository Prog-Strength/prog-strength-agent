"""Dataset loading + scoring math for the macro-accuracy eval.

Pure functions, no I/O beyond `load_dataset` — everything here is
unit-testable without an LLM in the loop. Scoring contract (from the
SOW): per-trial absolute percentage error per macro, median across
trials per case, a case passes when its median calorie APE is within
the case's tolerance, and the composite is the mean of per-category
pass rates so small categories aren't drowned out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

MACROS = ("calories", "protein_g", "fat_g", "carbs_g")
CATEGORIES = ("chain", "packaged", "generic")
DEFAULT_TOLERANCE_PCT = 15.0


class DatasetError(ValueError):
    """Raised when the golden dataset fails schema validation."""


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    message: str
    expect: dict[str, float]
    tolerance_pct: float
    source: str
    notes: str = ""
    verified_at: str = ""
    # Curated PR-smoke subset member. Smoke runs (the default in CI)
    # execute only these — every eval trial spends real tokens, so the
    # cheap representative slice is the default and the full dataset is
    # opt-in.
    smoke: bool = False


@dataclass
class TrialResult:
    """One run of one case. `logged` is the summed macros of every
    custom-meal entry the agent logged during the trial (agents may
    legitimately split a meal into multiple log_custom_meal calls);
    None means the agent never logged a custom meal — that's the
    `no_log` failure mode, scored as a miss even though no numbers
    exist to compare.
    """

    logged: dict[str, float] | None
    ape: dict[str, float] | None = None
    error: str | None = None

    @property
    def no_log(self) -> bool:
        return self.logged is None


@dataclass
class CaseResult:
    case: Case
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def median_ape(self) -> dict[str, float] | None:
        """Median APE per macro across trials that produced a log.
        None when every trial was a no_log."""
        scored = [t.ape for t in self.trials if t.ape is not None]
        if not scored:
            return None
        return {m: median(t[m] for t in scored) for m in MACROS}

    @property
    def no_log_rate(self) -> float:
        if not self.trials:
            return 1.0
        return sum(1 for t in self.trials if t.no_log) / len(self.trials)

    @property
    def passed(self) -> bool:
        """A case passes when the agent logged in a majority of trials
        AND the median calorie error is inside tolerance. A mostly-
        silent agent fails even if its rare logs were accurate —
        an agent that stops logging is a regression."""
        if self.no_log_rate > 0.5:
            return False
        apes = self.median_ape
        return apes is not None and apes["calories"] <= self.case.tolerance_pct


def score_trial(logged: dict[str, float] | None, expect: dict[str, float]) -> TrialResult:
    if logged is None:
        return TrialResult(logged=None)
    ape = {m: _ape(logged.get(m, 0.0), expect[m]) for m in MACROS}
    return TrialResult(logged=logged, ape=ape)


def _ape(actual: float, expected: float) -> float:
    """Absolute percentage error. Zero-expected macros (e.g. carbs in
    plain chicken) use a 1-unit floor so small absolute misses don't
    explode to infinity."""
    return abs(actual - expected) / max(abs(expected), 1.0) * 100.0


def aggregate(results: list[CaseResult]) -> dict[str, Any]:
    """Roll case results up to per-category and overall aggregates plus
    the composite score (mean of per-category pass rates, 0-100)."""
    by_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        rows = [r for r in results if r.case.category == category]
        if not rows:
            continue
        by_category[category] = _bucket_stats(rows)

    composite = (
        sum(c["pass_rate_pct"] for c in by_category.values()) / len(by_category)
        if by_category
        else 0.0
    )
    return {
        "composite": round(composite, 1),
        "total_cases": len(results),
        "categories": by_category,
        "overall": _bucket_stats(results) if results else {},
    }


def _bucket_stats(rows: list[CaseResult]) -> dict[str, Any]:
    pass_rate = sum(1 for r in rows if r.passed) / len(rows) * 100.0
    no_log_rate = sum(r.no_log_rate for r in rows) / len(rows) * 100.0
    scored = [r.median_ape for r in rows if r.median_ape is not None]
    median_ape = (
        {m: round(median(s[m] for s in scored), 1) for m in MACROS} if scored else None
    )
    return {
        "cases": len(rows),
        "pass_rate_pct": round(pass_rate, 1),
        "no_log_rate_pct": round(no_log_rate, 1),
        "median_ape": median_ape,
    }


def load_dataset(path: Path | str) -> list[Case]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list) or not raw:
        raise DatasetError("dataset must be a non-empty JSON array")
    cases: list[Case] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw):
        case = _validate_case(item, i)
        if case.id in seen_ids:
            raise DatasetError(f"duplicate case id {case.id!r}")
        seen_ids.add(case.id)
        cases.append(case)
    return cases


def _validate_case(item: Any, index: int) -> Case:
    where = f"case[{index}]"
    if not isinstance(item, dict):
        raise DatasetError(f"{where}: not an object")
    for key in ("id", "category", "message", "expect", "source"):
        if not item.get(key):
            raise DatasetError(f"{where}: missing required field {key!r}")
    if item["category"] not in CATEGORIES:
        raise DatasetError(
            f"{where} ({item['id']}): category {item['category']!r} "
            f"not one of {CATEGORIES}"
        )
    expect = item["expect"]
    if not isinstance(expect, dict) or set(expect) != set(MACROS):
        raise DatasetError(f"{where} ({item['id']}): expect must have exactly {MACROS}")
    for macro, value in expect.items():
        if not isinstance(value, int | float) or value < 0:
            raise DatasetError(
                f"{where} ({item['id']}): expect.{macro} must be a non-negative number"
            )
    if expect["calories"] <= 0:
        raise DatasetError(f"{where} ({item['id']}): expect.calories must be positive")
    tolerance = float(item.get("tolerance_pct", DEFAULT_TOLERANCE_PCT))
    if not 0 < tolerance <= 100:
        raise DatasetError(f"{where} ({item['id']}): tolerance_pct out of range")
    return Case(
        id=str(item["id"]),
        category=item["category"],
        message=str(item["message"]),
        expect={m: float(expect[m]) for m in MACROS},
        tolerance_pct=tolerance,
        source=str(item["source"]),
        notes=str(item.get("notes", "")),
        verified_at=str(item.get("verified_at", "")),
        smoke=bool(item.get("smoke", False)),
    )


def results_to_json(
    results: list[CaseResult], meta: dict[str, Any]
) -> dict[str, Any]:
    """Full-fidelity result document written to eval-results.json —
    the artifact baselines and PR comparisons are computed from."""
    return {
        "meta": meta,
        "aggregates": aggregate(results),
        "cases": [
            {
                "id": r.case.id,
                "category": r.case.category,
                "expect": r.case.expect,
                "tolerance_pct": r.case.tolerance_pct,
                "passed": r.passed,
                "no_log_rate": r.no_log_rate,
                "median_ape": r.median_ape,
                "trials": [
                    {"logged": t.logged, "ape": t.ape, "error": t.error}
                    for t in r.trials
                ],
            }
            for r in results
        ],
    }
