"""Baseline comparison + PR-comment rendering for eval results.

    uv run python -m evals.compare \
        --current eval-results.json \
        --baseline eval-baseline.json \
        --out eval-summary.md

The output markdown starts with the `<!-- macro-eval -->` marker the
sticky-comment workflow step keys on. Verdict thresholds are informed
guesses until evals/history.jsonl accumulates enough main-branch runs
to compute real run-to-run variance (SOW open question #3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MARKER = "<!-- macro-eval -->"

# Composite-score delta beyond which we call improvement/regression.
COMPOSITE_THRESHOLD = 3.0
# Per-category pass-rate delta (percentage points) flagged in the table.
CATEGORY_THRESHOLD_PP = 5.0

_CATEGORY_ORDER = ("chain", "packaged", "generic")


def restrict_baseline(
    baseline: dict[str, Any], case_ids: set[str]
) -> dict[str, Any] | None:
    """Recompute a baseline's aggregates over only `case_ids`, so a
    smoke-subset run compares apples to apples against a full-dataset
    baseline. Returns None when the overlap is empty (e.g. the dataset
    was renamed wholesale — comparison would be meaningless)."""
    rows = [c for c in baseline.get("cases", []) if c["id"] in case_ids]
    if not rows:
        return None
    if len(rows) == len(baseline.get("cases", [])):
        return baseline

    by_category: dict[str, dict[str, Any]] = {}
    for category in _CATEGORY_ORDER:
        bucket = [c for c in rows if c["category"] == category]
        if not bucket:
            continue
        pass_rate = sum(1 for c in bucket if c["passed"]) / len(bucket) * 100.0
        by_category[category] = {
            "cases": len(bucket),
            "pass_rate_pct": round(pass_rate, 1),
        }
    composite = sum(c["pass_rate_pct"] for c in by_category.values()) / len(by_category)
    restricted = dict(baseline)
    restricted["aggregates"] = {
        "composite": round(composite, 1),
        "total_cases": len(rows),
        "categories": by_category,
        "overall": {},
    }
    restricted["restricted_to"] = len(rows)
    return restricted


def verdict(current: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    if baseline is None:
        return "no_baseline"
    delta = current["aggregates"]["composite"] - baseline["aggregates"]["composite"]
    if delta >= COMPOSITE_THRESHOLD:
        return "improved"
    if delta <= -COMPOSITE_THRESHOLD:
        return "regressed"
    return "neutral"


_VERDICT_LINES = {
    "improved": "✅ improved",
    "regressed": "❌ regressed",
    "neutral": "➖ no significant change",
    "no_baseline": "🆕 no baseline available — this run becomes the reference",
}


def render_markdown(
    *, current: dict[str, Any], baseline: dict[str, Any] | None
) -> str:
    if baseline is not None:
        baseline = restrict_baseline(
            baseline, {c["id"] for c in current.get("cases", [])}
        )
    agg = current["aggregates"]
    meta = current.get("meta", {})
    lines = [MARKER, "## 🧪 Macro estimation eval", ""]
    if baseline is not None and baseline.get("restricted_to"):
        lines.append(
            f"_Subset run: baseline restricted to the {baseline['restricted_to']} "
            f"cases this run executed._"
        )
        lines.append("")

    v = verdict(current, baseline)
    if baseline is None:
        lines.append(f"**Verdict: {_VERDICT_LINES[v]}**")
    else:
        base_agg = baseline["aggregates"]
        lines.append(
            f"**Verdict: {_VERDICT_LINES[v]}** "
            f"(composite {base_agg['composite']:g} → {agg['composite']:g}, "
            f"threshold ±{COMPOSITE_THRESHOLD:g})"
        )
    lines.append("")

    lines.append(
        "| Category | Cases | Pass rate | Δ vs main | Cal MAPE | P/F/C MAPE | No-log |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    base_categories = (baseline or {}).get("aggregates", {}).get("categories", {})
    for category in _CATEGORY_ORDER:
        stats = agg["categories"].get(category)
        if not stats:
            continue
        lines.append(_category_row(category, stats, base_categories.get(category)))
    overall = agg.get("overall") or {}
    if overall:
        lines.append(
            _category_row(
                "**overall**",
                overall,
                (baseline or {}).get("aggregates", {}).get("overall"),
            )
        )
    lines.append("")

    failing = [c for c in current.get("cases", []) if not c["passed"]]
    if failing:
        lines.append(f"<details><summary>{len(failing)} failing cases</summary>")
        lines.append("")
        lines.append("| Case | Category | Expected cal | Cal MAPE | No-log rate |")
        lines.append("|---|---|---:|---:|---:|")
        for case in failing:
            ape = case.get("median_ape")
            cal_ape = f"{ape['calories']:.0f}%" if ape else "—"
            lines.append(
                f"| {case['id']} | {case['category']} "
                f"| {case['expect']['calories']:g} | {cal_ape} "
                f"| {case['no_log_rate'] * 100:.0f}% |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    baseline_note = ""
    if baseline is not None:
        base_meta = baseline.get("meta", {})
        baseline_note = (
            f"Baseline: {base_meta.get('git_sha', '?')} "
            f"(run {base_meta.get('ran_at', '?')}) · "
        )
    lines.append(
        f"{baseline_note}{meta.get('trials', '?')} trials/case · "
        f"{agg['total_cases']} cases · "
        f"lookup {'configured' if meta.get('lookup_configured') else 'NOT configured'} · "
        f"sha {meta.get('git_sha', '?')}"
    )
    lines.append("")
    return "\n".join(lines)


def _category_row(
    label: str, stats: dict[str, Any], base: dict[str, Any] | None
) -> str:
    delta = "—"
    if base:
        diff = stats["pass_rate_pct"] - base["pass_rate_pct"]
        flag = " ⚠️" if abs(diff) >= CATEGORY_THRESHOLD_PP and diff < 0 else ""
        delta = f"{diff:+.0f}pp{flag}"
    ape = stats.get("median_ape")
    cal = f"{ape['calories']:g}%" if ape else "—"
    pfc = (
        f"{ape['protein_g']:g}% / {ape['fat_g']:g}% / {ape['carbs_g']:g}%"
        if ape
        else "—"
    )
    return (
        f"| {label} | {stats['cases']} | {stats['pass_rate_pct']:g}% | {delta} "
        f"| {cal} | {pfc} | {stats['no_log_rate_pct']:g}% |"
    )


def history_line(current: dict[str, Any]) -> str:
    """One JSONL line for evals/history.jsonl — the long-lived accuracy
    track record appended on every main-branch run."""
    agg = current["aggregates"]
    meta = current.get("meta", {})
    return json.dumps(
        {
            "ran_at": meta.get("ran_at"),
            "git_sha": meta.get("git_sha"),
            "composite": agg["composite"],
            "lookup_configured": meta.get("lookup_configured", False),
            "pass_rates": {
                category: stats["pass_rate_pct"]
                for category, stats in agg["categories"].items()
            },
        },
        sort_keys=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--out", default="eval-summary.md")
    parser.add_argument(
        "--history-line",
        action="store_true",
        help="print the history.jsonl line for --current instead of writing markdown",
    )
    args = parser.parse_args()

    current = json.loads(Path(args.current).read_text())
    if args.history_line:
        print(history_line(current))
        return

    baseline = None
    if args.baseline and Path(args.baseline).exists():
        baseline = json.loads(Path(args.baseline).read_text())
    markdown = render_markdown(current=current, baseline=baseline)
    Path(args.out).write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
