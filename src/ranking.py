"""
Task 4 - Employee Ranking.

For every month produce:
  * Top three POSITIVE employees: highest monthly scores.
  * Top three NEGATIVE employees: lowest (most negative) monthly scores.

Sort order required by the spec: "first in descending order and then in
alphabetical order". We interpret this as:
  * Positive list: score DESCENDING, ties broken by employee name ASCENDING.
  * Negative list: score ASCENDING (most negative first), ties broken by
    employee name ASCENDING.
Only employees who actually sent messages in a month are eligible for that
month's ranking.
"""
from __future__ import annotations

import pandas as pd


def _rank_one_month(month_df: pd.DataFrame, top: bool, n: int = 3) -> pd.DataFrame:
    ascending = not top  # top positive -> descending; top negative -> ascending
    ranked = month_df.sort_values(
        ["score", "employee"], ascending=[ascending, True]
    ).head(n)
    return ranked


def rank_employees(monthly: pd.DataFrame, n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top_positive, top_negative) ranking tables across all months."""
    pos_frames, neg_frames = [], []
    for month, grp in monthly.groupby("month"):
        pos = _rank_one_month(grp, top=True, n=n).copy()
        pos["rank"] = range(1, len(pos) + 1)
        pos_frames.append(pos)

        neg = _rank_one_month(grp, top=False, n=n).copy()
        neg["rank"] = range(1, len(neg) + 1)
        neg_frames.append(neg)

    cols = ["month", "rank", "employee", "score", "messages", "positive", "negative", "neutral"]
    top_positive = pd.concat(pos_frames).reset_index(drop=True)[cols]
    top_negative = pd.concat(neg_frames).reset_index(drop=True)[cols]
    return top_positive, top_negative


def overall_ranking(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate score across all months for a single overall leaderboard."""
    overall = (
        monthly.groupby("employee", as_index=False)
        .agg(total_score=("score", "sum"), months_active=("month", "nunique"))
        .sort_values(["total_score", "employee"], ascending=[False, True])
        .reset_index(drop=True)
    )
    overall["rank"] = range(1, len(overall) + 1)
    return overall[["rank", "employee", "total_score", "months_active"]]
