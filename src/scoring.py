"""
Task 3 - Employee Score Calculation.

Assign each message a numeric value from its sentiment label:
    Positive -> +1   Negative -> -1   Neutral -> 0
then aggregate per employee per calendar month. The score resets every month
because we group by (employee, year-month) and sum independently per group.
"""
from __future__ import annotations

import pandas as pd

SENTIMENT_VALUE = {"Positive": 1, "Negative": -1, "Neutral": 0}


def add_message_value(df: pd.DataFrame, sentiment_col: str = "sentiment") -> pd.DataFrame:
    df = df.copy()
    df["msg_value"] = df[sentiment_col].map(SENTIMENT_VALUE).fillna(0).astype(int)
    return df


def monthly_scores(
    df: pd.DataFrame,
    employee_col: str = "from",
    date_col: str = "date",
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """Return a long table: one row per (employee, month) with the summed score."""
    df = add_message_value(df, sentiment_col)
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.to_period("M").astype(str)

    grouped = (
        df.groupby([employee_col, "month"], as_index=False)
        .agg(
            score=("msg_value", "sum"),
            messages=("msg_value", "size"),
            positive=("sentiment", lambda s: (s == "Positive").sum()),
            negative=("sentiment", lambda s: (s == "Negative").sum()),
            neutral=("sentiment", lambda s: (s == "Neutral").sum()),
        )
        .rename(columns={employee_col: "employee"})
    )
    return grouped.sort_values(["month", "employee"]).reset_index(drop=True)
