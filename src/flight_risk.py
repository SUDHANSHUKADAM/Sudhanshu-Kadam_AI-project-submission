"""
Task 5 - Flight Risk Identification.

Definition (from the spec): a flight risk is any employee who has sent 4 or
more NEGATIVE emails within a rolling 30-day window, irrespective of calendar
month boundaries and irrespective of their sentiment score.

Implementation: for each employee we take the sorted dates of their negative
emails and slide a window. Because the messages are date-stamped (day
resolution), an employee is flagged if there exists any index i in their sorted
negative-email dates such that:

        date[i + 3] - date[i] <= 30 days

i.e. four negative emails fall inside a 30-day span. This is an exact rolling
check, not a per-month bucket.
"""
from __future__ import annotations

import pandas as pd

WINDOW = pd.Timedelta(days=30)
MIN_NEGATIVE = 4


def identify_flight_risks(
    df: pd.DataFrame,
    employee_col: str = "from",
    date_col: str = "date",
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """Return a table of flagged employees with the triggering window details."""
    neg = df[df[sentiment_col] == "Negative"].copy()
    neg[date_col] = pd.to_datetime(neg[date_col])

    records = []
    for employee, grp in neg.groupby(employee_col):
        dates = grp[date_col].sort_values().reset_index(drop=True)
        flagged = False
        first_window = None
        for i in range(len(dates) - (MIN_NEGATIVE - 1)):
            span = dates[i + MIN_NEGATIVE - 1] - dates[i]
            if span <= WINDOW:
                flagged = True
                first_window = (dates[i], dates[i + MIN_NEGATIVE - 1])
                break
        if flagged:
            records.append(
                {
                    "employee": employee,
                    "total_negative_emails": len(dates),
                    "window_start": first_window[0].date(),
                    "window_end": first_window[1].date(),
                    "negatives_in_window": MIN_NEGATIVE,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "employee",
                "total_negative_emails",
                "window_start",
                "window_end",
                "negatives_in_window",
            ]
        )
    return pd.DataFrame(records).sort_values("employee").reset_index(drop=True)
