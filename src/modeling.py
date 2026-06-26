"""
Task 6 - Predictive Modeling (Linear Regression).

We model each employee's monthly sentiment score as a function of features
engineered from their messaging behaviour that month. The unit of analysis is
one (employee, month) row.

Features (independent variables):
    messages          - number of emails sent that month (message frequency)
    avg_msg_length    - average character length of those emails
    total_words       - total word count across the month
    avg_words         - average words per email
    pos_ratio         - share of the month's emails that were Positive
    neg_ratio         - share of the month's emails that were Negative

Target (dependent variable):
    score             - the monthly sentiment score from Task 3

We split into train/test (80/20), fit sklearn LinearRegression, and report
R^2, MAE and RMSE plus the standardized-ish coefficients for interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Behavioural features only - the variables the spec suggests (message
# frequency, length, average length, word count). These are independent of the
# sentiment labels, so this is the honest "can behaviour predict sentiment"
# model.
BEHAVIORAL_FEATURES = [
    "messages",
    "avg_msg_length",
    "total_words",
    "avg_words",
]

# Full feature set additionally includes the monthly share of positive/negative
# messages. These are strong predictors but partly circular (they are derived
# from the same labels that define the score), so we report this model as a
# comparison and discuss the caveat in the report.
FEATURES = BEHAVIORAL_FEATURES + ["pos_ratio", "neg_ratio"]


def build_features(
    df: pd.DataFrame,
    employee_col: str = "from",
    date_col: str = "date",
    subject_col: str = "Subject",
    body_col: str = "body",
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """Build the (employee, month) feature matrix with the target `score`."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.to_period("M").astype(str)
    text = df[subject_col].fillna("").astype(str) + " " + df[body_col].fillna("").astype(str)
    df["msg_length"] = text.str.len()
    df["word_count"] = text.str.split().map(len)
    df["val"] = df[sentiment_col].map({"Positive": 1, "Negative": -1, "Neutral": 0})

    feats = (
        df.groupby([employee_col, "month"])
        .agg(
            score=("val", "sum"),
            messages=("val", "size"),
            avg_msg_length=("msg_length", "mean"),
            total_words=("word_count", "sum"),
            avg_words=("word_count", "mean"),
            pos_ratio=(sentiment_col, lambda s: (s == "Positive").mean()),
            neg_ratio=(sentiment_col, lambda s: (s == "Negative").mean()),
        )
        .reset_index()
        .rename(columns={employee_col: "employee"})
    )
    return feats


@dataclass
class ModelResult:
    model: LinearRegression
    metrics: Dict[str, float]
    coefficients: pd.DataFrame
    X_test: pd.DataFrame = field(repr=False)
    y_test: pd.Series = field(repr=False)
    y_pred: np.ndarray = field(repr=False)


def train_linear_model(
    feats: pd.DataFrame, features: list[str] | None = None, random_state: int = 42
) -> ModelResult:
    features = features or FEATURES
    X = feats[features]
    y = feats["score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": rmse,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    coefs = (
        pd.DataFrame({"feature": features, "coefficient": model.coef_})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )
    return ModelResult(model, metrics, coefs, X_test, y_test, y_pred)
