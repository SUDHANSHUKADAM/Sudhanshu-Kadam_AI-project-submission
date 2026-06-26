"""
Task 1 - Sentiment Labeling (primary, reproducible method).

We label each message as Positive / Negative / Neutral using VADER
(Valence Aware Dictionary and sEntiment Reasoner). VADER is a rule/lexicon
based model tuned for short, informal English text (exactly the register of
workplace email) and is fully deterministic and offline. That makes the
graded results perfectly reproducible from raw data with no API keys.

For a "preferred LLM" alternative, see `sentiment_llm.py`. The two paths are
interchangeable: both add a single `sentiment` column to the dataframe.

Decision rule (standard VADER thresholds on the compound score, computed on
Subject + body combined):
    compound >=  0.05  -> Positive
    compound <= -0.05  -> Negative
    otherwise          -> Neutral
"""
from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05

_analyzer = SentimentIntensityAnalyzer()


def _classify(compound: float) -> str:
    if compound >= POS_THRESHOLD:
        return "Positive"
    if compound <= NEG_THRESHOLD:
        return "Negative"
    return "Neutral"


def score_text(text: str) -> float:
    """Return the VADER compound score for a single piece of text."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _analyzer.polarity_scores(text)["compound"]


def label_dataframe(
    df: pd.DataFrame,
    subject_col: str = "Subject",
    body_col: str = "body",
    out_col: str = "sentiment",
) -> pd.DataFrame:
    """Add a `sentiment` column (Positive/Negative/Neutral) to the dataframe.

    The message used for scoring is the Subject and body concatenated, so the
    headline of the email contributes to its sentiment.
    """
    df = df.copy()
    combined = (
        df[subject_col].fillna("").astype(str)
        + ". "
        + df[body_col].fillna("").astype(str)
    )
    df["compound"] = combined.map(score_text)
    df[out_col] = df["compound"].map(_classify)
    return df


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "Subject": ["Great job team!", "This is unacceptable", "Meeting at 3pm"],
            "body": [
                "Really proud of everyone, fantastic work.",
                "I am furious about the delays and broken promises.",
                "Please bring the quarterly figures.",
            ],
        }
    )
    print(label_dataframe(sample)[["Subject", "compound", "sentiment"]])
