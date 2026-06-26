"""
Task 1 - Sentiment Labeling (primary, reproducible method).

We label each message as Positive / Negative / Neutral using **TextBlob**.
TextBlob's pattern-based analyzer returns a polarity score in the range
[-1.0, +1.0]. It is deterministic and runs fully offline, so the graded results
are perfectly reproducible from raw data with no API keys.

For a "preferred LLM" alternative, see `sentiment_llm.py`. The two paths are
interchangeable: both add a single `sentiment` column to the dataframe.

Decision rule (sign of the polarity, computed on Subject + body combined):
    polarity  >  0  -> Positive
    polarity  <  0  -> Negative
    polarity == 0  -> Neutral
"""
from __future__ import annotations

import pandas as pd
from textblob import TextBlob


def _classify(polarity: float) -> str:
    if polarity > 0:
        return "Positive"
    if polarity < 0:
        return "Negative"
    return "Neutral"


def score_text(text: str) -> float:
    """Return the TextBlob polarity score for a single piece of text."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return TextBlob(text).sentiment.polarity


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
    df["polarity"] = combined.map(score_text)
    df[out_col] = df["polarity"].map(_classify)
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
    print(label_dataframe(sample)[["Subject", "polarity", "sentiment"]])
