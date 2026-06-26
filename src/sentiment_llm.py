"""
Task 1 - Sentiment Labeling (OPTIONAL "preferred LLM" path).

The problem statement says an LLM is *preferred* for sentiment labeling. This
module provides that path. It is OPTIONAL and not run by default, because:

  * It requires a paid API key and network access.
  * LLM outputs are non-deterministic, which works against the assignment's
    explicit reproducibility requirement.

The committed results in this repo were produced with the deterministic VADER
labeler in `sentiment.py`. This module is provided so a reviewer can re-label
with an LLM if desired; the schema it produces is identical (a `sentiment`
column with values Positive / Negative / Neutral).

Supported providers (configure via a .env file, see .env.example):
    LLM_PROVIDER=anthropic   ANTHROPIC_API_KEY=...
    LLM_PROVIDER=openai      OPENAI_API_KEY=...

Usage:
    from src.sentiment_llm import label_dataframe_llm
    df = label_dataframe_llm(df)            # labels every row via the LLM
"""
from __future__ import annotations

import os
import time
from typing import List

import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional
    pass

VALID = {"Positive", "Negative", "Neutral"}

SYSTEM_PROMPT = (
    "You are a precise sentiment classifier for workplace emails. "
    "Classify the overall sentiment of the message as exactly one word: "
    "Positive, Negative, or Neutral. Respond with only that single word."
)


def _clean_label(raw: str) -> str:
    raw = (raw or "").strip().strip(".").capitalize()
    return raw if raw in VALID else "Neutral"


def _label_anthropic(texts: List[str], model: str) -> List[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    out = []
    for t in texts:
        msg = client.messages.create(
            model=model,
            max_tokens=5,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": t[:4000]}],
        )
        out.append(_clean_label(msg.content[0].text))
        time.sleep(0.05)
    return out


def _label_openai(texts: List[str], model: str) -> List[str]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out = []
    for t in texts:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=5,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": t[:4000]},
            ],
        )
        out.append(_clean_label(resp.choices[0].message.content))
        time.sleep(0.05)
    return out


def label_dataframe_llm(
    df: pd.DataFrame,
    subject_col: str = "Subject",
    body_col: str = "body",
    out_col: str = "sentiment",
) -> pd.DataFrame:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    df = df.copy()
    combined = (
        df[subject_col].fillna("").astype(str)
        + ". "
        + df[body_col].fillna("").astype(str)
    ).tolist()

    if provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        df[out_col] = _label_anthropic(combined, model)
    elif provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        df[out_col] = _label_openai(combined, model)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    return df
