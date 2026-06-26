"""
Task 2 - Exploratory Data Analysis and visualizations.

Each function saves a PNG into the visualization/ folder and returns the path,
so the notebook can both display and persist the figures.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
VIZ_DIR = "visualization"


def _save(fig, name: str) -> str:
    os.makedirs(VIZ_DIR, exist_ok=True)
    path = os.path.join(VIZ_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def data_overview(df: pd.DataFrame) -> dict:
    return {
        "records": len(df),
        "columns": list(df.columns),
        "employees": df["from"].nunique(),
        "date_min": str(pd.to_datetime(df["date"]).min().date()),
        "date_max": str(pd.to_datetime(df["date"]).max().date()),
        "missing_values": int(df.isnull().sum().sum()),
    }


def plot_sentiment_distribution(df: pd.DataFrame) -> str:
    order = ["Positive", "Neutral", "Negative"]
    colors = {"Positive": "#2a9d8f", "Neutral": "#9aa0a6", "Negative": "#e76f51"}
    counts = df["sentiment"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(order, counts.values, color=[colors[o] for o in order])
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, f"{int(v)}", ha="center")
    ax.set_title("Distribution of Sentiment Labels")
    ax.set_ylabel("Number of messages")
    return _save(fig, "01_sentiment_distribution.png")


def plot_messages_over_time(df: pd.DataFrame) -> str:
    d = df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)
    monthly = d.groupby("month").size()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly.index, monthly.values, marker="o", color="#264653")
    ax.set_title("Message Volume Over Time (Monthly)")
    ax.set_ylabel("Messages")
    ax.set_xticks(range(len(monthly.index)))
    ax.set_xticklabels(monthly.index, rotation=45, ha="right")
    return _save(fig, "02_message_volume_over_time.png")


def plot_sentiment_over_time(df: pd.DataFrame) -> str:
    d = df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)
    pivot = (
        d.groupby(["month", "sentiment"]).size().unstack(fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    )
    colors = ["#2a9d8f", "#9aa0a6", "#e76f51"]
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.85)
    ax.set_title("Monthly Sentiment Composition")
    ax.set_ylabel("Messages")
    ax.set_xlabel("")
    ax.legend(title="Sentiment")
    return _save(fig, "03_sentiment_over_time.png")


def plot_net_sentiment_trend(df: pd.DataFrame) -> str:
    d = df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)
    d["val"] = d["sentiment"].map({"Positive": 1, "Negative": -1, "Neutral": 0})
    trend = d.groupby("month")["val"].sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in trend.values]
    ax.bar(trend.index, trend.values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Net Monthly Sentiment Score (whole organization)")
    ax.set_ylabel("Net score (Positive - Negative)")
    ax.set_xticklabels(trend.index, rotation=45, ha="right")
    return _save(fig, "04_net_sentiment_trend.png")


def plot_messages_per_employee(df: pd.DataFrame) -> str:
    counts = df["from"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#457b9d")
    ax.set_title("Messages per Employee")
    ax.set_xlabel("Messages")
    return _save(fig, "05_messages_per_employee.png")


def plot_sentiment_by_employee(df: pd.DataFrame) -> str:
    pivot = (
        df.groupby(["from", "sentiment"]).size().unstack(fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    colors = ["#2a9d8f", "#9aa0a6", "#e76f51"]
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", stacked=True, ax=ax, color=colors)
    ax.set_title("Sentiment Composition by Employee")
    ax.set_xlabel("Messages")
    ax.set_ylabel("")
    ax.legend(title="Sentiment")
    return _save(fig, "06_sentiment_by_employee.png")


def plot_message_length_distribution(df: pd.DataFrame) -> str:
    d = df.copy()
    d["length"] = (d["Subject"].fillna("") + " " + d["body"].fillna("")).str.len()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(d["length"], bins=40, color="#6d597a", ax=ax)
    ax.set_title("Message Length Distribution (characters)")
    ax.set_xlabel("Characters")
    return _save(fig, "07_message_length_distribution.png")


def run_all_eda(df: pd.DataFrame) -> list[str]:
    return [
        plot_sentiment_distribution(df),
        plot_messages_over_time(df),
        plot_sentiment_over_time(df),
        plot_net_sentiment_trend(df),
        plot_messages_per_employee(df),
        plot_sentiment_by_employee(df),
        plot_message_length_distribution(df),
    ]
