# Employee Sentiment Analysis

Analysis of an unlabeled corpus of **2,191 employee emails** (10 employees,
Jan 2010 – Dec 2011) to label sentiment, score and rank employees, flag
flight risks, and model sentiment trends with linear regression.

**Author:** Sudhanshu Kadam

---

## Summary of Findings

### Top 3 Positive Employees (overall, summed monthly scores)
| Rank | Employee | Total Score |
|------|----------|-------------|
| 1 | john.arnold@enron.com | 125 |
| 2 | lydia.delgado@enron.com | 125 |
| 3 | sally.beck@enron.com | 109 |

### Top 3 Negative Employees (overall, lowest summed scores)
| Rank | Employee | Total Score |
|------|----------|-------------|
| 1 | rhonda.denton@enron.com | 61 |
| 2 | kayne.coulter@enron.com | 89 |
| 3 | don.baughman@enron.com | 97 |

> Per-month top-3 lists are in `outputs/top_positive_by_month.csv` and
> `outputs/top_negative_by_month.csv`. Even the lowest-ranked employees have
> net-positive totals — the ranking is *relative*.

### Flight Risks (≥ 4 negative emails in a rolling 30-day window) — all 10 flagged
- bobette.riner@ipgdirect.com
- don.baughman@enron.com
- eric.bass@enron.com
- john.arnold@enron.com
- johnny.palmer@enron.com
- kayne.coulter@enron.com
- lydia.delgado@enron.com
- patti.thompson@enron.com
- rhonda.denton@enron.com
- sally.beck@enron.com

Full detail with the first triggering window per employee: `outputs/flight_risks.csv`.

### Key Insights & Recommendations
- **The corpus is positive-leaning** (~61% Positive, ~25% Neutral, ~14%
  Negative) and stable month to month — healthy baseline engagement.
- **Negatives are the signal worth watching.** Every employee hits the rolling
  30-day negative-email threshold at least once over the two years. Because the
  rule is an absolute count, high-volume senders are more exposed; pair the flag
  with a *rate*-based check before acting.
- **Monthly sentiment is predictable from behaviour** — message frequency is the
  dominant driver (behavioural model R² ≈ 0.51; ≈ 0.80 once sentiment-mix
  features are added).
- **Recommendation:** treat sustained negative streaks (not one-off spikes) as
  the retention signal, normalize the flight-risk count by email volume before
  acting, and re-run this pipeline monthly to track trends.

---

## Repository Structure
```
employee-sentiment-analysis/
├── Employee_Sentiment_Analysis.ipynb   # MAIN deliverable - all tasks, with narrative
├── run_pipeline.py                     # one-shot script: raw data -> all outputs
├── README.md                           # this file
├── REPORT.md / Final_Report.docx       # full methodology report
├── requirements.txt
├── .env.example                        # only needed for the optional LLM labeler
├── src/
│   ├── sentiment.py        # Task 1 - TextBlob labeling (primary)
│   ├── sentiment_llm.py    # Task 1 - optional LLM labeling (Anthropic/OpenAI)
│   ├── eda.py              # Task 2 - EDA + visualizations
│   ├── scoring.py          # Task 3 - monthly scores
│   ├── ranking.py          # Task 4 - employee ranking
│   ├── flight_risk.py      # Task 5 - rolling 30-day flight risk
│   └── modeling.py         # Task 6 - linear regression
├── data/
│   ├── test.csv            # raw input (provided dataset)
│   └── labeled_emails.csv  # generated: raw + sentiment column
├── outputs/                # generated tables (scores, rankings, risks, metrics)
└── visualization/          # generated PNG charts
```

---

## Setup

Requires **Python 3.9+**.

```bash
# 1. clone / unzip, then from the project root:
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt
```

---

## Usage

**Option A — run everything as a script** (regenerates every table and figure):
```bash
python run_pipeline.py
```

**Option B — open the notebook** (the main deliverable, with narrative):
```bash
jupyter notebook Employee_Sentiment_Analysis.ipynb
# then Run All
```

Both read `data/test.csv` and write to `data/`, `outputs/`, and `visualization/`.

### Optional: label with an LLM instead of TextBlob
The default labeler (TextBlob) needs no keys and is deterministic. To use the
preferred-LLM path instead:
```bash
cp .env.example .env        # then fill in your provider + API key
```
```python
from src.sentiment_llm import label_dataframe_llm
df = label_dataframe_llm(df_raw)   # adds the same `sentiment` column via an LLM
```

---

## Methodology (short version)

| Task | Method |
|------|--------|
| **Sentiment labeling** | TextBlob polarity on `Subject + body`; >0 Positive, <0 Negative, ==0 Neutral. Deterministic & offline for reproducibility. Optional LLM path provided. |
| **EDA** | Structure/missingness checks; distribution of labels; monthly volume and sentiment trends; per-employee composition; message-length distribution. |
| **Monthly score** | +1/−1/0 per message, summed per `(employee, month)`; resets each month by grouping. |
| **Ranking** | Per month, top-3 by score (desc for positive, asc for negative), employee name as alphabetical tiebreaker. |
| **Flight risk** | Rolling 30-day window over each employee's sorted negative-email dates; flag if 4 fall within 30 days. |
| **Linear regression** | Target = monthly score; behavioural features (frequency, length, word counts) + optional sentiment-mix features; 80/20 split; report R²/MAE/RMSE. |

Full write-up: see [`REPORT.md`](REPORT.md).

---

## Reproducibility notes
- TextBlob labeling is deterministic, so re-running yields identical labels,
  scores, rankings, and flight-risk flags.
- The regression uses `random_state=42` for a fixed train/test split.
- The committed `outputs/` and `visualization/` were generated by `run_pipeline.py`.
