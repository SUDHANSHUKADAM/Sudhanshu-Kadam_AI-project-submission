# Employee Sentiment Analysis — Final Report

**Author:** Sudhanshu Kadam
**Dataset:** `test.csv` — 2,191 employee emails, 10 senders, January 2010 – December 2011

---

## 1. Approach and Methodology

The objective was to take a raw, unlabeled corpus of workplace emails and turn it
into actionable measures of employee sentiment and engagement. The work is split
into six tasks, each implemented as a small, focused module under `src/` and
orchestrated by a single notebook (`Employee_Sentiment_Analysis.ipynb`) so the
entire pipeline is reproducible from raw data to final outputs.

**Data.** The dataset has four columns — `Subject`, `body`, `date`, `from` — with
2,191 rows, 10 unique senders, no missing values, and clean daily-resolution
dates spanning 24 consecutive months.

**Sentiment labeling (Task 1).** Each message (Subject concatenated with body) is
scored with **TextBlob**, whose pattern-based analyzer returns a polarity score
in the range [-1.0, +1.0]. We map the sign of the polarity to a label:
`> 0` → Positive, `< 0` → Negative, `== 0` → Neutral. TextBlob was chosen as the
primary method because it is deterministic and runs entirely offline, which
directly satisfies the assignment's reproducibility requirement — a reviewer
regenerates identical labels with no API keys or paid services. A
**preferred-LLM alternative** (Anthropic or OpenAI) is included in
`src/sentiment_llm.py` and produces the same `sentiment` column; it is optional
because LLM outputs are non-deterministic and incur cost, which would undermine
exact reproducibility. The labeling criterion is fully documented and
reproducible either way.

**Tooling.** Python, with pandas for data handling, TextBlob for labeling,
matplotlib/seaborn for visualization, and scikit-learn for the regression model,
in line with the brief's guidance to use sklearn for AI modeling.

---

## 2. Key Findings from the EDA

- **Clean, well-structured data.** 2,191 complete records, 10 employees, zero
  missing values, 24 consecutive months.
- **Positive-leaning corpus.** Approximately **61% Positive, 25% Neutral, 14%
  Negative**. This is typical of routine corporate email and means negative
  messages — the minority class — carry the most diagnostic value.
- **Stable volume and sentiment over time.** Monthly message counts are fairly
  even, and net monthly sentiment stays positive across the whole period, with a
  persistent negative undercurrent. Because volume is steady, monthly scores are
  comparable across the series.
- **Per-employee variation.** Employees differ in how much negative email they
  send, which is precisely the variation the flight-risk task is designed to
  surface.

Supporting figures (in `visualization/`): sentiment distribution, monthly volume,
monthly sentiment composition, net monthly sentiment trend, messages per
employee, sentiment by employee, and message-length distribution.

---

## 3. Employee Scoring and Ranking

**Scoring (Task 3).** Each message contributes **+1 (Positive), −1 (Negative),
0 (Neutral)**. Scores are summed per `(employee, calendar-month)`. Grouping by
month means each month's score is computed independently — it resets at the start
of every month by construction. This yields 240 rows (10 employees × 24 months),
confirming every employee was active every month.

**Ranking (Task 4).** For each month we produce the top-3 positive employees
(highest scores) and top-3 negative employees (lowest scores). The sort key is
score first, then employee name alphabetically as a tiebreaker — descending for
the positive list and ascending for the negative list.

**Overall leaderboard** (summed across all 24 months):

| Rank | Employee | Total Score |
|------|----------|-------------|
| 1 | john.arnold@enron.com | 125 |
| 2 | lydia.delgado@enron.com | 125 |
| 3 | sally.beck@enron.com | 109 |
| 4 | johnny.palmer@enron.com | 105 |
| 5 | patti.thompson@enron.com | 102 |
| 6 | bobette.riner@ipgdirect.com | 100 |
| 7 | don.baughman@enron.com | 97 |
| 8 | eric.bass@enron.com | 97 |
| 9 | kayne.coulter@enron.com | 89 |
| 10 | rhonda.denton@enron.com | 61 |

**Top 3 positive (overall):** john.arnold, lydia.delgado, sally.beck.
**Top 3 negative (overall):** rhonda.denton, kayne.coulter, don.baughman.

Per-month winners rotate (full lists in `outputs/`), but the overall ordering is
stable. Importantly, every employee's total is net-positive, so a low rank
reflects *relatively* less positive email rather than an outright negative
profile.

---

## 4. Flight Risk Identification

**Criterion (Task 5).** A flight risk is any employee who sent **four or more
negative emails within a rolling 30-day window**, independent of calendar-month
boundaries and independent of their sentiment score. The detection slides a
30-day window over each employee's chronologically sorted negative-email dates;
an employee is flagged the first time four negatives fall inside one window. The
implementation records the first triggering window (`window_start`,
`window_end`) for auditability.

**Outcome.** All ten employees are flagged at least once:

| Employee | Total negative emails | First triggering window |
|----------|----------------------:|-------------------------|
| bobette.riner@ipgdirect.com | 32 | 2010-10-21 → 2010-11-10 |
| don.baughman@enron.com | 25 | 2010-02-13 → 2010-03-07 |
| eric.bass@enron.com | 34 | 2010-03-19 → 2010-04-15 |
| john.arnold@enron.com | 31 | 2010-02-26 → 2010-03-26 |
| johnny.palmer@enron.com | 31 | 2010-02-06 → 2010-02-25 |
| kayne.coulter@enron.com | 24 | 2011-11-26 → 2011-12-26 |
| lydia.delgado@enron.com | 42 | 2010-06-17 → 2010-06-24 |
| patti.thompson@enron.com | 34 | 2010-04-05 → 2010-05-02 |
| rhonda.denton@enron.com | 32 | 2010-08-10 → 2010-08-28 |
| sally.beck@enron.com | 32 | 2010-01-23 → 2010-02-22 |

**Caveat.** Because the threshold is an absolute count rather than a rate,
employees who send more email overall are mechanically more likely to be flagged
— here every employee qualifies. In practice this flag should be paired with a
rate-based check, and sustained streaks rather than one-off spikes should drive
any retention action.

---

## 5. Predictive Model — Linear Regression

**Setup (Task 6).** The unit of analysis is one `(employee, month)` row (240
rows). The target is the monthly sentiment score. Features are engineered from
that month's messaging behaviour. We use an 80/20 train/test split
(`random_state=42`) and scikit-learn's `LinearRegression`, evaluated with R²,
MAE, and RMSE.

Two models are fitted:

1. **Behavioural model** — the features the brief suggests: message frequency
   (`messages`), `avg_msg_length`, `total_words`, `avg_words`. These are
   independent of the labels, so this is the honest test of whether behaviour
   predicts sentiment.
2. **Full model** — additionally includes `pos_ratio` and `neg_ratio` (the
   monthly share of positive/negative emails). Strong predictors, but partly
   circular since they derive from the same labels that define the score; the
   model is reported as a comparison and read with that caveat.

**Results (held-out test set):**

| Model | R² | MAE | RMSE |
|-------|----|-----|------|
| Behavioural | 0.51 | 1.63 | 2.35 |
| Full | 0.80 | 1.00 | 1.49 |

**Interpretation.**

- In the behavioural model, **message frequency is the dominant driver** —
  employees who send more email in a month tend to have higher net scores,
  because most messages are positive. Message length contributes little.
- In the full model, `neg_ratio` (large negative coefficient) and `pos_ratio`
  (positive coefficient) dominate, as expected; that lift is largely mechanical
  and should not be read as a new behavioural insight.
- **Takeaway:** behaviour alone — chiefly volume — explains a meaningful share of
  monthly sentiment (R² ≈ 0.51), while the sentiment mix is what really moves the
  score. Prediction error is modest (MAE around 1 score point on the full model).

Diagnostics (predicted-vs-actual, residuals, coefficients) are saved in
`visualization/`.

---

## 6. Conclusions and Recommendations

- Baseline engagement is healthy and stable: a consistently positive-leaning
  corpus with no data-quality problems.
- The actionable signal is concentrated in negative-email streaks. Every
  employee crosses the rolling 30-day threshold at least once; the accounts to
  prioritize are those with the highest negative volume and the tightest
  windows, with attention to *sustained* rather than one-off negativity and an
  awareness of the volume bias in the absolute-count rule.
- Monthly sentiment is quantifiable and forecastable from simple behavioural
  features, with message volume the leading indicator.
- **Operationally:** re-run this pipeline monthly, track the flagged employees'
  negative-email windows over time, and complement the flight-risk count with a
  rate-normalized version before taking retention action.

---

## 7. Reproducibility

All results in this report were generated by `run_pipeline.py` (or equivalently
by running the notebook end-to-end). TextBlob labeling is deterministic and the
regression uses a fixed random seed, so the labels, scores, rankings, flight-risk
flags, and model metrics reproduce exactly. Intermediate tables are in
`outputs/`; figures are in `visualization/`.
