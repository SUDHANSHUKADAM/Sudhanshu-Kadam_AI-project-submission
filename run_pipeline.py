"""Run the full pipeline end-to-end and write all outputs + visualizations."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src import sentiment, scoring, ranking, flight_risk, modeling, eda

RAW = "data/test.csv"

# --- Task 1: load + label ----------------------------------------------------
df = pd.read_csv(RAW, parse_dates=["date"])
df = sentiment.label_dataframe(df)
df.to_csv("data/labeled_emails.csv", index=False)
print("Labeled:", df.shape)
print(df["sentiment"].value_counts().to_dict())

# --- Task 2: EDA -------------------------------------------------------------
overview = eda.data_overview(df)
viz_paths = eda.run_all_eda(df)
print("Overview:", overview)
print("Saved", len(viz_paths), "figures")

# --- Task 3: monthly scores --------------------------------------------------
monthly = scoring.monthly_scores(df)
monthly.to_csv("outputs/monthly_scores.csv", index=False)
print("Monthly score rows:", len(monthly))

# --- Task 4: ranking ---------------------------------------------------------
top_pos, top_neg = ranking.rank_employees(monthly)
overall = ranking.overall_ranking(monthly)
top_pos.to_csv("outputs/top_positive_by_month.csv", index=False)
top_neg.to_csv("outputs/top_negative_by_month.csv", index=False)
overall.to_csv("outputs/overall_ranking.csv", index=False)
print("\nOVERALL RANKING:")
print(overall.to_string(index=False))

# --- Task 5: flight risk -----------------------------------------------------
risks = flight_risk.identify_flight_risks(df)
risks.to_csv("outputs/flight_risks.csv", index=False)
print("\nFLIGHT RISKS:")
print(risks.to_string(index=False) if len(risks) else "None")

# --- Task 6: linear regression ----------------------------------------------
feats = modeling.build_features(df)
feats.to_csv("outputs/model_features.csv", index=False)
result = modeling.train_linear_model(feats)
print("\nMODEL METRICS:", result.metrics)
print("COEFFICIENTS:")
print(result.coefficients.to_string(index=False))

# model diagnostics plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(result.y_test, result.y_pred, alpha=0.6, color="#264653")
lo = min(result.y_test.min(), result.y_pred.min())
hi = max(result.y_test.max(), result.y_pred.max())
ax.plot([lo, hi], [lo, hi], "--", color="#e76f51")
ax.set_xlabel("Actual monthly score")
ax.set_ylabel("Predicted monthly score")
ax.set_title(f"Predicted vs Actual (R2={result.metrics['r2']:.3f})")
fig.tight_layout(); fig.savefig("visualization/08_pred_vs_actual.png", dpi=120); plt.close(fig)

resid = result.y_test.values - result.y_pred
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(result.y_pred, resid, alpha=0.6, color="#6d597a")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
ax.set_title("Residual Plot")
fig.tight_layout(); fig.savefig("visualization/09_residuals.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
c = result.coefficients.sort_values("coefficient")
ax.barh(c["feature"], c["coefficient"], color="#457b9d")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Linear Regression Coefficients")
fig.tight_layout(); fig.savefig("visualization/10_coefficients.png", dpi=120); plt.close(fig)

# --- persist a machine-readable summary for the README/report ---------------
summary = {
    "overview": overview,
    "sentiment_counts": df["sentiment"].value_counts().to_dict(),
    "overall_ranking": overall.to_dict(orient="records"),
    "flight_risks": risks["employee"].tolist(),
    "model_metrics": result.metrics,
    "coefficients": result.coefficients.to_dict(orient="records"),
}
with open("outputs/summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\nWrote outputs/summary.json")
