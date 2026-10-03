import pandas as pd

# Two separate tables you'd commonly pull apart in an MLOps workflow:
# one has experiment run metadata (like from MLflow tracking),
# the other has the evaluation metrics logged for each run.
runs = pd.DataFrame({
    "run_id": ["run_001", "run_002", "run_003", "run_004"],
    "model_type": ["random_forest", "xgboost", "random_forest", "logistic_regression"],
    "team": ["platform", "platform", "research", "research"],
})

metrics = pd.DataFrame({
    "run_id": ["run_001", "run_002", "run_003", "run_005"],  # run_005 has no matching metadata row
    "accuracy": [0.87, 0.91, 0.84, 0.79],
    "f1_score": [0.85, 0.90, 0.81, 0.76],
})

print("---- runs table ----")
print(runs)
print("\n---- metrics table ----")
print(metrics)

# on="run_id" tells pandas which column to match rows on.
# how="inner" keeps ONLY rows where run_id exists in both tables,
# so run_004 (no metrics yet) and run_005 (no metadata) both disappear
print("\n---- inner join: only runs present in BOTH tables ----")
inner = pd.merge(runs, metrics, on="run_id", how="inner")
print(inner)

# how="left" keeps every row from the FIRST table (runs), and fills in
# NaN wherever there's no match in the second table (metrics)
print("\n---- left join: keep every row from 'runs', fill blanks where no metrics exist ----")
left = pd.merge(runs, metrics, on="run_id", how="left")
print(left)

# how="outer" keeps every row from BOTH tables, filling NaN on whichever
# side is missing. nothing gets silently dropped with this option
print("\n---- outer join: keep everything from both sides, even run_004 and run_005 ----")
outer = pd.merge(runs, metrics, on="run_id", how="outer")
print(outer)

# now that the tables are joined, you can ask a question that needed BOTH
# of them: which run had the best f1_score, for each model type?
print("\n---- best run per model type, after joining ----")
best = inner.sort_values("f1_score", ascending=False).groupby("model_type").first()
print(best[["run_id", "f1_score"]])
