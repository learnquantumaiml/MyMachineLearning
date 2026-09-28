import pandas as pd

# Simulate request logs from a deployed model endpoint, similar to what we would pull from Azure Monitor or App Insights for a review
data = {
    "request_id": range(1, 13),
    "model_version": ["v1","v1","v1","v2","v2","v2","v1","v2","v1","v2","v1","v2"],
    "region": ["EU","NA","EU","APAC","EU","NA","NA","APAC","EU","EU","APAC","NA"],
    "latency_ms": [220, 340, 210, 180, 195, 360, 300, 175, 205, 190, 165, 355],
    "status": ["success","success","success","success","fail","success","fail","success","success","success","success","fail"],
    "tokens_used": [512, 890, 480, 300, 275, 910, 860, 310, 495, 305, 288, 875],
}
df = pd.DataFrame(data)
print("---- raw request log ----")
print(df)

# groupby("model_version") splits the table into a v1 group and a v2 group.
# ["latency_ms"] picks out just that column, .mean() averages it within each group.
# change "model_version" to "region" to slice the same data a different way
print("\n---- average latency per model version ----")
print(df.groupby("model_version")["latency_ms"].mean().round(1))

# status is text ("success" / "fail"), not a number, so you can't average it directly.
# (s == "success") turns each row into True or False, and the average of a
# True/False column is exactly the success rate, since True counts as 1 and False as 0
print("\n---- success rate per model version ----")
success_rate = df.groupby("model_version")["status"].apply(lambda s: (s == "success").mean())
print((success_rate * 100).round(1))

# .agg() lets you compute several different statistics in one line instead of
# calling groupby() over and over. each keyword becomes a new column name.
print("\n---- multiple stats at once with agg() ----")
summary = df.groupby("model_version").agg(
    avg_latency_ms=("latency_ms", "mean"),
    max_latency_ms=("latency_ms", "max"),
    total_requests=("request_id", "count"),
    avg_tokens=("tokens_used", "mean"),
)
print(summary.round(1))

# passing a LIST to groupby() groups by both columns together,
# giving you latency broken down by version AND region at the same time
print("\n---- breakdown by version AND region ----")
by_region = df.groupby(["model_version", "region"])["latency_ms"].mean().round(1)
print(by_region)
