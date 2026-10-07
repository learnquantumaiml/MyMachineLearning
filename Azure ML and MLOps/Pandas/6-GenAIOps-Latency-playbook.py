import pandas as pd
import numpy as np

np.random.seed(42)  # keeps the random numbers identical every time this runs, remove this line for true randomness

# simulate a minute-by-minute latency log for a deployed model endpoint,
# the kind of raw telemetry you'd pull for a monitoring dashboard
timestamps = pd.date_range(start="2026-08-10 09:00", periods=180, freq="min")
latency = np.random.normal(loc=250, scale=30, size=180)  # centered around 250ms with some natural noise
# inject an artificial slow patch between minutes 100 and 120, simulating a real incident
latency[100:120] += 150

df = pd.DataFrame({"timestamp": timestamps, "latency_ms": latency.round(1)})
df = df.set_index("timestamp")  # once timestamp is the index, pandas can reason about it as actual time
print("---- first few rows of raw per-minute data ----")
print(df.head())

# resample() groups rows into time buckets, the way groupby() groups rows into categories.
# "h" means "by the hour", .mean() averages latency_ms within each hour
# try changing "h" to "30min" to see finer hourly buckets
print("\n---- resample to hourly averages ----")
hourly = df.resample("h").mean().round(1)
print(hourly)

# a rolling window looks BACKWARD over the last N rows and averages them,
# then slides forward one row at a time, smoothing out one-off noise while
# still keeping every original row (unlike resample, which collapses rows)
print("\n---- rolling 15-minute average (smooths out noise, keeps the same row count) ----")
df["rolling_15min_avg"] = df["latency_ms"].rolling(window=15).mean().round(1)
print(df.iloc[95:125])  # this slice covers the injected incident, watch the rolling average climb and fall

# a straightforward way to catch individual bad minutes: compare each value
# against a fixed number. change "threshold" below to match your own SLA
print("\n---- flag minutes where latency spiked above a threshold ----")
threshold = 350
spikes = df[df["latency_ms"] > threshold]
print(f"minutes above {threshold}ms:", len(spikes))
print(spikes.head())
