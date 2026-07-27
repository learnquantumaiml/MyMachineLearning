import numpy as np

# Simulate a feature matrix: 5 samples, 3 features (e.g., age, income, score)
data = np.array([
    [25, 50000, 0.8],
    [40, 80000, 0.6],
    [35, 65000, 0.9],
    [50, 120000, 0.4],
    [28, 45000, 0.7]
], dtype=float)

# --- Z-score standardization ---
mean = data.mean(axis=0)          # column-wise mean → shape (3,)
std = data.std(axis=0)            # column-wise std dev → shape (3,)
z_scaled = (data - mean) / std    # broadcasting subtracts/divides per column

# --- Min-Max normalization (0 to 1 range) ---
col_min = data.min(axis=0)
col_max = data.max(axis=0)
minmax_scaled = (data - col_min) / (col_max - col_min)

print("Z-score scaled:\n", z_scaled)
print("Min-Max scaled:\n", minmax_scaled)
