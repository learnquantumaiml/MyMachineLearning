## This code takes a table with missing values (NaN), finds the average of each column while ignoring the missing values, and then fills the missing spots with those column averages.
import numpy as np

# Feature matrix with missing values represented as NaN
data = np.array([
    [1.0, 2.0, np.nan],
    [4.0, np.nan, 6.0],
    [7.0, 8.0, 9.0],
    [np.nan, 11.0, 12.0]
])

# nan-aware mean ignores NaN when computing column averages
col_means = np.nanmean(data, axis=0)

# Build a boolean mask of where values are missing
missing_mask = np.isnan(data)

# Impute: copy the array, then fill NaN positions with the column mean
imputed = data.copy()
inds = np.where(missing_mask)          # (row_indices, col_indices) of NaNs
imputed[inds] = np.take(col_means, inds[1])  # map each NaN to its column's mean

print("Missing mask:\n", missing_mask)
print("Imputed data:\n", imputed)
