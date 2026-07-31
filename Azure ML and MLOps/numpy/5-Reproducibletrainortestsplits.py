import numpy as np

rng = np.random.default_rng(seed=42)   # seeded generator = reproducible results

n_samples = 20
X = rng.normal(size=(n_samples, 4))     # 20 rows, 4 features
y = rng.integers(0, 2, size=n_samples)  # binary labels

# Shuffle indices, not the data directly, so X and y stay aligned
indices = rng.permutation(n_samples)

split_point = int(0.8 * n_samples)      # 80/20 split
train_idx, test_idx = indices[:split_point], indices[split_point:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train label balance:", np.bincount(y_train))
