import numpy as np

categories = np.array(['low', 'medium', 'high', 'medium', 'low'])

# Get sorted unique categories → defines column order
unique_cats = np.unique(categories)          # ['high', 'low', 'medium']

# For each row, find the index of its category in unique_cats
cat_indices = np.searchsorted(unique_cats, categories)

# Build identity matrix and index into it → one-hot rows
one_hot = np.eye(len(unique_cats))[cat_indices]

print("Categories:", unique_cats)
print("One-hot matrix:\n", one_hot)
