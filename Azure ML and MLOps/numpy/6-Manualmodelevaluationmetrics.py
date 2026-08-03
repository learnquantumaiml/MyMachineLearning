import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1])

# Confusion matrix components via boolean masking
tp = np.sum((y_true == 1) & (y_pred == 1))
tn = np.sum((y_true == 0) & (y_pred == 0))
fp = np.sum((y_true == 0) & (y_pred == 1))
fn = np.sum((y_true == 1) & (y_pred == 0))

precision = tp / (tp + fp)              # of predicted positives, how many correct
recall = tp / (tp + fn)                 # of actual positives, how many caught
f1 = 2 * (precision * recall) / (precision + recall)

print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
print(f"Precision={precision:.2f} Recall={recall:.2f} F1={f1:.2f}")
