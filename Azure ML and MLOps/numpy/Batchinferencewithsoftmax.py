import numpy as np

# Simulated logits from a model for 5 samples, 3 classes
logits = np.array([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.2],
    [1.0, 1.0, 1.0],
    [3.0, 0.2, 0.1],
    [0.1, 0.2, 3.5]
])

def softmax(x):
    # Subtract row max first for numerical stability (avoids overflow in exp)
    shifted = x - x.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)  # normalize per row

probs = softmax(logits)
predicted_classes = np.argmax(probs, axis=1)   # class with highest probability per row

print("Probabilities:\n", np.round(probs, 3))
print("Predicted classes:", predicted_classes)
