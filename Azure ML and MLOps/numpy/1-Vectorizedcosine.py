import numpy as np

# query_vec: shape (d,)      - a single embedding vector for the query
# doc_matrix: shape (n, d)   - n document embeddings, each of dimension d

def cosine_similarity_batch(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Returns similarity scores between query_vec and every row in doc_matrix,
    computed in one vectorized pass instead of a per-document loop.
    """
    # Normalize the query vector to unit length (so dot product = cosine similarity)
    query_norm = query_vec / np.linalg.norm(query_vec)

    # Normalize every document row to unit length at once (norm along axis=1 = per-row)
    doc_norms = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True)

    # Dot product of the query against every row simultaneously -> shape (n,)
    # This one line replaces what would otherwise be a for-loop over n documents
    similarities = doc_norms @ query_norm

    return similarities


# Example usage:
query_vec = np.array([0.2, 0.8, 0.1])
doc_matrix = np.array([
    [0.1, 0.9, 0.0],
    [0.9, 0.1, 0.2],
    [0.2, 0.7, 0.2],
])

scores = cosine_similarity_batch(query_vec, doc_matrix)
top_doc_index = np.argsort(scores)[::-1][0]  # highest similarity first
print("Similarity scores:", scores)
print("Best matching document index:", top_doc_index)
