
import faiss
import numpy as np
import pandas as pd

def load_fraud_index(index_path, metadata_path):
    index = faiss.read_index(index_path)
    metadata = pd.read_csv(metadata_path)
    return index, metadata

def find_similar_fraud_cases(query_features_scaled, index, metadata, k=3):
    """query_features_scaled: 1D numpy array over the CURATED similarity
    feature set, already scaled with similarity_scaler.pkl. Returns top-k
    similar historical fraud cases with a bounded [0,100] similarity score.
    Uses exponential decay (not linear) so it degrades gracefully for
    outlier distances instead of clipping to 0."""
    query = query_features_scaled.reshape(1, -1).astype("float32")
    distances, indices = index.search(query, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        case = metadata.iloc[idx].to_dict()
        # Bounded exponential decay: dist=0 -> 100%, grows smoothly toward 0
        # rather than a linear formula that clips negative distances to 0
        similarity_pct = 100 * np.exp(-dist / 10)
        results.append({**case, "distance": float(dist), "similarity_pct": round(float(similarity_pct), 1)})
    return results
