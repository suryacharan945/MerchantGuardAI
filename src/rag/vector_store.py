import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

class PolicyVectorStore:
    def __init__(self, model_name=MODEL_NAME):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def build_index(self, corpus):
        self.metadata = corpus
        texts = [f"{doc['title']}: {doc['text']}" for doc in corpus]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        
        # Inner-product index on normalized vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        return self.index.ntotal

    def save(self, index_path, metadata_path):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def search(self, query_text, top_k=2):
        if self.index is None:
            raise ValueError("Index is not loaded or built.")
        query_emb = self.encoder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_emb.astype(np.float32), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = dict(self.metadata[idx])
                item["similarity_score"] = float(score)
                results.append(item)
        return results
