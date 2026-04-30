import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

def compare_visitors(visitor_ids):
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    collection = os.environ.get("QDRANT_COLLECTION", "frigate_faces_v21")
    client = QdrantClient(url=url)
    
    print(f"Connecting to Qdrant at {url}, collection: {collection}")
    
    visitor_data = {}
    
    for vid in visitor_ids:
        print(f"\nFetching data for {vid}...")
        res = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="subject_id", match=MatchValue(value=vid))]
            ),
            with_payload=True,
            with_vectors=True,
            limit=100
        )
        points = res[0]
        print(f"Found {len(points)} points for {vid}")
        
        if points:
            visitor_data[vid] = [np.array(p.vector) for p in points]

    if len(visitor_data) < 2:
        print("Not enough data found for comparison.")
        return

    v1, v2 = visitor_ids[0], visitor_ids[1]
    vecs1 = visitor_data[v1]
    vecs2 = visitor_data[v2]

    print(f"\n--- Similarity Analysis: {v1} vs {v2} ---")
    
    similarities = []
    for i, vec1 in enumerate(vecs1):
        for j, vec2 in enumerate(vecs2):
            # Cosine similarity assuming normalized vectors (dot product)
            # If not normalized: np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            sim = np.dot(vec1, vec2)
            similarities.append(sim)
    
    if similarities:
        print(f"Max Similarity: {max(similarities):.4f}")
        print(f"Min Similarity: {min(similarities):.4f}")
        print(f"Mean Similarity: {np.mean(similarities):.4f}")
        
        # Self-similarity for baseline
        print(f"\n--- Baseline: {v1} Self-Similarity ---")
        self_sims = []
        for i in range(len(vecs1)):
            for j in range(i + 1, len(vecs1)):
                self_sims.append(np.dot(vecs1[i], vecs1[j]))
        if self_sims:
            print(f"Average {v1} self-similarity: {np.mean(self_sims):.4f}")

if __name__ == "__main__":
    compare_visitors(["visiter-767", "visiter-19"])
