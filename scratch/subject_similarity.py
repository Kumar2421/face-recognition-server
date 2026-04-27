import argparse
import os
from typing import Any

import numpy as np
from qdrant_client import QdrantClient


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a) + 1e-9)
    return (a / n).astype(np.float32)


def _mean_subject_vector(
    client: QdrantClient,
    *,
    collection: str,
    subject_id: str,
    max_points: int,
    batch: int,
) -> tuple[np.ndarray | None, int]:
    sid = str(subject_id or "").strip()
    if not sid:
        return None, 0

    try:
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        flt = Filter(must=[FieldCondition(key="subject_id", match=MatchValue(value=sid))])
    except Exception:
        flt = None

    next_cur: Any = None
    scanned = 0
    total = 0
    v_sum: np.ndarray | None = None

    while scanned < int(max_points):
        limit = int(min(int(batch), int(max_points) - scanned))
        if limit <= 0:
            break

        kwargs: dict[str, Any] = {
            "collection_name": collection,
            "limit": limit,
            "with_payload": False,
            "with_vectors": True,
        }
        if flt is not None:
            kwargs["scroll_filter"] = flt
        if next_cur is not None:
            kwargs["offset"] = next_cur

        batch_points, next_cur2 = client.scroll(**kwargs)
        next_cur = next_cur2
        if not batch_points:
            break

        for p in batch_points:
            scanned += 1
            try:
                payload = getattr(p, "payload", None) or {}
                if flt is None:
                    if str(payload.get("subject_id") or "").strip() != sid:
                        continue

                vec_raw = getattr(p, "vector", None)
                if vec_raw is None:
                    continue
                vec = np.asarray(vec_raw, dtype=np.float32).reshape(-1)
                if vec.size == 0:
                    continue
            except Exception:
                continue

            if v_sum is None:
                v_sum = vec.astype(np.float32)
            else:
                v_sum = v_sum + vec
            total += 1

        if next_cur is None:
            break

    if v_sum is None or total <= 0:
        return None, 0

    mean = (v_sum / float(total)).astype(np.float32)
    return _l2_normalize(mean), int(total)


def _neighbors(client: QdrantClient, *, collection: str, vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    try:
        hits = client.search(
            collection_name=collection,
            query_vector=np.asarray(vec, dtype=np.float32).reshape(-1).tolist(),
            limit=int(top_k),
            with_payload=True,
        )
    except Exception:
        return out

    for h in hits or []:
        try:
            payload = getattr(h, "payload", None) or {}
            sid = str(payload.get("subject_id") or "").strip()
            score = float(getattr(h, "score", 0.0) or 0.0)
            if not sid:
                continue
            out.append((sid, score))
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject_id_a")
    ap.add_argument("subject_id_b")
    ap.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", "frigate_faces_v13"))
    ap.add_argument("--max-points", type=int, default=int(os.environ.get("SUBJECT_SIM_MAX_POINTS", "200") or "200"))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("SUBJECT_SIM_BATCH", "256") or "256"))
    ap.add_argument("--neighbors", type=int, default=10)
    args = ap.parse_args()

    max_points = max(1, min(int(args.max_points), 5000))
    batch = max(1, min(int(args.batch), 2048))
    neighbors_k = max(1, min(int(args.neighbors), 50))

    client = QdrantClient(url=str(args.url))
    collection = str(args.collection)

    vec_a, count_a = _mean_subject_vector(
        client,
        collection=collection,
        subject_id=str(args.subject_id_a),
        max_points=max_points,
        batch=batch,
    )
    vec_b, count_b = _mean_subject_vector(
        client,
        collection=collection,
        subject_id=str(args.subject_id_b),
        max_points=max_points,
        batch=batch,
    )

    print(f"url={args.url}")
    print(f"collection={collection}")
    print(f"subject_id_a={args.subject_id_a} count={count_a}")
    print(f"subject_id_b={args.subject_id_b} count={count_b}")

    if vec_a is None:
        print("mean_a=None")
    if vec_b is None:
        print("mean_b=None")

    if vec_a is not None and vec_b is not None:
        cos_sim = float(np.dot(vec_a.reshape(-1), vec_b.reshape(-1)))
        print(f"cosine_similarity={cos_sim:.6f}")

    if vec_a is not None:
        print("neighbors_a:")
        for sid, sim in _neighbors(client, collection=collection, vec=vec_a, top_k=neighbors_k):
            print(f"  {sim:.6f}  {sid}")

    if vec_b is not None:
        print("neighbors_b:")
        for sid, sim in _neighbors(client, collection=collection, vec=vec_b, top_k=neighbors_k):
            print(f"  {sim:.6f}  {sid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
