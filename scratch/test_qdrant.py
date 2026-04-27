import os
import json
from qdrant_client import QdrantClient

def test_qdrant():
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    collection = os.environ.get("QDRANT_COLLECTION", "frigate_faces_v10")
    client = QdrantClient(url=url)
    
    print(f"Connecting to Qdrant at {url}, collection: {collection}")
    
    # Try exact MatchValue (prefix is not a standard param for MatchValue in all versions)
    # Most reliable for Discovering is scrolling with NO filter and counting unique IDs
    # But let's try to count how many 'employee-' points exist
    
    scanned = 0
    unique_employees = set()
    next_offset = None
    
    while scanned < 50000:
        res = client.scroll(
            collection_name=collection,
            limit=1000,
            with_payload=True,
            with_vectors=False,
            offset=next_offset
        )
        batch, next_offset = res
        if not batch:
            break
            
        for p in batch:
            scanned += 1
            sid = p.payload.get("subject_id", "")
            if sid.startswith("employee-"):
                unique_employees.add(sid)
        
        if not next_offset:
            break
            
    print(f"Scanned {scanned} points")
    print(f"Found {len(unique_employees)} unique employees: {list(unique_employees)[:10]}...")
    
    # Now try with a filter if we can guess the syntax
    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchText
        # Try MatchText which often works as a partial match
        res = client.scroll(
            collection_name=collection,
            limit=10,
            scroll_filter=Filter(
                must=[FieldCondition(key="subject_id", match=MatchText(text="employee"))]
            )
        )
        print(f"MatchText('employee') returned {len(res[0])} points")
    except Exception as e:
        print(f"Filter test failed: {e}")

if __name__ == "__main__":
    test_qdrant()
