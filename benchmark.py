import os
import time
import json
import logging
import asyncio
import httpx
import numpy as np
from datetime import datetime, timezone
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark")

# Configuration from environment or defaults
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8001")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "frigate_faces_v10")

async def benchmark_api_endpoints():
    results = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Define endpoints from UI api.ts
        endpoints = [
            ("/health", "Health Check"),
            ("/v1/stats", "Dashboard Stats"),
            ("/v1/faces/subjects", "List All Subjects"),
            ("/v1/subjects?limit=50", "List Subjects (Paginated)"),
            ("/v1/events/recognition?limit=50", "Recognition Events"),
            ("/v1/events/recognition/cameras", "List Cameras"),
            ("/v1/events/recognition/feedback_stats", "Feedback Stats"),
            ("/v1/search_history?limit=50", "Search History"),
            ("/v1/search_history/stats?match_threshold=0.8", "Search Stats"),
            ("/v1/cross_check/visitors_vs_employees?limit=50", "Cross-Check Report"),
            ("/metrics", "Prometheus Metrics"),
            ("/v1/faces/subjects", "Subject Names Only"),
        ]

        for path, description in endpoints:
            t0 = time.perf_counter()
            try:
                resp = await client.get(f"{API_BASE_URL}{path}")
                latency = (time.perf_counter() - t0) * 1000
                results[path] = {
                    'description': description,
                    'status': resp.status_code,
                    'latency_ms': latency,
                    'ok': resp.status_code == 200
                }
            except Exception as e:
                results[path] = {'description': description, 'error': str(e), 'ok': False}

    return results

def get_qdrant_stats():
    try:
        client = QdrantClient(url=QDRANT_URL)
        collection_info = client.get_collection(collection_name=QDRANT_COLLECTION)
        
        # Measure a simple search latency
        t0 = time.perf_counter()
        # Dummy search with a random vector (assuming 512 dimensions for buffalo_l)
        client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=np.random.rand(512).tolist(),
            limit=1
        )
        search_latency = (time.perf_counter() - t0) * 1000

        return {
            'collection': QDRANT_COLLECTION,
            'points_count': collection_info.points_count,
            'status': collection_info.status,
            'indexed_vectors': collection_info.indexed_vectors_count,
            'search_latency_ms': search_latency,
            'ok': True
        }
    except Exception as e:
        return {'error': str(e), 'ok': False}

async def run_benchmark():
    print("Starting Benchmark...")
    print("-" * 30)
    
    api_results = await benchmark_api_endpoints()
    qdrant_results = get_qdrant_stats()
    
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'api_performance': api_results,
        'qdrant_performance': qdrant_results
    }
    
    # Save report
    report_file = f"benchmark_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nBenchmark completed. Report saved to {report_file}")
    print("\nSummary:")
    print(f"Qdrant Points: {qdrant_results.get('points_count', 'N/A')}")
    print(f"Qdrant Search Latency: {qdrant_results.get('search_latency_ms', 'N/A'):.2f}ms")
    
    print("\nAPI Performance (All UI-related endpoints):")
    print(f"{'Path':<50} | {'Latency':<10} | {'Status':<5}")
    print("-" * 70)
    for path, data in api_results.items():
        if data.get('ok'):
            print(f"{path:<50} | {data['latency_ms']:>8.2f}ms | OK")
        else:
            print(f"{path:<50} | {'FAILED':>8} | ERR")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
