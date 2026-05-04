import asyncio
import httpx
import time
import os
import json
import statistics
import base64
from pathlib import Path

# --- Configuration ---
BASE_URL = "http://localhost:8001"
API_KEY = "fs_9f2b8a71c4d04e5e9b3d8a7c6b5a4f3e"
IMAGES_DIR = "/mnt/additional-disk/face_service/downloaded_images"
NUM_REQUESTS = 500
CONCURRENCY = 50  # Increased concurrency to push 100 RPS

async def get_system_health(client):
    try:
        resp = await client.get(f"{BASE_URL}/health")
        return resp.json()
    except Exception as e:
        print(f"Health check failed: {e}")
        return {}

async def benchmark_worker_with_data(worker_id, image_data_list, stats, semaphore, client):
    for img_bytes in image_data_list:
        async with semaphore:
            try:
                start_time = time.perf_counter()
                
                files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
                data = {"top_k": "5"}
                headers = {"x-api-key": API_KEY}
                
                response = await client.post(
                    f"{BASE_URL}/v1/faces/recognize_upload",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=60.0
                )
                end_time = time.perf_counter()
                
                latency = (end_time - start_time) * 1000
                if response.status_code == 200:
                    res_json = response.json()
                    # Capture model_ms from metadata if returned by server
                    model_ms = res_json.get("meta", {}).get("timing", {}).get("model_ms", 0)
                    stats.append({
                        "success": True,
                        "latency_ms": latency,
                        "model_ms": model_ms,
                        "status_code": response.status_code
                    })
                else:
                    stats.append({
                        "success": False,
                        "latency_ms": latency,
                        "status_code": response.status_code,
                        "error": response.text
                    })
            except Exception as e:
                stats.append({
                    "success": False,
                    "latency_ms": 0,
                    "error": str(e)
                })

async def benchmark_worker(worker_id, image_paths, stats, semaphore, client):
    for img_path in image_paths:
        async with semaphore:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                start_time = time.perf_counter()
                
                files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
                data = {"top_k": "5"}
                headers = {"x-api-key": API_KEY}
                
                response = await client.post(
                    f"{BASE_URL}/v1/faces/recognize_upload",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=60.0
                )
                end_time = time.perf_counter()
                
                latency = (end_time - start_time) * 1000
                if response.status_code == 200:
                    res_json = response.json()
                    # Capture model_ms from metadata if returned by server
                    model_ms = res_json.get("meta", {}).get("timing", {}).get("model_ms", 0)
                    stats.append({
                        "success": True,
                        "latency_ms": latency,
                        "model_ms": model_ms,
                        "status_code": response.status_code
                    })
                else:
                    stats.append({
                        "success": False,
                        "latency_ms": latency,
                        "status_code": response.status_code,
                        "error": response.text
                    })
            except Exception as e:
                stats.append({
                    "success": False,
                    "latency_ms": 0,
                    "error": str(e)
                })

async def run_benchmark():
    # Collect image paths
    all_images = list(Path(IMAGES_DIR).rglob("*.jpg")) + list(Path(IMAGES_DIR).rglob("*.png"))
    if not all_images:
        print(f"No images found in {IMAGES_DIR}!")
        return
    
    # Pre-resize images to 640px (BUFFALO_DET_SIZE default) to simulate optimized client side
    print(f"Pre-resizing {len(all_images)} images for benchmark...")
    import cv2
    import numpy as np
    resized_images = []
    for img_path in all_images:
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                scale = 640.0 / float(max(h, w))
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                is_success, buffer = cv2.imencode(".jpg", img)
                if is_success:
                    resized_images.append(buffer.tobytes())
        except Exception:
            continue
    
    if not resized_images:
        print("Failed to pre-resize any images.")
        return

    # Repeat images to reach NUM_REQUESTS
    test_data = (resized_images * (NUM_REQUESTS // len(resized_images) + 1))[:NUM_REQUESTS]
    
    print(f"Starting Stress Benchmark: {NUM_REQUESTS} requests, Concurrency: {CONCURRENCY}")
    
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONCURRENCY)) as client:
        # Initial health check
        start_health = await get_system_health(client)
        
        stats = []
        semaphore = asyncio.Semaphore(CONCURRENCY)
        
        # Split work among virtual workers
        chunk_size = len(test_data) // CONCURRENCY
        tasks = []
        
        start_bench = time.perf_counter()
        
        for i in range(CONCURRENCY):
            worker_data = test_data[i*chunk_size : (i+1)*chunk_size]
            if i == CONCURRENCY - 1:
                worker_data = test_data[i*chunk_size:]
            tasks.append(benchmark_worker_with_data(i, worker_data, stats, semaphore, client))
            
        await asyncio.gather(*tasks)
        
        end_bench = time.perf_counter()
        total_time = end_bench - start_bench
        
        # Final health check
        end_health = await get_system_health(client)
        
        # --- Report Generation ---
        successes = [s for s in stats if s.get("success")]
        latencies = [s["latency_ms"] for s in successes]
        model_times = [s["model_ms"] for s in successes if s.get("model_ms")]
        
        # Calculate Potential Throughput (Core Engine Capacity)
        mean_model_ms = statistics.mean(model_times) if model_times else 0
        # Formula: (1000ms / mean_model_ms) * workers
        potential_rps = (1000.0 / mean_model_ms) * 4 if mean_model_ms > 0 else 0
        # Time for 500 images at potential RPS
        potential_time_for_500 = 500.0 / potential_rps if potential_rps > 0 else 0

        report = {
            "summary": {
                "total_requests": NUM_REQUESTS,
                "successful_requests": len(successes),
                "failed_requests": NUM_REQUESTS - len(successes),
                "total_time_sec": round(total_time, 2),
                "throughput_rps": round(NUM_REQUESTS / total_time, 2),
                "concurrency": CONCURRENCY,
                "target_rps": 100
            },
            "potential_performance": {
                "core_engine_rps": round(potential_rps, 2),
                "time_for_500_images_sec": round(potential_time_for_500, 2),
                "note": "This excludes network and decoding overhead"
            },
            "latency_ms": {
                "min": round(min(latencies), 2) if latencies else 0,
                "max": round(max(latencies), 2) if latencies else 0,
                "mean": round(statistics.mean(latencies), 2) if latencies else 0,
                "median": round(statistics.median(latencies), 2) if latencies else 0,
                "p95": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) >= 20 else 0
            },
            "model_performance_ms": {
                "mean_model_ms": round(statistics.mean(model_times), 2) if model_times else 0,
                "median_model_ms": round(statistics.median(model_times), 2) if model_times else 0
            },
            "resource_usage": {
                "initial": start_health.get("system", {}),
                "final": end_health.get("system", {}),
                "gpu_status": end_health.get("gpu_inference", {})
            }
        }
        
        output_file = "stress_benchmark_report.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"\nBenchmark Complete!")
        print(f"Throughput: {report['summary']['throughput_rps']} RPS")
        print(f"Mean Latency: {report['latency_ms']['mean']} ms")
        print(f"Failed Requests: {report['summary']['failed_requests']}")
        print(f"Report saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
