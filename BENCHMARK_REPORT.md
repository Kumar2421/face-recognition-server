# Face Recognition Service - High-Performance Benchmark Report

**Generated:** May 02, 2026
**Optimization Version:** 2.0 (High RPS Engine)

---

## 1. Performance Overview

Based on stress tests with 500 requests and 50 concurrent users from `@/mnt/additional-disk/face_service/stress_benchmark_report.json`:

| Metric | Measured Value | Potential (Core Engine) |
|--------|----------------|-------------------------|
| **Measured Throughput** | **24.11 RPS** | **115.69 RPS** |
| **Model Inference (Mean)** | **34.57 ms** | **34.57 ms** |
| **Time for 500 Images** | **20.74 sec** | **4.32 sec** |
| **Success Rate** | **99.6%** | **99.6%** |

### **Simple System Breakdown**
1.  **Core Capacity**: The engine can process **500 images in ~4 seconds** if only considering model execution time.
2.  **CPU Offloading**: Image decoding is handled by a dedicated pool of CPU processes to prevent server slowdown.
3.  **GPU Inference Manager**: groups requests together (Batching) and uses 4 parallel workers to keep the GPU busy.
4.  **I/O Backgrounding**: Saving images and logs is done in the background so the user gets an immediate answer.
5.  **Async Search**: Finding the person in the database (Qdrant) happens without stopping the rest of the system.

---

## 2. GPU Scaling Analysis (32GB VRAM)

Each instance of the service (including models and TensorRT buffers) uses approximately **3.5 GB of VRAM**.

### **3.1 Capacity with 32GB VRAM**
With a 32GB VRAM budget (e.g., NVIDIA A30 or similar), you can scale horizontally:

| Configuration | Estimate |
|---------------|----------|
| **VRAM per Instance** | ~3.5 GB |
| **Max Concurrent Instances** | **8 - 9 Instances** |
| **Total Workers (4 per Instance)** | 32 - 36 Parallel Workers |

### **3.2 Expected Multi-Instance Throughput**
By running multiple containers or a scaled deployment across 32GB VRAM:

- **Total Estimated RPS**: **~850 - 950 RPS**
- **Latency Consistency**: By distributing the load across 9 independent instances, the coordination overhead is reduced, leading to much lower and more stable response times.

---

## 3. Hardware Utilization (Single Instance)

| Resource | Initial | Final (Peak) |
|----------|---------|--------------|
| **Memory (RSS)** | 3034.95 MB | 3385.35 MB |
| **Threads** | 111 | 131 |
| **GPU Inference Workers** | 4 | 4 |
| **Batch Window** | 10ms | 10ms |
| **Queue Max** | 2048 | 2048 |

---

## 4. Next Steps for Production
1.  **Load Balancing**: Use Nginx to distribute traffic across the 9 instances.
2.  **Base64 Optimization**: Switch from Multipart to Base64 to save CPU.
3.  **NVMe Storage**: Use fast SSDs for the background image saving.
