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


---

# Addendum — Measured Re-Benchmark (2026-07-08)

> Live measurements on the actual deployment. **Corrects several stale figures above.**
> Hardware: NVIDIA GeForce RTX 5060 Ti (16 GB VRAM, shared with other services), 18 vCPU `QEMU Virtual CPU 2.5+`. Model: InsightFace buffalo_l (`det_10g` + `w600k_r50`) via ONNXRuntime + TensorRT.

## A. Corrections to figures above

| Claim above | Reality (measured) |
|-------------|--------------------|
| VRAM per instance **3.5 GB** | **~7.3 GB** (7496 MiB) — TensorRT + CUDA arena ~2× the model |
| GPU workers **4**, batch window **10 ms** | Runtime env is **2 workers**, batch window **0 ms** (batching disabled) |
| Max instances on 32 GB **8–9** | At 7.3 GB/instance: **~4 instances / 32 GB**, **~2 / 16 GB** |
| RSS **~3.0 GB** | Host RSS **~1.26 GB** (python), docker mem **~1.87 GiB** |

## B. GPU throughput (real end-to-end HTTP, `recognize_upload`)

500 requests, concurrency 50, against the live service:

| Metric | Value |
|--------|-------|
| **Throughput** | **34–38 RPS** |
| **Mean latency** | ~1450 ms (under conc 50) |
| **GPU compute util** | **95–97 %** (saturated) |
| **GPU VRAM** | **7496 MiB, constant** (no per-request growth, no leak) |
| **Host CPU** | ~180 % (≈1.8 cores), peak 240 % |
| **Host RAM** | ~1.87 GiB, **flat under load** |

GPU compute is the ceiling (95%+ at only 2 workers). CPU and RAM have headroom → more RPS needs **more GPU**, not more CPU/RAM.

## C. CPU-only inference (ONNXRuntime `CPUExecutionProvider`)

Single-image detect+embed, no GPU:

| Config | Latency / image | Single-stream RPS |
|--------|-----------------|-------------------|
| 18 threads (all cores) | 463 ms | 2.2 |
| 1 thread (`OMP_NUM_THREADS=1`) | 497 ms | 2.0 |
| 1 pinned core | 713 ms | 1.4 |

**Threads barely help** (18-thread ≈ 1-thread) → the model is **memory-bandwidth bound**, not compute bound. Adding worker processes contends for the same bandwidth → sub-linear scaling.

**CPU sustained throughput on this box ≈ 2–8 RPS.**

### Caveats
- This is a **QEMU virtual CPU** (weak/no AVX-512) — pessimistic. On bare-metal Xeon/EPYC, buffalo_l CPU is typically **~80–150 ms/img → ~8–15 RPS single-stream, more across cores**.
- GPU is shared (other services resident) — GPU numbers are slightly suppressed too.

### Conclusion: GPU vs CPU
- **GPU ≈ 34–38 RPS. CPU ≈ 2–8 RPS on this VM. GPU is ~5–15× faster here.**
- **Stay on GPU.** CPU is a dev/fallback path only. If ever forced onto CPU: INT8 quantization (2–4× on CPU) + bare metal + 1 intra-op thread × N processes — still will not beat GPU.

## D. Stability (load-shedding, post crash-fix)

After the crash-hardening changes (concurrency gate, event-loop offload, decode-pool self-heal):

| Metric under conc-50 storm | Result |
|----------------------------|--------|
| Server crashes | **0** |
| HTTP 500 errors | **0** |
| Requests shed as **503** (gate, `FACE_SERVICE_MAX_INFLIGHT=32`) | 240–272 |
| Successful **200** | 227–260 |
| `/health` latency **during** full load | **~0.1 s** (idle 0.02 s) |
| GPU queue depth | stays 0 (gate sheds before queue fills) |

Server sheds excess load with `503 Retry-After` instead of OOM-crashing. Event loop stays responsive throughout.

## E. FAQ — plain-language answers

### Q1. Why did GPU VRAM jump from 3.5 GB to 7.5 GB? Is that a leak?

**No leak.** 7.3 GB (7496 MiB) is the true steady-state, measured — and it stays **constant** before, during, and after load (no per-request growth). The old 3.5 GB figure was wrong: it counted only the model weights.

The extra VRAM is:
- **TensorRT engine buffers** — TensorRT builds optimized GPU kernels + a workspace; large.
- **CUDA memory arena** — ONNXRuntime pre-allocates a VRAM pool at startup and holds it (it does not shrink back).
- Model weights are only a small slice.

*Can be reduced* (trading some speed) via `ORT_CUDA_MEMORY_LIMIT_MB` and `ORT_CUDA_ARENA_EXTEND_STRATEGY=kSameAsRequested` in `embedders/buffalo_l.py`.

### Q2. The report says CPU only gets ~2 RPS — is that right?

Yes on **this server**, but that number is misleading without context:

- This machine runs on a **virtual CPU** (`QEMU Virtual CPU 2.5+`) — a software-emulated CPU, not real silicon.
- The face model relies on fast SIMD math instructions (**AVX2 / AVX-512**). The emulated CPU hides/fakes these, so ONNXRuntime falls back to slow generic math → **3–5× slower**.
- Proof: 18 threads ran no faster than 1 thread — an emulated-CPU signature (real CPUs scale with SIMD/cores).

| | per image | RPS |
|---|-----------|-----|
| **This VM** (QEMU, no AVX) | ~500 ms | ~2 |
| **Bare-metal** (real Xeon/EPYC w/ AVX-512) | ~80–150 ms | ~8–15 |

Same model, same code — the only difference is emulated CPU vs real physical CPU.

**Important:** this handicap affects **CPU only**. The **GPU is passed through** to the VM as the real RTX 5060 Ti, so all GPU numbers (34–38 RPS) are genuine.

### Bottom line
GPU VRAM at 7.5 GB is normal and stable (no leak). CPU does ~2 RPS on this VM (~8–15 on bare metal) versus 34–38 RPS on GPU. **Stay on GPU; CPU is a dev/fallback path only.**
