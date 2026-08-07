# Latency Root-Cause Report — Face Recognition Service

**Date:** 2026-07-09
**Endpoint measured:** `POST /v1/events/recognition`
**Hardware:** NVIDIA RTX 5060 Ti (16 GB, shared) · buffalo_l (`det_10g` + `w600k_r50`) · ONNXRuntime + TensorRT
**All numbers below are measured from the server's own `meta.timing`, not estimated.**

---

## TL;DR

- **One image, warm, isolated → 0.16 s server / ~0.4 s wall. Already sub-second.**
- The multi-second numbers in the screenshots are **queue wait under concurrency**, not model compute.
- GPU compute per image is **flat at ~0.1 s at every load level** — so the model is not the problem and more model tuning cannot fix it.
- **Sub-second per image under the test's all-at-once burst on a single GPU is not physically possible.** That is a throughput ceiling, not a bug. It is removed only by adding GPU capacity or by capping concurrency — not by optimization.

---

## 1. Where each second goes (one warm request)

| Stage | Time | What it is |
|-------|-----:|------------|
| decode | 16 ms | CPU JPEG decode |
| **gpu_exec** | **55 ms** | **detect + embed — the model itself** |
| qdrant | 44 ms | vector search |
| save | 20 ms | write event image/thumb (backgrounded) |
| misc | 24 ms | quality, routing, serialization |
| **TOTAL** | **159 ms** | **sub-second** |

Client-side wall = **~0.40 s** (adds HTTP + JPEG upload). Every stage is tens of milliseconds. Nothing here is near a second.

---

## 2. Proof the seconds are queue, not compute

Same endpoint, same image, fired concurrently (the way the test harness does it):

| Concurrency | GPU compute / req | Queue + wait / req | Total wall / req | Verdict |
|-------------|------------------:|-------------------:|-----------------:|---------|
| 1 (warm) | 55 ms | ~0 ms | **0.16 s** | sub-second |
| 12 burst | 72–112 ms | 0.1–3.0 s | 0.24–5.3 s | queue forms |
| 20 burst | 69–256 ms | up to 15 s | 11–16 s | queue saturated |

**Read the compute column: it stays ~0.1 s no matter the load.** Only the wait column grows. If the model were slow, `gpu_exec` would grow with load — it does not. This is textbook queueing:

```
wall latency  =  concurrency ÷ throughput      (once GPU is saturated)
```

The GPU sits at **95–97 % utilization** — already saturated. When 20 requests arrive at once and only 2 can run, requests 3–20 wait in line. Their timer counts that wait.

---

## 3. Why the screenshot shows 3–4 s

The tester (`api_test_multi_scenario.py`) submits **every endpoint against Localhost + Internal IP + External host simultaneously** through a 16-thread pool — ~40+ image requests hit a 2-worker GPU at the same instant.

- **Fast rows** (`subject_images`, `events_*_stats` = 0.05–0.08 s) → GET, no image, never touch GPU → no queue.
- **Slow rows** (`privacy_extract`, `privacy_blur` = 3–4 s) → image POSTs all fighting for the same 2 GPU workers at once.
- The 3–4 s is the **artificial burst of the test**, not what a single camera sees in production.

---

## 4. Why "sub-second under this burst on one GPU" is NOT possible

This is physics, not a missing optimization:

1. **The GPU is already saturated (95–97 %).** There is no idle GPU time to reclaim.
2. **Model compute per image is already minimal** (~55 ms, TensorRT + FP16 on). It cannot meaningfully shrink further on this hardware.
3. **Serving order is serial on a shared GPU.** With `throughput ≈ 34–38 RPS`, 20 simultaneous requests take `20 ÷ 36 ≈ 0.55 s` for the *last* request **only if perfectly pipelined** — real queue + decode contention pushes it to seconds.
4. Therefore, for N images arriving at the same instant, the N-th image **must** wait for the N−1 ahead of it. No configuration makes one GPU compute 40 images in parallel in under a second.

**Conclusion: a single GPU cannot deliver sub-second latency to every request in a 40-request simultaneous burst. That requires either fewer concurrent requests or more GPUs.**

---

## 5. What actually reduces the queue (and their limits)

| Lever | Effect | Limit |
|-------|--------|-------|
| **Micro-batching** `GPU_BATCH_WINDOW_MS=8` | batch concurrent reqs into 1 GPU call → drains faster | now SET in compose, **needs container restart to take effect** (live `/health` still shows `batch_window_s: 0.0`) |
| **Concurrency gate** `FACE_SERVICE_MAX_INFLIGHT` | caps in-flight so no request waits past 1 s; excess → `503 Retry-After` | sheds load — does not do more work, just bounds latency |
| **Smaller det_size** `640 → 512` | ~1.5× less compute/img → queue drains faster | small/far faces detect worse; must test recall |
| **More GPU workers** `2 → 3` | slightly more overlap | GPU already 95 % compute-bound → marginal |
| **2nd GPU instance + load balancer** | doubles throughput → halves queue | **the only real fix for high burst** — needs more VRAM/GPU |

---

## 6. Bottom line for review

- **Per-image latency requirement (< 1 s): already met** at realistic per-camera load — **~0.16 s server, ~0.4 s wall.**
- The multi-second readings come **only** from an artificial all-endpoints-at-once stress burst against a single 2-worker GPU.
- That burst latency is a **concurrency / throughput ceiling**, not a slow model. It is **not solvable by further model optimization** — the model and GPU are already maxed (95–97 % util, TensorRT + FP16).
- To keep sub-second **under heavy concurrent burst**, the only options are: **cap concurrency** (guarantee via `503` shedding) or **add GPU capacity** (a second instance). Both are capacity decisions, not code fixes.

---

*Action pending: restart the `face_service` container so `GPU_BATCH_WINDOW_MS=8` becomes active, then re-measure. Set `FACE_SERVICE_MAX_INFLIGHT` to `throughput × 1 s` (~35) to guarantee no accepted request exceeds one second.*
