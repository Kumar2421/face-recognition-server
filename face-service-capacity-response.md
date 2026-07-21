# Face Service capacity — response to EI 1000 events/min request

**To:** Event-Ingestion (EI) team
**From:** Face Service maintainer
**Date:** 2026-07-11
**Instance under test:** `192.168.0.192:8001` (2 uvicorn workers, 1× RTX 5060 Ti 16 GB)
**Endpoint:** `POST /v1/faces/privacy_extract`, `recognition: true`

---

## 0. ⚠️ Read this first — payload mismatch

Your spec (§2) describes **1080p full frames, ~400–600 KB base64**. The image supplied for this test (`0b50f28699aefbba.jpg`) is a **165×256, ~12 KB single-face crop** — model ≈ 72 ms.

**These behave very differently.** All measured numbers below are for the **12 KB crop**. For your stated **1080p ~500 KB frames**, detection alone is ~200 ms, so the ceiling is **much lower** (see §6b). **Confirm which payload production actually sends** — it changes the answer.

---

## 1. Direct answer

- **With the tested 12 KB crop:** ✅ **Yes** — sustains **17 and 20 rps** with **p95 ≈ 80 ms, 0 % errors**. Comfortable. **Max sustainable ≈ 30 rps** (p95 < 400 ms); collapses beyond ~32 rps.
- **With 1080p ~500 KB frames (your stated spec):** ⚠️ **Borderline** — projected ceiling **~12–16 rps**. 17 rps is at the edge; 20 rps would breach p95 and trip your brownout. **Fix available** (§6c): `det_size 320` or client-side downscale → back above 20 rps.

---

## 6. Results — measured (12 KB crop, 120 s sustained each)

| Counter | @ 17 rps | @ 20 rps | @ MAX (~30 rps) |
|---|---|---|---|
| Achieved throughput (req/s) | **17.0** | **20.0** | **29.9** |
| Latency p50 (ms) | 72 | 68 | 186 |
| Latency p95 (ms) | **81** | **75** | 359 |
| Latency p99 (ms) | 101 | 96 | 452 |
| Error rate (%) | **0.0** | **0.0** | 0.0 |
| Timeout / 429 rate (%) | 0.0 | 0.0 | 0.0 |
| Max concurrent in-flight served | 4 | 3 | 9 |
| GPU utilization (%) | 54 | 66 | 91 |
| CPU utilization (load avg, 18 vCPU) | 4.1 | 3.9 | 7.4 |
| Memory headroom | GPU 10.3 / 16 GB · host RAM ample | GPU 10.8 / 16 GB | GPU **15.0 / 16 GB (near full)** |

**Beyond the ceiling** (for reference): at 40 & 50 rps the system **could not sustain** — throughput plateaued at **~32 rps**, queue backed up, **p95 → 9–11 s** (0 % errors, just slow). This is the saturation point.

---

## 6a. Prose answers

- **Max sustainable `privacy_extract` rps (this 12 KB crop):** **~30 rps** with p95 < 400 ms; hard ceiling ~32 rps (throughput plateaus, latency explodes past it).
- **Shared budget?** **Yes.** `privacy_extract` shares **one GPU + one concurrency gate** with recognize / search / blur / compare / enroll. It is **not** additive — the ~30 rps ceiling (small images) is the **combined** budget across all image endpoints. Your "~30–35 rps total" figure is **correct for small crops**, **optimistic for 1080p frames.**
- **Backpressure at overload:** **queues, then slows** (bounded FIFO, cap 512, gate 12 concurrent). It returns **503 Retry-After** (not 429) only when the queue exceeds 512 deep OR a request waits > 60 s. Below that it just gets slower (as seen at 40 rps: 0 % errors, p95 9 s). **It does not 429.**
- **Internal breakdown per call** (from `X-Model-Ms` / `X-Queue-Ms` headers we added):
  - 12 KB crop: model p50 **62 ms** (detect + crop + blur + encode), queue ~2 ms until saturation.
  - 1080p frame: model ~**200 ms** (detection on 1920 px dominates).
- **Knobs to raise throughput on current hardware:**
  | Knob | Effect |
  |------|--------|
  | `BUFFALO_DET_SIZE=640 → 320` | detection ~2× faster → **~2× rps** on large frames (small-face recall drops — test) |
  | Client downscales frame to ~640 px before send | smaller upload + ~2× faster detect |
  | Add a 2nd GPU | +~30 rps (small) / +~12 rps (1080p) per GPU |
  | More workers on THIS GPU | **not possible** — 2 instances already use 15 / 16 GB (GPU full) |

---

## 6b. Projection for your stated 1080p ~500 KB frames

Not yet load-tested with a real 1080p frame (test image was a 12 KB crop). Based on single-request measurement (model ~200 ms for 1080p) and the same saturation behaviour:

| Payload | model / call | projected sustainable rps | 17 rps? | 20 rps? |
|---------|-------------:|--------------------------:|:-------:|:-------:|
| 12 KB crop (tested) | 62 ms | **~30** | ✅ | ✅ |
| **1080p ~500 KB (your spec)** | **~200 ms** | **~12–16** | ⚠️ edge | ❌ |

**If production sends 1080p frames, this instance likely misses the 17 rps target** at p95 < 1 s.

## 6c. Fix to guarantee 17–20 rps on 1080p

1. **`BUFFALO_DET_SIZE=320`** (server) — detection ~2× faster → 1080p ceiling ~24–30 rps → comfortably meets 20 rps. Cost: small/far-face recall; validate on your frames.
2. **Client downscales to ~640 px** before upload — also cuts your 500 KB body to ~80 KB (faster upload).
3. Either one moves 1080p from "borderline" to "comfortable."

---

## 7. Bottom line for your governor

- **If you send small crops (~12 KB):** set governor to **~25 rps** (safe under the ~30 ceiling). 1000/min ✅ met with wide margin.
- **If you send 1080p ~500 KB frames:** as-is ceiling **~12–16 rps** → 1000/min **not safely met**; apply `det_size=320` (or client 640 px downscale) → then **20 rps ✅**.
- **Backpressure:** it slows/queues and 503s past the queue cap — never 429. Keep your brownout threshold (12 s) well above our p95; at target load p95 is 80 ms (crop) — no risk.

**Single most important number:** max sustainable ≈ **30 rps (small crops)** / **~12–16 rps (1080p, as-is)** / **~24–30 rps (1080p with det_size=320)** on this single-GPU instance.

*Reproduce: `IMG=<image> RPS=<rate> DURATION=<sec> URL=http://192.168.0.192:8001 python3 ratetest.py`*
