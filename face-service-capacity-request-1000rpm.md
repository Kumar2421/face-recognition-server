# Face Service capacity request — validating 1000 events/min

**To:** Face Service maintainer
**From:** Event-Ingestion (EI) team
**Date:** 2026-07-11
**Priority:** Blocker for the 1000 events/min throughput target
**Instance under test:** `http://192.168.0.192:8001` (the dedicated Face Service EI points at)

---

## 1. TL;DR — what we need from you

We are trying to sustain **1000 ingestion events per minute**. In our current flow, **one ingestion event = exactly one `POST /v1/faces/privacy_extract` call**. So 1000 events/min means the Face Service must sustain:

> **≈ 16.7 requests/second, sustained (not a burst), on `/v1/faces/privacy_extract`.**
> We govern our side at **20 rps / 16 concurrent** to leave headroom.

**Please confirm one of the following and send back the results table in §6:**

1. **Yes** — the current instance sustains **≥ 17 rps** (ideally 20 rps) of `privacy_extract` for ≥ 10 minutes within the latency/error limits in §4. Send the measured numbers.
2. **It exceeds that** — great; then please report the **actual maximum sustainable throughput** of the current system and the **counter values** at that point (see §6), so we can size our rate governor and worker fleet to it.
3. **No** — it tops out below 17 rps. Send the real ceiling + counters so we can plan (scale the FS, add instances, or accept a lower target).

---

## 2. How we call you (so the test matches production)

- **Endpoint:** `POST /v1/faces/privacy_extract`
- **Auth:** `x-api-key` header
- **Body (JSON):**
  ```json
  {
    "image_b64": "<base64 of a full camera frame>",
    "recognition": true,
    "branch": "<branch filter>"
  }
  ```
- **Image characteristics:** full-frame JPEG, ~**1920×1136**, ~300–450 KB raw → ~**400–600 KB base64** per request. (These are real Hikvision frames, typically 1 face per frame.)
- **Work per call (as we understand it):** detect + recognize + crop + privacy-blur of other faces, returned as per-face crops. One HTTP call does all of it.
- **Observed latency today:** ~**205 ms** per call at low load (from our pipeline dashboard). We need to know how that holds as concurrency rises.

## 3. The throughput requirement (numbers)

| Metric | Value |
|---|---|
| Target ingestion rate | **1000 events / min** |
| = Face Service request rate | **16.7 rps** on `privacy_extract` |
| Our governor refill (with headroom) | **20 rps** |
| Our governor max concurrent | **16** |
| Implied FS concurrency @ 205 ms | ~3.5 in-flight @ 17 rps, ~4.1 @ 20 rps (rises if latency rises) |
| Sustained duration to validate | **≥ 10 minutes** (~10,000 calls) |

## 4. Pass/fail limits (why they matter)

Our client has an **adaptive brownout**: if it observes the Face Service getting slow or erroring, it **automatically halves our request rate** (which would drop us to ~500/min and oscillate — i.e. unstable). To keep us at full rate, the FS must stay **well inside** these:

| Signal | Our trip threshold | What we actually want |
|---|---|---|
| Per-call latency (EWMA) | halve rate if **> 12 s** | p95 **< 1 s** at target load |
| Error / timeout rate | halve rate if **> 30%** over the last 20 calls | **< 2–5%** at target load |

So the real question isn't just "can it do 17 rps once" — it's **"can it hold 17–20 rps for 10 minutes with p95 < ~1 s and errors < ~5%."**

## 5. Suggested test procedure

1. Warm up the models (first calls are usually slower).
2. Drive `POST /v1/faces/privacy_extract` with our payload shape (§2) at a **fixed 17 rps for 10 min**, then **20 rps for 10 min**.
3. If both pass comfortably, **ramp until it breaks** (latency p95 > ~1 s or errors climb) to find the real ceiling.
4. Record the counters in §6 at: (a) 17 rps, (b) 20 rps, (c) the max sustainable point.

A simple load tool (k6/vegeta/locust) or your own harness is fine — the key is a **sustained** rate with **realistic ~500 KB base64 image bodies**, `recognition: true`.

## 6. Results to send back (fill this in)

| Counter | @ 17 rps | @ 20 rps | @ MAX sustainable |
|---|---|---|---|
| Achieved throughput (req/s) | | | |
| Latency p50 (ms) | | | |
| Latency p95 (ms) | | | |
| Latency p99 (ms) | | | |
| Error rate (%) | | | |
| Timeout / 429 rate (%) | | | |
| Max concurrent in-flight served | | | |
| GPU utilization (%) | | | |
| CPU utilization (%) | | | |
| Memory headroom | | | |

Plus, in prose:

- **Max sustainable `privacy_extract` rps** on the current instance (single most important number).
- **Does `privacy_extract` share a compute/rate budget** with the other endpoints (recognize / enroll / search / compare / blur)? If yes, what is the **combined ceiling**, and how is it split? (We were told the shared upstream is ~30–35 rps total — please confirm for THIS instance.)
- **Backpressure behavior** at overload: does it queue, return 429, or just slow down?
- **Internal breakdown** if available: detect vs recognize vs blur time per call.
- **Any knobs** to raise throughput on the current hardware (batch size, worker/replica count, GPU, model settings), and what each would buy.

## 7. What we do with your answer

- We set our governor `rps` to your **sustainable** number (currently provisioned for 20).
- We size our worker fleet to match.
- If the sustainable rate is **< 17 rps**, 1000/min isn't reachable on this instance and we'll plan around your reported ceiling (scale FS / add instances / lower the target).

Thanks — a single "sustainable rps + p95 latency + error rate at that rps" is the minimum we need; the full table is ideal.
