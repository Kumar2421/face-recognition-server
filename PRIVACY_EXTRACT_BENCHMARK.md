# Benchmark Report — `POST /v1/faces/privacy_extract` (per-event model latency)

**Date:** 2026-07-11
**Hardware:** NVIDIA RTX 5060 Ti (16 GB) · CPU: QEMU Virtual CPU (18 vCPU, no AVX)
**Model:** InsightFace buffalo_l (`det_10g` detect + `w600k_r50` recognize + `1k3d68` landmark) · ONNXRuntime + TensorRT FP16
**Test image:** `0b50f28699aefbba.jpg` — **165×256, 12 KB, 1 face** (a real single-face event)
**Method:** live `privacy_extract`, warm, per-stage split from `X-Model-Ms` / server timing headers.

---

## 1. Executive summary

- **A single 1-face event is NOT 150–800 ms.** Warm, it is **~60 ms model / ~72 ms end-to-end** at `det_size=640`, and **~42 ms model / ~46 ms end-to-end at `det_size=320`.**
- **Sub-50 ms is easily achievable — and now measured** (§3): dropping detection input 640 → 320 takes the event to **~46 ms wall**.
- **The GPU model forwards are tiny (~12 ms total).** The rest of the per-event time is **CPU pre/post** (image resize, NMS anchor decode, face align) — that is what `det_size` reduces.
- The large numbers seen earlier (207–785 ms) were a **different, heavy case**: a 1920×1136 frame with **5 faces** under concurrency-12 contention. That is not a single 1-face event — see §5.

---

## 2. Per-stage breakdown — 1-face event (measured, warm)

From the server's own timing (`decode / detect / quality / crop+blur / encode`), same image, warm:

| Stage | det 640 | det 320 | what it is |
|-------|--------:|--------:|------------|
| decode | 1 ms | 1 ms | JPEG decode (image is only 12 KB) |
| **detect (full `app.get`)** | **55 ms** | **36 ms** | detection + landmark + recognition + all CPU pre/post |
| quality | 0 ms | 0 ms | eval |
| crop + blur | 0 ms | 0 ms | 1 face → trivial |
| encode + b64 | 1 ms | 1 ms | crop → JPEG |
| **model total** | **~60 ms** | **~42 ms** | server processing |
| queue | ~1 ms | ~1 ms | no contention at low load |
| **wall end-to-end** | **~72 ms** | **~46 ms ✅** | client-measured |

**The whole event is ~46–72 ms — well under the 150 ms you were quoted.**

---

## 3. Where the model time actually goes (this is the key correction)

The `detect` stage (55 ms at det 640) is **not** GPU-heavy. It is `app.get(frame)` = the full InsightFace pipeline, and the GPU part of it is tiny:

| Piece | time | on |
|-------|-----:|----|
| detection forward (`det_10g`) | ~5 ms | GPU |
| landmark forward (`1k3d68`) | ~3 ms | GPU |
| recognition forward (`w600k_r50`) | ~3.6 ms | GPU |
| **— GPU forwards total** | **~12 ms** | GPU |
| resize/pad to det size, normalize | | CPU |
| NMS + anchor decode | | CPU |
| face align / warp to 112 px | | CPU |
| numpy conversions | | CPU |
| **— CPU pre/post** | **~40 ms (640) → ~24 ms (320)** | CPU (GIL) |

**GPU inference is only ~12 ms; the ~40 ms is CPU pre/post.** `det_size=320` shrinks the CPU pre/post (fewer pixels to resize + fewer anchors to decode) → detect 55 → 36 ms → **event under 50 ms.**

*(GPU forward numbers measured directly: raw ONNXRuntime CUDA — detection 5.2 ms, recognition 3.6 ms.)*

---

## 4. det 640 vs det 320 — proof of sub-50 ms

Same image, warm, live service:

| | model total | wall e2e | detect stage | faces found |
|--|------------:|---------:|-------------:|------------:|
| `det_size=640` | ~60 ms | ~72 ms | 55 ms | 1 |
| **`det_size=320`** | **~42 ms** | **~46 ms** | 36 ms | 1 |

**~36 % faster, under 50 ms, face still detected.**

⚠️ **Accuracy caveat:** 320 detects large/near faces fine (this event unaffected), but **small/distant faces in wide 1080p frames may be missed**. Validate on real camera frames before committing `det_size=320` in production.

---

## 5. Why earlier numbers looked huge (not a contradiction)

| Scenario | model time | why |
|----------|-----------:|-----|
| **1-face event (this report)** | **~42–60 ms** | 1 face, small image, warm |
| 1080p frame, 5 faces, single req | ~207 ms | big frame detect + **5× per-face** crop/blur/encode |
| 1080p, 5 faces, **concurrency 12** | ~785 ms | above + **GPU-worker contention** (12 reqs ÷ 4 workers) |

The 785 ms was **5 faces on a 1080p frame under load** — not a single event. Per-event, per-face, warm, it's ~42–60 ms. Latency scales with **frame size × face count × concurrency**, not with a fixed 150 ms.

---

## 6. To reduce per-event latency further

| Lever | effect |
|-------|--------|
| **`det_size=320`** (done) | detect 55 → 36 ms → **event ~46 ms** ✅ (test small-face recall) |
| Client sends smaller frame (~640 px) | less CPU resize/decode → less pre/post |
| `return_image=false` (skip crop in response) | cuts response payload (transport, not model) |
| Fewer faces per frame | model scales ~linearly with face count |

**Not helpful for latency:** more GPU workers (GPU is only ~12 ms — the bottleneck is CPU pre/post), Triton (only accelerates the already-tiny ONNX forwards).

---

## 7. GPU vs CPU (context)

Full pipeline per image, single stream:

| Engine | median / img | RPS | note |
|--------|-------------:|----:|------|
| **GPU** RTX 5060 Ti (det 320) | ~42 ms | ~24 | fast path |
| **GPU** (det 640) | ~60 ms | ~14 | |
| CPU (QEMU, no AVX) | 409–601 ms | 1.5–2.3 | 7–10× slower; one inference eats ~16.7 of 18 cores |

CPU is not viable on this box (emulated, no AVX). GPU forwards are ~12 ms; the CPU pre/post is the real per-event cost.

---

## 8. Bottom line

| Question | Answer |
|----------|--------|
| Single 1-face event, warm? | **~42 ms model / ~46 ms wall at det 320** (~60 / ~72 ms at det 640) |
| Can we get under 50 ms? | **Yes — measured ~46 ms with `det_size=320`.** |
| Where does the model time go? | **~12 ms GPU forwards + ~30–40 ms CPU pre/post** (resize / NMS / align) |
| Why were earlier numbers 200–800 ms? | those were **5-face 1080p frames under concurrency**, not a single event |
| Biggest latency lever? | **`det_size`** + smaller input frame (both cut CPU pre/post) — NOT more GPU |

**Per-event latency is ~46 ms (det 320) / ~72 ms (det 640), not 150 ms. GPU inference is only ~12 ms; the rest is CPU pre/post, which `det_size=320` and smaller input frames reduce. Sub-50 ms is achieved and measured.**

*Reproduce: `IMG=0b50f28699aefbba.jpg` → POST `/v1/faces/privacy_extract`; read `X-Model-Ms` header + `%{time_total}`.*
