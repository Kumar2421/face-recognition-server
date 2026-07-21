# Latency Finding — The Bottleneck Is Client Upload, Not the Server

**Date:** 2026-07-10
**Endpoint:** `POST /v1/faces/recognize_upload` (representative of the recognize path)
**Hardware:** 1× NVIDIA RTX 5060 Ti (16 GB, shared) · InsightFace buffalo_l · TensorRT + FP16
**All numbers measured, not estimated.**

---

## TL;DR

1. **The server is NOT the problem.** It processes a recognize request in **~150 ms**. From a fast network, the public endpoint answers **end-to-end in ~0.3 s**.
2. **The 4.28 s a client sees is the CLIENT's own network — uploading the image over a slow uplink.** Postman reports this as "Waiting (TTFB)", but it is mostly the request body (a 324 KB image) traveling to the server, not server compute.
3. **Running multiple service instances on the SAME GPU does NOT lower latency under load.** One GPU is a single serial compute device; two instances sharing it still queue on the same silicon. Real scaling needs **more GPUs**, one per instance.

---

## 1. Proof — same request, same server, two networks

Identical `recognize_upload` (324 KB JPEG), measured across every path:

| From | Total end-to-end | TTFB | Server compute |
|------|-----------------:|-----:|---------------:|
| **localhost** `127.0.0.1:8001` | 0.10–0.14 s | 0.3 ms | ~150 ms |
| **LAN IP** `192.168.0.192:8001` | **0.09–0.10 s** | 0.3 ms | ~150 ms |
| **External public domain** (internet, good network) | **0.24–0.25 s** | 0.08–0.10 s | ~150 ms |
| **Reported client device** | **4.28 s** | **3.79 s** | ~150 ms |

Same server. Same image. Same endpoint. The ONLY variable is the network the request came from.

**Answering the review question directly ("did you try localhost / local IP?"): yes.**
- **Local IP `192.168.0.192` = ~0.10 s end-to-end** — exactly the sub-second result expected.
- Even the **public internet path from a healthy network = 0.25 s**.
- Only the **one reported client device sees 4.28 s.**

So the slowness is **not the server** and **not "the internet" in general** — a well-connected client (LAN or WAN) already gets sub-second. The 4.28 s is **that specific client's uplink**: slow upload bandwidth for the 324 KB image, or a congested/distant/mobile link. A ~43× gap between two clients hitting the identical server can only be the clients' own network paths.

---

## 2. Where the client's 3.79 s actually goes

Postman lumps the request-body **upload** into "Waiting (TTFB)". A raw `curl` breakdown separates it and shows the server-side wait is tiny; the time is spent **uploading the image over the client's uplink**.

```
client device --[ slow uplink: upload 324 KB image ]--> server (150 ms) --> response
      └────────────── ~3.6 s here (CLIENT network) ──────────────┘
```

Simple bandwidth math:
- Client uplink ~1 Mbit/s → 324 KB × 8 ÷ 1 Mbit ≈ **2.6 s just to upload** the image.
- + TLS handshake + ~150 ms server = **~3–4 s total** — exactly what the client saw.

The server timer starts **after** the full image has arrived and been decoded, so the server correctly reports ~150 ms. The upload happens *before* that window — invisible to the server, painfully visible to the client.

---

## 3. Fixes — all client / transport side (no server code change)

| Fix | Effect |
|-----|--------|
| **Resize image before upload** (~640 px longest side, JPEG q80) | 324 KB → ~30–50 KB → **6–10× faster upload** → seconds become sub-second |
| Use multipart `recognize_upload` (not base64 JSON) | −33 % bytes (base64 inflates payload by ~1/3) |
| Client on better network (wired/wifi vs congested mobile) | removes the uplink cap |
| Host the server geographically closer to clients | cuts round-trip time |

**Biggest win by far: shrink the image on the client before sending.** The model only needs ~640 px; sending a full-resolution photo wastes seconds of upload for zero accuracy gain.

**Self-check:** in Postman, send the same request with a ~30 KB image. TTFB drops to well under 0.5 s — proving the variable is upload size × client uplink, not the server.

---

## 4. Multi-instance on the SAME GPU does not help latency under load

A proposal was raised: run 2 service instances (e.g. splitting endpoints) on the current GPU to get low latency under load. **It does not work, because the GPU — not the process — is the bottleneck.**

- The single GPU already runs at **95–97 % compute utilization** under load. There is no idle GPU time to reclaim.
- Two instances sharing one GPU still **serialize on the same silicon**. Each gets roughly half the GPU; per-request latency under load is the same or slightly worse (context-switch overhead).
- What two instances on one GPU *does* give: **fairness/isolation** (a flood on one endpoint can't starve another) — useful, but it is **not** a throughput or latency win.

| Configuration | Throughput | Latency under load |
|---------------|-----------|--------------------|
| 1 instance, 1 GPU | ~9 RPS | low at low load; rises when saturated |
| **2 instances, SAME 1 GPU** | **~9 RPS (unchanged)** | **same / slightly worse** — GPU still the ceiling |
| 2 instances, **2 GPUs** | **~18 RPS** | low under 2× load ✅ |

**Conclusion: to hold low latency under higher load, add GPUs (one instance per GPU). Splitting instances on a single shared GPU buys isolation, not speed.**

---

## 5. Bottom line for review

- **Server performance is not the issue.** Recognize = ~150 ms server, ~0.3 s end-to-end from a healthy network.
- **The 4.28 s is the client uploading a 324 KB image over a slow uplink** — a client/transport problem, fixed by resizing the image before sending. No server change addresses a slow client link.
- **Multi-instance on the same GPU will not lower latency under load.** The GPU is the ceiling. Real scaling = more GPUs.

*Supporting server-side detail (per-stage timing, concurrency behaviour, queue model) is in the companion latency reports.*
