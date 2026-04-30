# Face Recognition Service - Comprehensive Benchmark Report

**Generated:** April 30, 2026  
**Benchmark Date:** April 19, 2026  
**Report Version:** 1.0

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Total API Events** | 167,674 |
| **Events Processed** | 27,653 |
| **Events Skipped** | 66,847 (already in system) |
| **Success Rate** | 100% (0 failures) |
| **Pages Processed** | 621 |
| **Processing Duration** | ~3 hours 13 minutes |
| **Average Model Latency** | 66.86 ms |
| **Average Processing Latency** | 22.44 ms |
| **Last Processed ID** | 12516248 |

---

## 2. System Architecture

### 2.1 Service Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Face Recognition Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│  Input → Detection → Quality Check → Embedding → Search → Output │
└─────────────────────────────────────────────────────────────────┘
```

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Face Detection** | InsightFace (det_10g.onnx) | Detect faces in images |
| **Embedding Model** | Buffalo-L (w600k_r50.onnx) | Generate 512-dim face embeddings |
| **Vector Database** | Qdrant (v24 collection) | Similarity search & storage |
| **Inference** | ONNX Runtime + TensorRT | GPU-accelerated inference |
| **API Framework** | FastAPI | REST API endpoints |

---

## 3. Model Configuration

### 3.1 Buffalo-L Model Suite

| Model | File | Size | Purpose |
|-------|------|------|---------|
| **Detection** | `det_10g.onnx` | 16.9 MB | Face detection (RetinaFace) |
| **Embedding** | `w600k_r50.onnx` | 174.4 MB | 512-dim feature extraction |
| **Landmarks 2D** | `2d106det.onnx` | 5.0 MB | 106-point 2D landmarks |
| **Landmarks 3D** | `1k3d68.onnx` | 143.6 MB | 68-point 3D landmarks |
| **Attributes** | `genderage.onnx` | 1.3 MB | Gender/age estimation |

### 3.2 Detection Parameters

```yaml
BUFFALO_DET_SIZE: 640              # Detection input size (640x640)
BUFFALO_MIN_DET_SCORE: 0.3          # Minimum detection confidence
BUFFALO_PROVIDERS:                  # Execution providers (priority order)
  - TensorrtExecutionProvider       # TensorRT (fastest)
  - CUDAExecutionProvider           # CUDA fallback
  - CPUExecutionProvider            # CPU fallback
```

### 3.3 TensorRT Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| `ORT_TENSORRT_ENGINE_CACHE_ENABLE` | 1 | Enable engine caching |
| `ORT_TENSORRT_CACHE_PATH` | `/data/trt_cache` | Cache storage location |
| `ORT_TENSORRT_FP16_ENABLE` | 1 | FP16 precision for speed |

---

## 4. Vector Search Configuration (Qdrant)

### 4.1 Collection Settings

| Parameter | Value |
|-----------|-------|
| **Collection Name** | `frigate_faces_v24` |
| **Vector Size** | 512 dimensions |
| **Distance Metric** | Cosine Similarity |
| **HNSW EF Parameter** | 64 |
| **Exact Search** | Disabled (approximate) |

### 4.2 Search Performance Tuning

```yaml
QDRANT_HNSW_EF: 64        # Search-time quality/speed tradeoff
QDRANT_EXACT: 0           # Use HNSW index (faster)
```

**HNSW EF Recommendations:**
- Lower (32-64): Faster search, slightly lower recall
- Higher (128-256): Better recall, slower search

---

## 5. Quality Control Thresholds

### 5.1 Face Quality Filters

| Metric | Minimum | Maximum | Notes |
|--------|---------|---------|-------|
| **Blur Score** | 35 | - | Laplacian variance (higher = sharper) |
| **Face Ratio** | 1% | - | Min face size relative to image |
| **Brightness** | 28 | 220 | Avoid under/over-exposed images |
| **Resolution** | 40px | - | Minimum face dimension |
| **Landmark Confidence** | 0.5 | - | Keypoint detection quality |
| **Yaw Angle** | - | 70° | Head turn left/right |
| **Pitch Angle** | - | 60° | Head tilt up/down |

### 5.2 Detection Confidence

- **Minimum Detection Score:** 0.30
- **Fallback Variants:** Disabled (strict detection)
- **Multi-scale Detection:** Enabled (scales: 1.0, 1.25, 1.5, 2.0, 0.75, 0.5, 0.33, 0.25)

---

## 6. Recognition Parameters

### 6.1 Matching Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| `min_similarity` | 0.45 | Minimum cosine similarity for match |
| `top_k` | 6 | Number of candidates to retrieve |
| `min_top2_margin` | 0.05 | Required gap between #1 and #2 matches |
| `top2_high_conf_sim` | 0.35 | High confidence secondary threshold |

### 6.2 Auto-Enrollment Features

| Feature | Setting | Purpose |
|---------|---------|---------|
| `AUTO_ADD_EMBEDDING_ENABLE` | 1 | Auto-add high-confidence matches |
| `AUTO_ADD_EMBEDDING_MIN_SIM` | 0.75 | Threshold for auto-add |
| `NO_MATCH_AUTO_ENROLL_ENABLE` | 1 | Create visitor entries for unknowns |
| `NO_MATCH_AUTO_ENROLL_PREFIX` | `visiter` | Visitor ID prefix |
| `SUBJECT_MAX_EMBEDDINGS` | 10 | Max embeddings per person |

---

## 7. Performance Metrics

### 7.1 Benchmark Results (April 19, 2026)

```
Start Time:    2026-04-30 06:25:50
End Time:      2026-04-30 09:38:59
Duration:      3h 13m 9s

Events:
├── Total Available:     167,674
├── Processed:            27,653
├── Skipped (existing):  66,847
└── Failed:                    0

Performance:
├── Avg Model Time:       66.86 ms
├── Avg Processing Time:  22.44 ms
└── Total Model Time:   1,848,881 ms (~30.8 min of compute)
```

### 7.2 Processing Rate

| Metric | Value |
|--------|-------|
| **Events per Second** | ~2.4 events/sec |
| **Pages per Hour** | ~193 pages/hour |
| **Events per Page (avg)** | ~44.5 events/page |

### 7.3 Latency Breakdown

| Stage | Average Time |
|-------|--------------|
| Model Inference | 66.86 ms |
| Processing Overhead | 22.44 ms |
| **Total per Event** | ~89.3 ms |

---

## 8. Data Pipeline

### 8.1 External API Poller Flow

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  External API   │────▶│  Face Service│────▶│  Qdrant Vector  │
│  (Fusion Apps)  │     │  (Inference) │     │     DB          │
└─────────────────┘     └──────────────┘     └─────────────────┘
        │                                               ▲
        │                                               │
        └────────────────── Poll Loop ──────────────────┘
                    (5 workers, 300s interval)
```

### 8.2 State Management

```json
{
  "last_id_2026-04-19": "12516248"
}
```

- **State File:** `poller_state.json`
- **Benchmark File:** `poller_benchmark.json`
- **Resume Capability:** Yes (processes from last_id)

---

## 9. Resource Utilization

### 9.1 GPU Configuration

| Setting | Value |
|---------|-------|
| `GPU_INFERENCE_MANAGER` | 1 (Enabled) |
| `GPU_QUEUE_MAX` | 256 |
| `GPU_BATCH_WINDOW_MS` | 0 (Immediate processing) |
| `NVIDIA_VISIBLE_DEVICES` | all |

### 9.2 Volume Mounts

| Path | Purpose |
|------|---------|
| `/data/facefolder` | Reference face images (read-only) |
| `/data/thumbs` | Thumbnail storage |
| `/data/images` | Processed images |
| `/data/events` | Event images |
| `/data/trt_cache` | TensorRT engine cache |
| `/models` | ONNX model files (read-only) |

---

## 10. Recommendations

### 10.1 Performance Optimizations

1. **Batch Processing**: Consider increasing `GPU_BATCH_WINDOW_MS` for higher throughput
2. **HNSW Tuning**: If recall issues occur, increase `QDRANT_HNSW_EF` to 128
3. **Worker Scaling**: Current 5 workers; can increase if GPU utilization is low

### 10.2 Quality Improvements

1. **Blur Threshold**: Current 35 is aggressive; consider 40-50 for stricter quality
2. **Detection Score**: 0.3 may allow low-quality detections; consider 0.5
3. **Face Ratio**: 1% minimum may be too small; consider 2-3%

### 10.3 Monitoring Metrics

Track these key metrics for ongoing optimization:

| Metric | Target | Current |
|--------|--------|---------|
| Events/sec | > 5 | ~2.4 |
| Model Latency P99 | < 100ms | ~67ms |
| Match Rate | > 80% | Monitor |
| Quality Reject Rate | < 20% | Monitor |

---

## 11. Appendix

### 11.1 Configuration Files

| File | Purpose |
|------|---------|
| `config.yaml` | Recognition thresholds & API settings |
| `docker-compose.yml` | Service orchestration |
| `poller_state.json` | Sync checkpoint state |
| `poller_benchmark.json` | Performance metrics |

### 11.2 Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/events/recognition` | POST | Submit image for recognition |
| `/v1/events/recognition` | GET | Query existing events |
| `/v1/search` | POST | Search faces by image |
| `/health` | GET | Service health check |

---

*Report generated from benchmark data on April 30, 2026*
