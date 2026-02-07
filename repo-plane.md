Strategic Recommendation (Expanded & Grounded)
Why single-model first is the right move

Your service:

consumes cropped faces

from many cameras

with unknown quality

in real-time

In this setup:

80% of failures come from bad crops, not bad embeddings

adding ArcFace does not fix:

blur

partial occlusion

extreme lighting

misaligned crops

hybrid fusion hides root causes and complicates ops

Correct strategy

Make Buffalo-L extremely reliable

Reject bad faces early

Log why faces fail

Stabilize thresholds

Only then consider second model

This is how commercial systems are built.

Master TODO List (Non-Over-Engineered)

I’ll break this into P0 → P3, where each phase is deployable.

🔴 P0 — Stabilize the Core (MANDATORY)
Goal

Make recognition predictable across cameras.

Tasks
1. Introduce Face Quality Gate (CRITICAL)

This is your biggest missing component.

Add a lightweight quality check before embedding.

Signals (cheap, CPU-friendly):

blur score (Laplacian variance)

brightness mean

face size ratio (face area / image area)

landmark confidence (from InsightFace)

Rules (example):

reject if blur < threshold

reject if face < X% of image

reject if landmarks < confidence

Output:

quality: {
  blur: 112.3,
  brightness: 0.42,
  face_ratio: 0.18,
  landmarks_score: 0.91,
  status: "ok | rejected"
}


Why now

prevents garbage embeddings

stabilizes similarity scores

reduces false positives dramatically

2. Make Enrollment Deterministic

Prevent duplicate or drifting identities.

Do this:

compute image_hash = sha256(original_bytes)

define:

point_id = sha256(subject_id + image_hash)


Result

same image can never be enrolled twice

re-enrollment is idempotent

backfill becomes safe later

3. Normalize Crop Assumptions

Your service consumes already-cropped faces, but cameras differ.

Standardize internally:

enforce minimum resolution

re-align using 5-point landmarks

convert all crops → canonical size before embedding

Outcome

cross-camera stability improves immediately

Acceptance Criteria (P0)

low-quality faces are rejected with reason

same image cannot be enrolled twice

similarity scores become tighter and more separable

🟠 P1 — Clean Abstractions (No Feature Expansion)
Goal

Prepare for future growth without changing behavior.

4. Extract Embedder into Module (Still Single Model)

You already know this, but keep it minimal.

Do NOT add registry yet.

Create:

embedders/
  buffalo_l.py
  base.py


Interface:

class FaceEmbedder:
    name
    dim
    preprocess()
    embed()


Why

isolates InsightFace logic

enables future swap with zero API change

reduces app.py complexity

5. Centralize Similarity & Threshold Logic

Right now thresholding is implicit.

Add explicit logic:

similarity calculation

decision (match / no-match)

configurable per request (optional)

Expose in response:

decision: {
  matched: true,
  threshold: 0.36,
  similarity: 0.41
}

Acceptance Criteria (P1)

no API behavior change

codebase becomes easier to reason about

one clear place to tune thresholds

🟡 P2 — Operational Confidence
Goal

Know when and why the system fails.

6. Add Minimal Metrics (No Tracing Yet)

Expose /metrics:

requests/sec

average embedding latency

quality rejection rate

recognition success rate

Qdrant latency

Why

tells you if cameras are bad

tells you if thresholds are wrong

tells you if GPU is overloaded

7. Store Failure Reasons in Payload

When recognition fails, log why.

Example:

failure_reason: "low_quality | below_threshold | no_face"


This becomes gold for tuning.

Acceptance Criteria (P2)

you can answer: “why did this face fail?”

you can compare cameras objectively
