# Root-Cause Analysis: Duplicate `visiter-*` IDs Created for Same Person

## Problem Statement
The system is configured to auto-enroll a new Qdrant embedding (and generate a new `visiter-*` subject ID) when a recognition event is classified as `no_match`. In production, the same physical person can later be assigned **a new `visiter-*` ID again**, instead of being recognized as the previously-enrolled visitor.

Impact:
- Visitor identity fragmentation (same person becomes multiple visitor IDs)
- Recognition quality appears worse than it is
- Downstream analytics and UI become noisy

---

## Where the Behavior Comes From (Code Path)
### Recognition endpoint
The behavior is implemented in `POST /v1/events/recognition` in `app.py`:

- The service detects faces and produces an embedding (`emb`).
- It searches Qdrant (`_qdrant_search`) and evaluates match/no-match:
  - A match requires `best_similarity >= min_similarity`.
  - It may also require passing top-2 margin gating (`_passes_top2_margin`).
- If decision is `no_match` and `NO_MATCH_AUTO_ENROLL_ENABLE=1`, it creates:
  - `new_subject_id = f"{prefix}-{seq}"` (prefix from `NO_MATCH_AUTO_ENROLL_PREFIX`)
  - and upserts a Qdrant point with payload `subject_id=new_subject_id` and `source="no_match_auto_enroll"`.

### Key point
A **new visitor ID is created any time a repeat sighting produces `no_match`**, even if an earlier `visiter-*` embedding exists in Qdrant.

---

## Current Configuration Evidence
From repo config and runtime env (as provided):

### `config.yaml` (recognition)
```yaml
recognition:
  min_similarity: 0.50
  top_k: 5
  min_top2_margin: 0.06
  top2_high_conf_sim: 0.50
```

### `docker-compose.yml` (face_service env)
```yaml
NO_MATCH_AUTO_ENROLL_ENABLE=1
NO_MATCH_AUTO_ENROLL_PREFIX=visiter
NO_MATCH_AUTO_ENROLL_BLOCK_MIN_SIM=0.55
AUTO_ADD_EMBEDDING_ENABLE=0
AUTO_ADD_EMBEDDING_MIN_SIM=0.80
SUBJECT_MAX_EMBEDDINGS=10
BUFFALO_ENABLE_FALLBACK_VARIANTS=0
```

---

## Root Causes (Most Likely)
### RC1) Visitors get only 1 embedding, so later recall is weak
With `NO_MATCH_AUTO_ENROLL_ENABLE=1`, each new visitor starts with **a single embedding**.

With `AUTO_ADD_EMBEDDING_ENABLE=0`, the system does **not** add additional embeddings for that visitor on later sightings.

Consequences:
- Embeddings vary across:
  - pose / yaw/pitch
  - lighting
  - distance / resolution
  - blur and occlusion
- The later embedding may not match strongly enough to the single stored vector.
- The result becomes `no_match` again.
- A **new** visitor ID is created.

### RC2) Match gating can convert a “likely match” into `no_match`
A match requires:
- `best_similarity >= min_similarity` (0.50)
- and may require passing top-2 margin gating:
  - margin requirement is `min_top2_margin` (0.06)
  - but `top2_high_conf_sim` is 0.50, so high-confidence bypass can trigger early depending on env mapping

When the gallery grows (many visitors), the second-best candidate often gets close to the best candidate, reducing the margin. This can force `no_match` even when similarity is above threshold.

### RC3) Auto-enroll is not blocked aggressively enough for “possible matches”
The code attempts to prevent new visitor creation if there is a possible match:

- If best hit has `best_sim >= NO_MATCH_AUTO_ENROLL_BLOCK_MIN_SIM`, auto-enroll is skipped.
- Current value: `0.55`.

If real true-positive similarities often land in ~`0.50 - 0.60`, this threshold is borderline:
- Too high => duplicates
- Too low => risk of mistakenly merging different people

### RC4) Reduced robustness due to fallback variants disabled
`BUFFALO_ENABLE_FALLBACK_VARIANTS=0` can reduce robustness for rotated/awkward faces.

Result:
- lower similarity for repeat sightings
- more `no_match`
- more auto-enrolled visitor IDs

---

## Recommended Mitigations
### Mitigation A (Primary): Enable auto-add embeddings for matched events
Goal: once a visitor starts matching, continually strengthen that visitor with more embeddings.

Recommended:
- `AUTO_ADD_EMBEDDING_ENABLE=1`
- Choose `AUTO_ADD_EMBEDDING_MIN_SIM` based on data; a practical starting point is `0.80`.
- Keep `SUBJECT_MAX_EMBEDDINGS=10` (already set)

Notes:
- This does not change match decisions.
- It improves recall over time and reduces duplicates.

### Mitigation B: Tune/validate top-2 margin gating
If duplicates are driven by margin gating, consider:
- temporarily disabling margin gating (set margin requirement to `0.0`) to validate
- or lowering the margin requirement

### Mitigation C: Tune possible-match block for no-match auto-enroll
If you want “never create new visitor if there is any decent near-hit”, increase protection by tuning:
- `NO_MATCH_AUTO_ENROLL_BLOCK_MIN_SIM`

Rule of thumb:
- Higher value => more new visitors, fewer mistaken merges
- Lower value => fewer new visitors, but higher chance of merging different people

### Mitigation D: Consider enabling fallback variants
- `BUFFALO_ENABLE_FALLBACK_VARIANTS=1`

---

## Verification / Debug Checklist
To validate which cause is active, inspect an event that produced a duplicate visitor:

1. Pick a `no_match` event that auto-enrolled a visitor.
2. Check event `meta` fields:
   - `meta.decision.min_similarity`
   - `meta.top2_required`, `meta.top2_margin`, `meta.top2_second`
   - `meta.decision.no_match_auto_enroll` (enrolled? possible_match?)
3. Confirm whether the best Qdrant similarity was:
   - below `min_similarity` (threshold issue)
   - above `min_similarity` but failed top-2 margin (margin gating issue)

Expected outcomes:
- If similarity is just below threshold: tune `min_similarity` and improve embeddings.
- If margin fails often: tune/disable margin or reduce gallery ambiguity.
- If similarity varies strongly: enable auto-add embeddings and/or improve quality gating.

---

## Suggested Next Change Set (Low Risk)
1. Enable auto-add:
   - `AUTO_ADD_EMBEDDING_ENABLE=1`
   - keep `AUTO_ADD_EMBEDDING_MIN_SIM=0.80` (or adjust after observing similarities)
2. Keep `NO_MATCH_AUTO_ENROLL_BLOCK_MIN_SIM=0.55` as current starting point.
3. Re-test duplicates over a few hours/days.

If duplicates persist:
- Evaluate whether top-2 margin gating is rejecting matches and tune accordingly.

---

## Open Questions
- What are typical similarity values for the same person across cameras/conditions?
- Is `FACE_SERVICE_TOP2_MARGIN` mapped from `config.yaml` in this deployment, and what is the effective runtime value?
- Do you want “strict identity” (more new IDs) or “aggressive merging” (fewer new IDs but risk of merging different people)?
