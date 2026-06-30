from typing import Any

import cv2
import numpy as np
import os

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        out = np.zeros_like(v, dtype=np.float32)
        if out.size > 0:
            out[0] = 1.0
        return out
    return (v / n).astype(np.float32)


def _pick_best_face(faces: list[Any], min_det_score: float) -> Any | None:
    best = None
    best_area = -1.0
    for f in faces:
        try:
            score = float(getattr(f, "det_score", 0.0) or 0.0)
            if score < float(min_det_score):
                continue
            bbox = np.asarray(getattr(f, "bbox", None), dtype=np.float32).reshape(-1)
            if bbox.size != 4:
                continue
            area = float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))
            if area > best_area:
                best = f
                best_area = area
        except Exception:
            continue
    return best


def _debug(msg: str) -> None:
    # No-op here; app controls logging verbosity.
    try:
        import logging
        logging.getLogger("uvicorn.error").info("%s", msg)
    except Exception:
        pass


def _quality_check_and_embed(
    bgr: np.ndarray,
    *,
    embedder: "BuffaloLEmbedder",
    evaluator: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run detection + optional quality eval + embedding.

    This is intentionally framework-agnostic (no FastAPI/HTTP exceptions, no metrics).
    Callers should translate exceptions into their API errors.
    """

    best_face = embedder.detect_best(bgr)

    quality_meta: dict[str, Any] | None = None
    if evaluator is not None:
        try:
            q = evaluator.evaluate(bgr, best_face)
        except Exception as e:
            raise RuntimeError("quality evaluator failure") from e
        quality_meta = q
        if q.get("status") == "rejected":
            raise ValueError(f"quality_reject:{q.get('reason')}")

    emb = getattr(best_face, "normed_embedding", None)
    if emb is None:
        emb = getattr(best_face, "embedding", None)
    if emb is None:
        raise ValueError("no embedding")
    out = _l2_normalize(np.asarray(emb, dtype=np.float32))
    meta: dict[str, Any] = {"quality": quality_meta, "decision": {"status": "embedded"}}
    return out, meta


class BuffaloLEmbedder:
    def __init__(
        self,
        model_root: str,
        model_name: str = "buffalo_l",
        det_size: int = 640,
        min_det_score: float = 0.5,
        providers: str = "CUDAExecutionProvider,CPUExecutionProvider",
    ):
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            raise RuntimeError(
                "insightface is required for Buffalo-L embedding generation"
            ) from e

        self.min_det_score = float(min_det_score)
        self.det_size = int(det_size)
        
        # Parse providers and set up optimized options
        raw_providers = [p.strip() for p in str(providers).split(",") if p.strip()]
        self.providers = []
        
        cuda_mem_limit = int(os.environ.get("ORT_CUDA_MEMORY_LIMIT_MB", "0")) * 1024 * 1024
        arena_strategy = os.environ.get("ORT_CUDA_ARENA_EXTEND_STRATEGY", "kNextPowerOfTwo")
        
        # cuDNN conv algo search: EXHAUSTIVE (ORT default) probes every algorithm
        # and reserves GBs of workspace per session. HEURISTIC is far leaner with
        # negligible latency cost for these small models.
        cudnn_algo = os.environ.get("ORT_CUDNN_CONV_ALGO_SEARCH", "HEURISTIC").strip()

        for p in raw_providers:
            if p == "CUDAExecutionProvider":
                opts = {"device_id": 0}
                if cuda_mem_limit > 0:
                    opts["gpu_mem_limit"] = cuda_mem_limit
                if arena_strategy:
                    opts["arena_extend_strategy"] = arena_strategy
                if cudnn_algo:
                    opts["cudnn_conv_algo_search"] = cudnn_algo
                self.providers.append((p, opts))
            else:
                self.providers.append(p)

        self.enable_fallback_variants = str(
            os.environ.get("BUFFALO_ENABLE_FALLBACK_VARIANTS", "1")
        ).strip() not in ("0", "false", "False")

        # Only load the models we actually use. buffalo_l ships 5 ONNX models
        # (detection, recognition, genderage, landmark_2d_106, landmark_3d_68);
        # genderage + landmark_2d_106 are unused and each adds a CUDA session +
        # cuDNN workspace. landmark_3d_68 is kept for pose (yaw/pitch) quality.
        modules_env = os.environ.get(
            "BUFFALO_MODULES", "detection,recognition,landmark_3d_68"
        ).strip()
        allowed_modules = [m.strip() for m in modules_env.split(",") if m.strip()] or None

        fa_kwargs: dict = {
            "name": str(model_name),
            "root": str(model_root),
            "providers": self.providers,
        }
        if allowed_modules:
            fa_kwargs["allowed_modules"] = allowed_modules
        try:
            self.app = FaceAnalysis(**fa_kwargs)
        except TypeError:
            # Older insightface without allowed_modules support.
            fa_kwargs.pop("allowed_modules", None)
            self.app = FaceAnalysis(**fa_kwargs)
        self.app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))

    def detect_best(self, bgr: np.ndarray):
        faces = self.app.get(bgr)
        if not faces:
            if not self.enable_fallback_variants:
                _debug("detected_faces=0")
                raise ValueError("no face detected")

            def _resize(img: np.ndarray, scale: float) -> np.ndarray:
                if 0.999 <= float(scale) <= 1.001:
                    return img
                h, w = img.shape[:2]
                nh = max(2, int(round(h * scale)))
                nw = max(2, int(round(w * scale)))
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                out = cv2.resize(img, (nw, nh), interpolation=interp)
                try:
                    oh, ow = out.shape[:2]
                    m = max(oh, ow)
                    if m > 1600:
                        s2 = 1600.0 / float(m)
                        out = cv2.resize(
                            out,
                            (max(2, int(round(ow * s2))), max(2, int(round(oh * s2)))),
                            interpolation=cv2.INTER_AREA,
                        )
                except Exception:
                    pass
                return out

            variants: list[tuple[np.ndarray, str]] = []
            for s in (1.0, 1.25, 1.5, 2.0, 0.75, 0.5, 0.33, 0.25):
                img_s = _resize(bgr, s)
                tag_s = f"s{int(s*100)}"
                variants.append((img_s, tag_s))
                variants.append((cv2.rotate(img_s, cv2.ROTATE_90_CLOCKWISE), tag_s + "_r90"))
                variants.append((cv2.rotate(img_s, cv2.ROTATE_180), tag_s + "_r180"))
                variants.append((cv2.rotate(img_s, cv2.ROTATE_90_COUNTERCLOCKWISE), tag_s + "_r270"))

            best_face = None
            best_tag = None
            best_area = -1.0
            best_any_face = None
            best_any_tag = None
            best_any_score = -1.0
            for img, tag in variants:
                try:
                    fs = self.app.get(img)
                except Exception:
                    fs = None
                if not fs:
                    continue
                cand = _pick_best_face(fs, self.min_det_score)
                if cand is None:
                    any_face = _pick_best_face(fs, 0.0)
                    if any_face is not None:
                        try:
                            s = float(getattr(any_face, "det_score", 0.0) or 0.0)
                        except Exception:
                            s = 0.0
                        if s > best_any_score:
                            best_any_score = s
                            best_any_face = any_face
                            best_any_tag = tag
                    continue
                try:
                    bbox = np.asarray(getattr(cand, "bbox", None), dtype=np.float32).reshape(-1)
                    if bbox.size == 4:
                        area = float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))
                    else:
                        area = 0.0
                except Exception:
                    area = 0.0
                if area > best_area:
                    best_face = cand
                    best_area = area
                    best_tag = tag
            if best_face is None:
                if best_any_face is not None:
                    _debug(f"detected_faces=1 rotation={best_any_tag or 'none'}")
                    raise ValueError(
                        f"no face above min_det_score={float(self.min_det_score):.2f} (best_det_score={float(best_any_score):.2f})"
                    )
                _debug("detected_faces=0")
                raise ValueError("no face detected")
            _debug(f"detected_faces=1 rotation={best_tag or 'none'}")
            best = best_face
        else:
            _debug(f"detected_faces={len(faces) if faces else 0}")
            best = _pick_best_face(faces, self.min_det_score)
        if best is None:
            raise ValueError("no face above min_det_score")

        try:
            bbox = np.asarray(getattr(best, "bbox", None), dtype=np.float32).reshape(-1)
            score = float(getattr(best, "det_score", 0.0) or 0.0)
            if bbox.size == 4:
                _debug(
                    f"best_face_det_score={score:.3f} bbox=[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}]"
                )
        except Exception:
            pass

        return best

    def detect_all(self, bgr: np.ndarray):
        faces = self.app.get(bgr)
        if not faces:
            raise ValueError("no face detected")
        return faces

    def embed_bgr(self, bgr: np.ndarray) -> np.ndarray:
        best = self.detect_best(bgr)

        emb = getattr(best, "normed_embedding", None)
        if emb is None:
            emb = getattr(best, "embedding", None)
        if emb is None:
            raise ValueError("no embedding")
        out = _l2_normalize(np.asarray(emb, dtype=np.float32))
        _debug(f"embedding_dim={int(out.reshape(-1).shape[0])}")
        return out
