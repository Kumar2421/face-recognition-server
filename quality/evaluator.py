import os
from typing import Any, Optional, Dict

import cv2
import numpy as np


class FaceQualityEvaluator:
    def __init__(self) -> None:
        self.min_blur = float(os.environ.get("FACE_QUALITY_BLUR_MIN", "60.0"))
        self.min_face_ratio = float(os.environ.get("FACE_QUALITY_FACE_RATIO_MIN", "0.04"))
        self.min_brightness = float(os.environ.get("FACE_QUALITY_BRIGHTNESS_MIN", "40.0"))
        self.max_brightness = float(os.environ.get("FACE_QUALITY_BRIGHTNESS_MAX", "220.0"))
        self.min_landmark_conf = float(os.environ.get("FACE_QUALITY_LANDMARK_MIN", "0.3"))
        self.max_abs_yaw = float(os.environ.get("FACE_QUALITY_MAX_ABS_YAW", "45"))
        self.max_abs_pitch = float(os.environ.get("FACE_QUALITY_MAX_ABS_PITCH", "35"))
        self.min_resolution = int(os.environ.get("FACE_MIN_RESOLUTION", "64"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _crop_face(
        self, bgr: np.ndarray, bbox: Optional[np.ndarray], pad: float = 0.3
    ) -> Optional[np.ndarray]:
        """Return a padded crop of the face region.

        The crop expands the tight bbox by `pad` (fraction of bbox side)
        in each direction so that we include forehead / chin context
        while still being much tighter than the full frame.

        Returns None if the bbox is invalid or the crop is degenerate.
        """
        if bbox is None:
            return None
        try:
            h, w = bgr.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]]
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                return None
            # Expand by pad fraction
            px = bw * pad
            py = bh * pad
            cx1 = max(0, int(x1 - px))
            cy1 = max(0, int(y1 - py))
            cx2 = min(w, int(x2 + px))
            cy2 = min(h, int(y2 + py))
            if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                return None
            return bgr[cy1:cy2, cx1:cx2]
        except Exception:
            return None

    def _blur_score(self, bgr: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = bgr if bgr.ndim == 2 else np.zeros((1, 1), dtype=np.uint8)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if not np.isfinite(score):
            score = 0.0
        return score

    def _brightness(self, bgr: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = bgr if bgr.ndim == 2 else np.zeros((1, 1), dtype=np.uint8)
        m = float(np.mean(gray))
        if not np.isfinite(m):
            m = 0.0
        return m

    def _face_ratio(self, img_shape: tuple, bbox: Optional[np.ndarray]) -> float:
        """Face area as a fraction of the full image area (observability only)."""
        if bbox is None:
            return 0.0
        try:
            h, w = img_shape[:2]
            area_img = float(max(1, h) * max(1, w))
            x1, y1, x2, y2 = [float(v) for v in bbox.reshape(-1)[:4]]
            area_face = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
            r = area_face / area_img if area_img > 0 else 0.0
            return r if np.isfinite(r) else 0.0
        except Exception:
            return 0.0

    def _face_abs_size(self, bbox: Optional[np.ndarray]) -> float:
        """Return the short-side pixel length of the face bbox.

        This is the primary size gate: a face 44x55 px on a 1300x700 frame
        is valid regardless of its ratio to the frame.  Ratio is kept as a
        metric but must NOT be used as the sole rejection criterion.
        """
        if bbox is None:
            return 0.0
        try:
            x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]]
            fw = max(0.0, x2 - x1)
            fh = max(0.0, y2 - y1)
            return float(min(fw, fh))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------

    def evaluate(self, bgr: np.ndarray, face: Optional[Any] = None) -> Dict[str, Any]:
        h, w = bgr.shape[:2]

        # ---- Extract face attributes from InsightFace object ----
        bbox: Optional[np.ndarray] = None
        landmark_score: Optional[float] = None
        yaw: Optional[float] = None
        pitch: Optional[float] = None

        if face is not None:
            try:
                bbox = np.asarray(getattr(face, "bbox", None), dtype=np.float32)
                if bbox.size != 4:
                    bbox = None
            except Exception:
                bbox = None
            try:
                landmark_score = float(getattr(face, "det_score", 0.0) or 0.0)
            except Exception:
                landmark_score = None

            # Optional pose signal (InsightFace sometimes exposes this)
            try:
                pose = getattr(face, "pose", None)
                if pose is not None:
                    p = np.asarray(pose, dtype=np.float32).reshape(-1)
                    if p.size >= 2:
                        pitch = float(p[0])
                        yaw = float(p[1])
                    elif p.size == 1:
                        yaw = float(p[0])
            except Exception:
                yaw = None
                pitch = None

        # ---- Crop to face region for blur / brightness ----
        # Key fix: for wide/tall full-frame images (e.g. 1300x700 CCTV frames)
        # the background dominates the Laplacian and mean-brightness metrics.
        # We crop to the padded face bbox so the signal reflects the actual face.
        face_crop = self._crop_face(bgr, bbox, pad=0.3)
        region = face_crop if face_crop is not None else bgr
        face_crop_shape = (
            (face_crop.shape[0], face_crop.shape[1]) if face_crop is not None else None
        )

        blur = self._blur_score(region)
        brightness = self._brightness(region)

        # ---- Face-ratio is still relative to the full frame ----
        face_ratio = self._face_ratio((h, w), bbox)

        # ---- Face absolute pixel size (primary size gate) ----
        face_abs = self._face_abs_size(bbox)

        # ---- Adaptive thresholds ----
        is_hires = h > 1000 or w > 1000

        # blur: when we have a valid face crop the threshold applies normally.
        # When we fall back to the full image (no bbox) be more lenient for hi-res.
        effective_min_blur = self.min_blur
        if face_crop is None and is_hires:
            effective_min_blur = self.min_blur * 0.7

        # ---- Evaluate ----
        # NOTE: face_ratio is NOT used for rejection.
        # For CCTV full-frame images a person 3m away will have a tiny ratio but
        # a perfectly sharp, high-confidence face.  We use absolute pixel size
        # (short side of bbox >= min_resolution) as the sole size gate.
        status = "ok"
        reason = ""

        if face_abs > 0 and face_abs < self.min_resolution:
            # Face was detected but is below the minimum usable pixel size
            status = "rejected"
            reason = "face_too_small_px"
        elif face_abs == 0 and min(h, w) < self.min_resolution:
            # No bbox available, fall back to whole-image size check
            status = "rejected"
            reason = "too_small"
        elif blur < effective_min_blur:
            status = "rejected"
            reason = "low_blur"
        elif (brightness < self.min_brightness) or (brightness > self.max_brightness):
            status = "rejected"
            reason = "too_dark" if brightness < self.min_brightness else "too_bright"
        elif (landmark_score is not None) and (landmark_score < self.min_landmark_conf):
            status = "rejected"
            reason = "low_landmark_conf"
        elif (yaw is not None) and (abs(float(yaw)) > self.max_abs_yaw):
            status = "rejected"
            reason = "pose_yaw"
        elif (pitch is not None) and (abs(float(pitch)) > self.max_abs_pitch):
            status = "rejected"
            reason = "pose_pitch"

        return {
            "blur": float(blur),
            "brightness": float(brightness),
            "face_ratio": float(face_ratio),          # informational — not used for rejection
            "face_abs_px": float(face_abs),            # short-side px of bbox — this is the size gate
            "landmark_score": (float(landmark_score) if landmark_score is not None else None),
            "yaw": (float(yaw) if yaw is not None else None),
            "pitch": (float(pitch) if pitch is not None else None),
            "face_crop_shape": face_crop_shape,        # (h, w) of the region used for blur/brightness
            "status": status,
            "reason": reason,
        }
