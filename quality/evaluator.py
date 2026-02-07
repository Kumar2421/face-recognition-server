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

    def _face_ratio(self, img_shape: tuple[int, int], bbox: Optional[np.ndarray]) -> float:
        if bbox is None:
            return 0.0
        try:
            h, w = img_shape[:2]
            area_img = float(max(1, h) * max(1, w))
            x1, y1, x2, y2 = [float(v) for v in bbox.reshape(-1)[:4]]
            area_face = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
            r = area_face / area_img if area_img > 0 else 0.0
            if not np.isfinite(r):
                r = 0.0
            return r
        except Exception:
            return 0.0

    def evaluate(self, bgr: np.ndarray, face: Optional[Any] = None) -> Dict[str, Any]:
        h, w = bgr.shape[:2]
        blur = self._blur_score(bgr)
        brightness = self._brightness(bgr)

        bbox = None
        landmark_score = None
        yaw = None
        pitch = None
        if face is not None:
            try:
                bbox = np.asarray(getattr(face, "bbox", None), dtype=np.float32)
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

        face_ratio = self._face_ratio((h, w), bbox)

        status = "ok"
        reason = ""

        if min(h, w) < self.min_resolution:
            status = "rejected"
            reason = "too_small"
        elif blur < self.min_blur:
            status = "rejected"
            reason = "low_blur"
        elif face_ratio < self.min_face_ratio:
            status = "rejected"
            reason = "face_too_small"
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
            "face_ratio": float(face_ratio),
            "landmark_score": (float(landmark_score) if landmark_score is not None else None),
            "yaw": (float(yaw) if yaw is not None else None),
            "pitch": (float(pitch) if pitch is not None else None),
            "status": status,
            "reason": reason,
        }
