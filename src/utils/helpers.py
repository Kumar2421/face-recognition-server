import base64
import hashlib
import uuid
import time
import os
import cv2
import numpy as np
from datetime import datetime, timezone
from fastapi import HTTPException

def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _sha1_bytes_hex(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _uuid5_from_name(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, name))

def _decode_b64_image(image_b64: str) -> np.ndarray:
    try:
        img = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid base64")
    arr = np.frombuffer(img, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="unable to decode image")
    return bgr

def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return bgr
    except Exception:
        return None

def _t() -> float:
    return time.time()

def _now_ts() -> float:
    return time.time()

def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _save_thumb(bgr: np.ndarray, thumbs_dir: str, image_id: str) -> str:
    try:
        h, w = bgr.shape[:2]
        scale = 256.0 / float(max(h, w)) if max(h, w) > 0 else 1.0
        if scale < 1.0:
            nh = max(2, int(round(h * scale)))
            nw = max(2, int(round(w * scale)))
            thumb = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            thumb = bgr
        rel = f"{image_id}.jpg"
        _ensure_dir(thumbs_dir)
        abs_path = os.path.join(thumbs_dir, rel)
        cv2.imwrite(abs_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return "/thumbs/" + rel
    except Exception:
        return ""
