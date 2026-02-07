import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _stable_id(rel_path: str) -> str:
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=r"D:\projects\frigate\facefolder\security",
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\projects\frigate\facefolder\security_embeddings",
    )
    parser.add_argument(
        "--model-root",
        default=r"D:\projects\frigate\buffalo_l\models",
        help="Directory that contains a 'buffalo_l' folder with onnx models.",
    )
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--min-det-score", type=float, default=0.5)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_root = Path(args.model_root)

    if not input_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except Exception as e:
        raise RuntimeError(
            "opencv-python is required for this script. Install it in your environment."
        ) from e

    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        raise RuntimeError(
            "insightface is required for Buffalo-L embedding generation. Install it in your environment."
        ) from e

    providers = [p.strip() for p in str(args.providers).split(",") if p.strip()]

    app = FaceAnalysis(
        name=args.model_name,
        root=str(model_root),
        providers=providers,
    )
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    index: list[dict] = []
    num_images = 0
    num_embedded = 0

    for img_path in _iter_images(input_dir):
        num_images += 1

        rel_path = str(img_path.relative_to(input_dir)).replace("\\", "/")
        label = img_path.parent.name
        face_id = _stable_id(rel_path)

        img = cv2.imread(str(img_path))
        if img is None:
            index.append(
                {
                    "image": rel_path,
                    "label": label,
                    "success": False,
                    "error": "cv2.imread returned None",
                }
            )
            continue

        faces = app.get(img)
        if not faces:
            index.append(
                {
                    "image": rel_path,
                    "label": label,
                    "success": False,
                    "error": "no face detected",
                }
            )
            continue

        best = None
        best_area = -1.0
        for f in faces:
            score = float(getattr(f, "det_score", 0.0))
            if score < float(args.min_det_score):
                continue
            bbox = np.asarray(f.bbox, dtype=np.float32)
            area = float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))
            if area > best_area:
                best = f
                best_area = area

        if best is None:
            index.append(
                {
                    "image": rel_path,
                    "label": label,
                    "success": False,
                    "error": f"no face above min_det_score={args.min_det_score}",
                }
            )
            continue

        emb = getattr(best, "normed_embedding", None)
        if emb is None:
            emb = getattr(best, "embedding", None)

        if emb is None:
            index.append(
                {
                    "image": rel_path,
                    "label": label,
                    "success": False,
                    "error": "no embedding on detected face object",
                }
            )
            continue

        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 1:
            emb = emb.reshape(-1)

        emb_norm = np.linalg.norm(emb)
        if emb_norm > 0:
            emb = emb / emb_norm

        emb_file = output_dir / "embeddings" / f"{face_id}.npy"
        np.save(str(emb_file), emb)
        num_embedded += 1

        bbox = np.asarray(best.bbox, dtype=np.float32).tolist()
        det_score = float(getattr(best, "det_score", 0.0))

        index.append(
            {
                "image": rel_path,
                "label": label,
                "success": True,
                "face_id": face_id,
                "bbox": bbox,
                "det_score": det_score,
                "embedding": str(emb_file.relative_to(output_dir)).replace("\\", "/"),
                "dim": int(emb.shape[0]),
            }
        )

    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "model_root": str(model_root),
                "model_name": args.model_name,
                "num_images": num_images,
                "num_embedded": num_embedded,
                "items": index,
            },
            f,
            indent=2,
        )

    print(f"Done. images={num_images} embedded={num_embedded} index={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
