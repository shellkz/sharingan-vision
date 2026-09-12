"""開發用的一次性工具: 比較 FastSAM 在 imgsz=1024(正式版) vs imgsz=640(候選)下
的偵測/分類結果差異。

因為兩種解析度的候選框集合本質上不是同一批(不同 grid、不同 NMS 結果), 沒辦法像
DINOv2 量化那次一樣逐一配對比較, 這裡改成看整體統計: 候選框總數、containment 去重
後的數量、以及套用現有 catalog(INT8 DINOv2, 跟正式環境一致)分類後每個已知類別的
候選框數量有沒有差很多。
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.vision import embedding, objectness  # noqa: E402
from app.vision.recognize import CROP_PADDING_RATIO, SCORE_THRESHOLD, _cosine_sim  # noqa: E402

IMAGES_DIR = BACKEND_DIR / "app" / "vision" / "images"
TEST_PHOTOS = [
    BACKEND_DIR.parent / "etc" / "eletronics_1.jpg",
    BACKEND_DIR.parent / "etc" / "eletronics_2.jpg",
]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

FASTSAM_1024_PATH = BACKEND_DIR / "models" / "FastSAM-s.onnx"
FASTSAM_640_PATH = BACKEND_DIR / "models" / "_fastsam_640_export" / "FastSAM-s.onnx"
DINOV2_PATH = BACKEND_DIR / "models" / "dinov2_vits14.onnx"  # 現在正式版是 INT8


def build_catalog(session) -> dict[str, np.ndarray]:
    result = {}
    for class_dir in sorted(IMAGES_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        files = sorted(
            (p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.stem.isdigit()),
            key=lambda p: int(p.stem),
        )
        if not files:
            continue
        vecs = [embedding.embed(Image.open(p), session) for p in files]
        result[class_dir.name] = np.stack(vecs).mean(axis=0)
    return result


def classify(vec: np.ndarray, catalog: dict[str, np.ndarray]) -> str:
    scores = {name: _cosine_sim(vec, proto) for name, proto in catalog.items()}
    best_class = max(scores, key=scores.get)
    return best_class if scores[best_class] >= SCORE_THRESHOLD else "unknown"


def run_fastsam(fastsam_session, dino_session, catalog, photo: Path, imgsz: int) -> Counter:
    image = Image.open(photo).convert("RGB")
    W, H = image.size
    objectness.IMGSZ = imgsz  # find_objects() 內部靠這個模組常數做 letterbox, 這裡臨時切換
    detections = objectness.find_objects(image, fastsam_session)

    counts = Counter()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = w * CROP_PADDING_RATIO, h * CROP_PADDING_RATIO
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(W, int(x2 + pad_x))
        cy2 = min(H, int(y2 + pad_y))
        crop = image.crop((cx1, cy1, cx2, cy2))
        vec = embedding.embed(crop, dino_session)
        label = classify(vec, catalog)
        counts[label] += 1
    return counts


dino_session = embedding.load_session(onnx_path=str(DINOV2_PATH))
catalog = build_catalog(dino_session)

fastsam_1024 = objectness.load_session(onnx_path=str(FASTSAM_1024_PATH))
fastsam_640 = objectness.load_session(onnx_path=str(FASTSAM_640_PATH))

for photo in TEST_PHOTOS:
    print(f"\n=== {photo.name} ===")
    print("-- imgsz=1024(正式版) --")
    counts_1024 = run_fastsam(fastsam_1024, dino_session, catalog, photo, imgsz=1024)
    print("-- imgsz=640(候選) --")
    counts_640 = run_fastsam(fastsam_640, dino_session, catalog, photo, imgsz=640)

    all_labels = sorted(set(counts_1024) | set(counts_640))
    print(f"\n{'類別':<24} {'1024':>6} {'640':>6}")
    for label in all_labels:
        print(f"{label:<24} {counts_1024.get(label, 0):>6} {counts_640.get(label, 0):>6}")
    print(f"{'總計':<24} {sum(counts_1024.values()):>6} {sum(counts_640.values()):>6}")
