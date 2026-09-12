"""開發用的一次性工具: 驗證 INT8 動態量化對「實際分類結果」的影響, 不是單純比較
embedding 向量的 cosine similarity(那樣沒意義, 因為沒有跟 catalog 放在同一個
embedding 空間比較)。

做法: FP32 catalog 配 FP32 DINOv2、INT8 catalog(現場用 INT8 模型重建, 不動
catalog.py)配 INT8 DINOv2, 兩條「自洽」的 pipeline 各自跑一次分類, 比對同一批
候選框的 class_name/score 有沒有不同。FastSAM 沒被量化, 只跑一次共用同一批候選框,
確保兩邊比較的是完全一樣的裁切圖。
"""
import sys
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


def classify(vec: np.ndarray, catalog: dict[str, np.ndarray]) -> tuple[str, float]:
    scores = {name: _cosine_sim(vec, proto) for name, proto in catalog.items()}
    best_class = max(scores, key=scores.get)
    best_score = scores[best_class]
    label = best_class if best_score >= SCORE_THRESHOLD else "unknown"
    return label, best_score


fastsam_session = objectness.load_session(onnx_path=str(BACKEND_DIR / "models" / "FastSAM-s.onnx"))
fp32_session = embedding.load_session(onnx_path=str(BACKEND_DIR / "models" / "dinov2_vits14.onnx"))
int8_session = embedding.load_session(onnx_path=str(BACKEND_DIR / "models" / "dinov2_vits14_int8.onnx"))

print("建 FP32 catalog ...")
fp32_catalog = build_catalog(fp32_session)
print("建 INT8 catalog ...")
int8_catalog = build_catalog(int8_session)

total = 0
total_mismatch = 0

for photo in TEST_PHOTOS:
    print(f"\n=== {photo.name} ===")
    image = Image.open(photo).convert("RGB")
    W, H = image.size

    detections = objectness.find_objects(image, fastsam_session)

    crops = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = w * CROP_PADDING_RATIO, h * CROP_PADDING_RATIO
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(W, int(x2 + pad_x))
        cy2 = min(H, int(y2 + pad_y))
        crops.append(image.crop((cx1, cy1, cx2, cy2)))

    print(f"{'idx':>4} {'FP32 label':<24} {'FP32 score':>10}   {'INT8 label':<24} {'INT8 score':>10}   {'結果'}")
    mismatch = 0
    for i, crop in enumerate(crops):
        fp32_vec = embedding.embed(crop, fp32_session)
        int8_vec = embedding.embed(crop, int8_session)
        fp32_label, fp32_score = classify(fp32_vec, fp32_catalog)
        int8_label, int8_score = classify(int8_vec, int8_catalog)
        same = fp32_label == int8_label
        if not same:
            mismatch += 1
        print(
            f"{i:>4} {fp32_label:<24} {fp32_score:>10.3f}   {int8_label:<24} {int8_score:>10.3f}   "
            f"{'OK' if same else '*** DIFF ***'}"
        )

    print(f"這張圖: {len(crops)} 個候選框, {mismatch} 個分類結果不同")
    total += len(crops)
    total_mismatch += mismatch

print(f"\n總計: {total} 個候選框, {total_mismatch} 個分類結果不同({total_mismatch / total * 100:.1f}%)")
