"""串起 objectness(找候選物件)+ embedding(比對已知類別)的完整辨識流程。
執行期完全不依賴 PyTorch/torchvision/ultralytics, 只靠 onnxruntime + numpy + PIL。

比對用特徵池(catalog: list[CatalogEntry]),對每個候選框跟池裡每一筆向量算相似度、
取全域最高分——不管這筆最高分向量來自哪個 instance,天然就是 pool-based matching
(見 docs/schema.md 的討論:拆不拆 instance 對準確度沒有影響,只影響能不能額外
取得「比對到哪個變體」)。

永遠回傳全域最高分的身份猜測(catalog 非空的話), 不因為低於 score_threshold
就強制改成 unknown——判斷「這個分數算不算數」的責任交給呼叫端, 服務本身只透過
每筆候選框的 meets_threshold 欄位、以及頂層的 score_threshold 給建議值。
entity_name="unknown" 只在 catalog 完全沒有樣本可比對時才出現。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .embedding import embed
from .objectness import find_objects

CROP_PADDING_RATIO = 0.1  # 裁切候選框時往外多留一點, 避免切到邊緣(recognition_config 沒設定時的預設值)
SCORE_THRESHOLD = 0.6  # 最高分低於此值就判定 unknown(recognition_config 沒設定時的預設值)


@dataclass
class CatalogEntry:
    instance_id: int
    entity_id: int
    entity_name: str
    instance_name: str
    vector: np.ndarray


@dataclass
class RecognizedObject:
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2), 原圖座標系
    instance_id: int | None
    instance_name: str | None
    entity_id: int | None
    entity_name: str  # catalog 是空的(完全沒有樣本)才會是 "unknown", 否則永遠是最佳猜測身份
    score: float  # 全域最高分(不管有沒有超過 threshold)
    meets_threshold: bool  # best_score >= score_threshold, 判斷責任交給呼叫端, 這裡只給建議值


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pad_bbox(
    bbox: tuple[float, float, float, float], padding_ratio: float, image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """bbox 往外多留一點(框自身寬高的比例,不是像素、不是整張圖比例),clip 到圖片範圍內。
    recognize() 的候選框裁切、sample_ops.py 建樣本時的自動裁切都用這個, 邏輯保持一致。"""
    W, H = image_size
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * padding_ratio, h * padding_ratio
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(W, int(x2 + pad_x))
    cy2 = min(H, int(y2 + pad_y))
    return cx1, cy1, cx2, cy2


def recognize(
    image: Image.Image,
    catalog: list[CatalogEntry],
    objectness_session: ort.InferenceSession,
    embedding_session: ort.InferenceSession,
    score_threshold: float = SCORE_THRESHOLD,
    crop_padding_ratio: float = CROP_PADDING_RATIO,
) -> list[RecognizedObject]:
    """對圖片跑完整辨識流程, 只回傳資料, 不畫圖。吃已經載入好的 PIL Image(RGB)
    —— 例如 FastAPI 收到上傳檔案的 bytes 包成 Image, 不用先存成暫存檔。

    catalog: 拿來比對的特徵池, 呼叫端自己決定怎麼準備(例如查詢 recognition_samples
    joined recognition_instances/recognition_entities, 排除 is_deleted 的列)。

    objectness_session/embedding_session: 呼叫端明確傳入的 ONNX session(例如
    FastAPI lifespan 只建立一次的那兩顆), 這裡不負責建立或快取, 也沒有預設值
    ——漏傳會直接 TypeError, 不會悄悄現場重載模型。
    """
    detections = find_objects(image, objectness_session)

    crops = []
    for det in detections:
        cx1, cy1, cx2, cy2 = pad_bbox(det.bbox, crop_padding_ratio, image.size)
        crops.append(image.crop((cx1, cy1, cx2, cy2)))

    vecs = [embed(crop, embedding_session) for crop in crops]

    results = []
    for det, vec in zip(detections, vecs):
        if catalog:
            best_score, best_entry = max(
                ((_cosine_sim(vec, entry.vector), entry) for entry in catalog),
                key=lambda pair: pair[0],
            )
        else:
            best_score, best_entry = 0.0, None

        if best_entry is not None:
            results.append(
                RecognizedObject(
                    bbox=det.bbox,
                    instance_id=best_entry.instance_id,
                    instance_name=best_entry.instance_name,
                    entity_id=best_entry.entity_id,
                    entity_name=best_entry.entity_name,
                    score=best_score,
                    meets_threshold=best_score >= score_threshold,
                )
            )
        else:
            results.append(
                RecognizedObject(
                    bbox=det.bbox,
                    instance_id=None,
                    instance_name=None,
                    entity_id=None,
                    entity_name="unknown",
                    score=best_score,
                    meets_threshold=False,
                )
            )

    return results


def recognize_file(
    image_path: str | Path,
    catalog: list[CatalogEntry],
    objectness_session: ort.InferenceSession,
    embedding_session: ort.InferenceSession,
    score_threshold: float = SCORE_THRESHOLD,
    crop_padding_ratio: float = CROP_PADDING_RATIO,
) -> list[RecognizedObject]:
    """方便的路徑版本: 讀檔案 -> 轉 RGB -> 呼叫 recognize()。CLI/測試腳本用這個比較順手。"""
    image = Image.open(image_path).convert("RGB")
    return recognize(image, catalog, objectness_session, embedding_session, score_threshold, crop_padding_ratio)
