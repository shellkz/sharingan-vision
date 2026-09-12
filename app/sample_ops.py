"""建立/修正 recognition_samples 的共用邏輯。instances.py(建立 instance 順便帶圖)、
samples.py(單獨追加一張圖、PATCH 修正)都會用到, 避免各自寫一份。

建樣本時先跑 FastSAM 抓物件的 bbox(挑面積最大的候選框), 裁切後才算 embedding——
不這樣做的話背景(常見是拍攝用的桌面)會稀釋掉物件本身的視覺特徵, 構圖類似的不同
物件容易在 embedding 空間裡意外地相近, 見 docs/影像辨識.md 第 6 節的診斷紀錄。
這個自動偵測不保證每次都對(低對比度物件可能被整個框錯), 所以留 predicted_bbox
記錄原始猜測、final_bbox 讓使用者事後用 PATCH /samples/{id} 修正。
"""
import onnxruntime as ort
from PIL import Image
from sqlalchemy.orm import Session

from .config import SAMPLES_DIR
from .image_utils import load_and_validate_image, save_image
from .models import RecognitionSample
from .vision.embedding import embed
from .vision.objectness import find_objects
from .vision.recognize import CROP_PADDING_RATIO, pad_bbox


def _area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _detect_and_crop(
    image: Image.Image, objectness_session: ort.InferenceSession, crop_padding_ratio: float
) -> tuple[Image.Image, list[float] | None]:
    """回傳(裁切後的圖, bbox)。找不到任何候選框時 fallback 回整張圖, bbox 是 None。"""
    detections = find_objects(image, objectness_session)
    if not detections:
        return image, None

    largest = max(detections, key=lambda d: _area(d.bbox))
    cx1, cy1, cx2, cy2 = pad_bbox(largest.bbox, crop_padding_ratio, image.size)
    crop = image.crop((cx1, cy1, cx2, cy2))
    return crop, [float(cx1), float(cy1), float(cx2), float(cy2)]


def create_sample(
    db: Session,
    instance_id: int,
    raw: bytes,
    objectness_session: ort.InferenceSession,
    embedding_session: ort.InferenceSession,
    crop_padding_ratio: float = CROP_PADDING_RATIO,
) -> RecognitionSample:
    image = load_and_validate_image(raw)
    filename = save_image(raw, SAMPLES_DIR)  # 存完整原圖, 之後 PATCH 修正時才有東西可以重裁

    crop, bbox = _detect_and_crop(image, objectness_session, crop_padding_ratio)
    vector = embed(crop, embedding_session)

    sample = RecognitionSample(
        instance_id=instance_id,
        vector=vector.tolist(),
        predicted_bbox=bbox,
        final_bbox=bbox,
        image_path=filename,
    )
    db.add(sample)
    return sample


def recrop_sample(
    sample: RecognitionSample, new_bbox: list[float], embedding_session: ort.InferenceSession
) -> None:
    """PATCH /samples/{id}: 拿存好的原圖用修正後的 bbox 重新裁切、重算 embedding。
    predicted_bbox 保留最初自動偵測的結果不動, 只更新 final_bbox/vector。"""
    image = Image.open(SAMPLES_DIR / sample.image_path).convert("RGB")
    x1, y1, x2, y2 = new_bbox
    crop = image.crop((x1, y1, x2, y2))
    vector = embed(crop, embedding_session)

    sample.final_bbox = list(new_bbox)
    sample.vector = vector.tolist()
