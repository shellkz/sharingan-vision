"""圖片上傳驗證與落地儲存。限制見 docs/辨識服務API.md「圖片上傳限制」。"""
import io
import uuid
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from .config import ALLOWED_IMAGE_FORMATS, MAX_UPLOAD_BYTES, MAX_UPLOAD_LONG_EDGE
from .errors import AppError


def load_and_validate_image(raw: bytes) -> Image.Image:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise AppError(413, "image_too_large", f"檔案大小超過上限 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError:
        raise AppError(415, "unsupported_image_format", "無法解碼圖片,僅支援 JPEG/PNG")

    if image.format not in ALLOWED_IMAGE_FORMATS:
        raise AppError(415, "unsupported_image_format", f"不支援的圖片格式: {image.format},僅支援 JPEG/PNG")

    if max(image.size) > MAX_UPLOAD_LONG_EDGE:
        raise AppError(413, "image_too_large", f"圖片長邊超過上限 {MAX_UPLOAD_LONG_EDGE}px")

    return image.convert("RGB")


async def read_and_validate_upload(file: UploadFile) -> tuple[Image.Image, bytes]:
    raw = await file.read()
    image = load_and_validate_image(raw)
    return image, raw


def save_image(raw: bytes, directory: Path) -> str:
    """存進固定資料夾,回傳相對檔名(不含資料夾路徑)。一律存成 .jpg,跟原始格式無關,
    避免同一批圖片裡 jpg/png 混雜造成檔名比對麻煩。"""
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image.save(directory / filename, format="JPEG")
    return filename
