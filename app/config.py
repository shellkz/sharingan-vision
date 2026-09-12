"""服務設定值:環境變數集中在這裡讀取,其他模組不直接碰 os.environ。

模型路徑(FASTSAM_ONNX_PATH/DINOV2_ONNX_PATH)、DATABASE_URL、VISION_API_KEY 開放環境變數,
因為每個部署的內容本來就會不同。圖片存放路徑(SAMPLES_DIR/SCANS_DIR)刻意不開環境變數,
固定是服務容器內的已知路徑,見 docs/schema.md「圖片實體檔案存放」的說明——第三方部署時只要
掛 volume 到這兩個固定路徑,不需要多記一個環境變數。
"""
import os
from pathlib import Path

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./data/vision.db")

VISION_API_KEY = os.environ.get("VISION_API_KEY")

SAMPLES_DIR = Path("data/samples")
SCANS_DIR = Path("data/scans")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_UPLOAD_LONG_EDGE = 8000  # px
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
