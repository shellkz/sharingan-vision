"""service-to-service 共用金鑰驗證,見 docs/辨識服務API.md「認證」。
金鑰由部署端自己產生、設進 VISION_API_KEY 環境變數,這個服務不簽發也不管理金鑰。
"""
from fastapi import Header

from .config import VISION_API_KEY
from .errors import AppError


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not VISION_API_KEY:
        return  # 沒設定金鑰時視為開發模式,不擋(部署端務必記得設定)

    expected = f"Bearer {VISION_API_KEY}"
    if authorization != expected:
        raise AppError(401, "unauthorized", "缺少或錯誤的 Authorization header")
