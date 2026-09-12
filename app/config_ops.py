"""recognition_config 是永遠只有一列的表,取用/建立第一列的邏輯集中在這裡,
routers/config.py 跟 routers/recognize.py 都要用到。"""
from sqlalchemy.orm import Session

from .models import RecognitionConfig


def get_or_create_config(db: Session) -> RecognitionConfig:
    config = db.get(RecognitionConfig, 1)
    if config is None:
        config = RecognitionConfig(id=1)
        db.add(config)
        db.commit()
        db.refresh(config)
    return config
