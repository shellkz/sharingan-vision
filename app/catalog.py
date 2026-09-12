"""從資料庫組出 /recognize 要用的特徵池(見 app/vision/recognize.py 的 CatalogEntry)。
每次 /recognize 請求都重新查一次, 不做記憶體快取(見 docs/辨識服務部署方案.md:DINOv2
推論本身要 1 秒多, 查詢開銷可忽略, 順便繞開快取何時失效的問題)。
"""
import numpy as np
from sqlalchemy.orm import Session

from .models import RecognitionEntity, RecognitionInstance, RecognitionSample
from .vision.recognize import CatalogEntry


def build_catalog(db: Session) -> list[CatalogEntry]:
    rows = (
        db.query(RecognitionSample, RecognitionInstance, RecognitionEntity)
        .join(RecognitionInstance, RecognitionSample.instance_id == RecognitionInstance.id)
        .join(RecognitionEntity, RecognitionInstance.entity_id == RecognitionEntity.id)
        .filter(RecognitionInstance.is_deleted.is_(False))
        .filter(RecognitionEntity.is_deleted.is_(False))
        .all()
    )
    return [
        CatalogEntry(
            instance_id=instance.id,
            entity_id=entity.id,
            entity_name=entity.name,
            instance_name=instance.name,
            vector=np.array(sample.vector, dtype=np.float32),
        )
        for sample, instance, entity in rows
    ]
