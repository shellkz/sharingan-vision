"""掃描圖清除規則,見 docs/schema.md「掃描圖清除規則」。只清 confirmed 的紀錄,
pending_review 永遠不動;只刪圖片檔案本體,recognition_predictions 的資料列不刪
(image_path 變成死連結,但 predicted_*/final_* 等欄位還在,不影響訓練資料匯出統計)。

在每次 /recognize 寫入新的 prediction 之後呼叫,不用另外的排程機制。
"""
from sqlalchemy.orm import Session

from .config import SCANS_DIR
from .models import AnnotationStatus, RecognitionConfig, RecognitionPrediction


def evict_old_scan_images(db: Session) -> None:
    config = db.get(RecognitionConfig, 1)
    limit = config.max_retained_scan_images if config else 10000

    confirmed = (
        db.query(RecognitionPrediction)
        .filter(RecognitionPrediction.annotation_status == AnnotationStatus.CONFIRMED)
        .order_by(RecognitionPrediction.created_at.desc())
        .all()
    )

    for prediction in confirmed[limit:]:
        path = SCANS_DIR / prediction.image_path
        path.unlink(missing_ok=True)
