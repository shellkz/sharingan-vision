import io

import onnxruntime as ort
from fastapi import APIRouter, Depends, Query
from PIL import Image
from sqlalchemy.orm import Session

from ..auth import require_api_key
from ..config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, SAMPLES_DIR, SCANS_DIR
from ..db import get_db
from ..errors import AppError
from ..helpers import get_or_404
from ..image_utils import save_image
from ..models import AnnotationStatus, RecognitionPrediction, RecognitionSample
from ..schemas import PredictionPatch
from ..sessions import get_embedding_session
from ..vision.embedding import embed

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/predictions")
def list_predictions(
    status: str | None = Query(default=None),
    image_path: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    db: Session = Depends(get_db),
):
    if status is not None and status not in (s.value for s in AnnotationStatus):
        raise AppError(400, "invalid_request", f"不合法的 status: {status}")

    query = db.query(RecognitionPrediction)
    if status is not None:
        query = query.filter(RecognitionPrediction.annotation_status == status)
    if image_path is not None:
        query = query.filter(RecognitionPrediction.image_path == image_path)

    total = query.count()
    rows = query.order_by(RecognitionPrediction.id).offset((page - 1) * limit).limit(limit).all()

    return {
        "result": [
            {
                "id": p.id,
                "image_path": p.image_path,
                "predicted_bbox": p.predicted_bbox,
                "predicted_instance_id": p.predicted_instance_id,
                "predicted_score": p.predicted_score,
                "annotation_status": p.annotation_status.value,
            }
            for p in rows
        ],
        "total": total,
    }


@router.patch("/predictions/{prediction_id}")
def patch_prediction(
    prediction_id: int,
    body: PredictionPatch,
    db: Session = Depends(get_db),
    embedding_session: ort.InferenceSession = Depends(get_embedding_session),
):
    prediction = get_or_404(db, RecognitionPrediction, prediction_id, "prediction")

    fields_set = body.model_fields_set
    if "final_instance_id" in fields_set:
        prediction.final_instance_id = body.final_instance_id
    if "final_bbox" in fields_set:
        prediction.final_bbox = body.final_bbox
    prediction.annotation_status = AnnotationStatus.CONFIRMED

    # 運作中優化(見 docs/schema.md):final_instance_id 有值時, 拿 final_bbox 對應的
    # 裁切圖重算 embedding, 寫進 recognition_samples, 下一次 /recognize 馬上生效。
    if prediction.final_instance_id is not None and prediction.final_bbox is not None:
        scan_path = SCANS_DIR / prediction.image_path
        if scan_path.exists():
            image = Image.open(scan_path).convert("RGB")
            x1, y1, x2, y2 = prediction.final_bbox
            crop = image.crop((x1, y1, x2, y2))
            vector = embed(crop, embedding_session)

            buffer = io.BytesIO()
            crop.save(buffer, format="JPEG")
            filename = save_image(buffer.getvalue(), SAMPLES_DIR)
            db.add(
                RecognitionSample(
                    instance_id=prediction.final_instance_id,
                    vector=vector.tolist(),
                    image_path=filename,
                )
            )

    db.commit()
    db.refresh(prediction)
    return {
        "id": prediction.id,
        "annotation_status": prediction.annotation_status.value,
        "final_instance_id": prediction.final_instance_id,
        "final_bbox": prediction.final_bbox,
    }
