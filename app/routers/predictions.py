from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..auth import require_api_key
from ..config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from ..db import get_db
from ..errors import AppError
from ..helpers import get_or_404
from ..models import AnnotationStatus, RecognitionPrediction
from ..schemas import PredictionPatch

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
):
    prediction = get_or_404(db, RecognitionPrediction, prediction_id, "prediction")

    fields_set = body.model_fields_set
    if "final_instance_id" in fields_set:
        prediction.final_instance_id = body.final_instance_id
    if "final_bbox" in fields_set:
        prediction.final_bbox = body.final_bbox
    prediction.annotation_status = AnnotationStatus.CONFIRMED

    db.commit()
    db.refresh(prediction)
    return {
        "id": prediction.id,
        "annotation_status": prediction.annotation_status.value,
        "final_instance_id": prediction.final_instance_id,
        "final_bbox": prediction.final_bbox,
    }
