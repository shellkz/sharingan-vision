import onnxruntime as ort
from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from ..auth import require_api_key
from ..catalog import build_catalog
from ..cleanup import evict_old_scan_images
from ..config import SCANS_DIR
from ..config_ops import get_or_create_config
from ..db import get_db
from ..image_utils import load_and_validate_image, save_image
from ..models import RecognitionConfig, RecognitionPrediction
from ..sessions import get_embedding_session, get_objectness_session
from ..vision.recognize import recognize as run_recognize

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    objectness_session: ort.InferenceSession = Depends(get_objectness_session),
    embedding_session: ort.InferenceSession = Depends(get_embedding_session),
):
    raw = await file.read()
    image = load_and_validate_image(raw)
    scan_filename = save_image(raw, SCANS_DIR)

    catalog = build_catalog(db)
    config: RecognitionConfig = get_or_create_config(db)

    results = run_recognize(
        image,
        catalog,
        objectness_session,
        embedding_session,
        score_threshold=config.score_threshold,
        crop_padding_ratio=config.crop_padding_ratio,
    )

    predictions = []
    for r in results:
        prediction = RecognitionPrediction(
            image_path=scan_filename,
            predicted_instance_id=r.instance_id,
            predicted_score=r.score,
            predicted_bbox=list(r.bbox),
        )
        db.add(prediction)
        predictions.append(prediction)
    db.commit()

    evict_old_scan_images(db)

    return {
        "score_threshold": config.score_threshold,
        "result": [
            {
                "bbox": list(r.bbox),
                "instance_id": r.instance_id,
                "instance_name": r.instance_name,
                "entity_id": r.entity_id,
                "entity_name": r.entity_name,
                "score": r.score,
                "meets_threshold": r.meets_threshold,
                "prediction_id": p.id,
            }
            for r, p in zip(results, predictions)
        ],
    }
