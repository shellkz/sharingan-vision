from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..auth import require_api_key
from ..config_ops import get_or_create_config
from ..db import get_db
from ..models import RecognitionConfig
from ..schemas import ConfigPatch

router = APIRouter(dependencies=[Depends(require_api_key)])


def _serialize(config: RecognitionConfig) -> dict:
    return {
        "score_threshold": config.score_threshold,
        "crop_padding_ratio": config.crop_padding_ratio,
        "max_retained_scan_images": config.max_retained_scan_images,
    }


@router.get("/config")
def get_config(db: Session = Depends(get_db)):
    return _serialize(get_or_create_config(db))


@router.patch("/config")
def patch_config(body: ConfigPatch, db: Session = Depends(get_db)):
    config = get_or_create_config(db)
    for field in body.model_fields_set:
        setattr(config, field, getattr(body, field))
    db.commit()
    db.refresh(config)
    return _serialize(config)
