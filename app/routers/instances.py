import onnxruntime as ort
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..auth import require_api_key
from ..config import SAMPLES_DIR
from ..db import get_db
from ..errors import AppError
from ..helpers import get_or_404
from ..models import RecognitionEntity, RecognitionInstance, RecognitionSample
from ..sample_ops import create_sample
from ..sessions import get_embedding_session, get_objectness_session

router = APIRouter(dependencies=[Depends(require_api_key)])


def _sample_count(db: Session, instance_id: int) -> int:
    return (
        db.query(func.count(RecognitionSample.id))
        .filter(RecognitionSample.instance_id == instance_id)
        .scalar()
    )


@router.post("/entities/{entity_id}/instances", status_code=201)
async def create_instance(
    entity_id: int,
    name: str | None = Form(default=None),
    images: list[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
    objectness_session: ort.InferenceSession = Depends(get_objectness_session),
    embedding_session: ort.InferenceSession = Depends(get_embedding_session),
):
    entity = get_or_404(db, RecognitionEntity, entity_id, "entity")
    instance_name = name or entity.name

    exists = (
        db.query(RecognitionInstance)
        .filter(RecognitionInstance.entity_id == entity_id, RecognitionInstance.name == instance_name)
        .first()
    )
    if exists:
        raise AppError(409, "duplicate_name", f"instance '{instance_name}' already exists under entity {entity_id}")

    instance = RecognitionInstance(entity_id=entity_id, name=instance_name)
    db.add(instance)
    db.flush()  # 取得 instance.id, 還沒 commit

    created_samples = []
    for image_file in images:
        raw = await image_file.read()
        created_samples.append(create_sample(db, instance.id, raw, objectness_session, embedding_session))

    db.commit()
    db.refresh(instance)
    for sample in created_samples:
        db.refresh(sample)
    return {
        "id": instance.id,
        "entity_id": instance.entity_id,
        "name": instance.name,
        "created_at": instance.created_at,
        "samples": [{"id": s.id, "bbox": s.final_bbox} for s in created_samples],
    }


@router.get("/entities/{entity_id}/instances")
def list_instances(entity_id: int, db: Session = Depends(get_db)):
    get_or_404(db, RecognitionEntity, entity_id, "entity")
    instances = (
        db.query(RecognitionInstance)
        .filter(RecognitionInstance.entity_id == entity_id, RecognitionInstance.is_deleted.is_(False))
        .order_by(RecognitionInstance.id)
        .all()
    )
    return {
        "result": [
            {"id": i.id, "name": i.name, "sample_count": _sample_count(db, i.id)}
            for i in instances
        ]
    }


@router.delete("/instances/{instance_id}")
def delete_instance(instance_id: int, db: Session = Depends(get_db)):
    instance = get_or_404(db, RecognitionInstance, instance_id, "instance")

    samples = db.query(RecognitionSample).filter(RecognitionSample.instance_id == instance_id).all()
    for sample in samples:
        (SAMPLES_DIR / sample.image_path).unlink(missing_ok=True)
        db.delete(sample)

    instance.is_deleted = True
    db.commit()
    return {"id": instance.id, "is_deleted": True}
