from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..auth import require_api_key
from ..config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from ..db import get_db
from ..errors import AppError
from ..models import RecognitionEntity, RecognitionInstance
from ..schemas import EntityCreate

router = APIRouter(dependencies=[Depends(require_api_key)])


def _instance_count(db: Session, entity_id: int) -> int:
    return (
        db.query(func.count(RecognitionInstance.id))
        .filter(RecognitionInstance.entity_id == entity_id)
        .filter(RecognitionInstance.is_deleted.is_(False))
        .scalar()
    )


def _get_entity_or_404(db: Session, entity_id: int) -> RecognitionEntity:
    entity = db.get(RecognitionEntity, entity_id)
    if entity is None:
        raise AppError(404, "not_found", f"entity {entity_id} 不存在")
    return entity


@router.post("/entities", status_code=201)
def create_entity(body: EntityCreate, db: Session = Depends(get_db)):
    if db.query(RecognitionEntity).filter(RecognitionEntity.name == body.name).first():
        raise AppError(409, "duplicate_name", f"entity '{body.name}' already exists")

    entity = RecognitionEntity(name=body.name)
    db.add(entity)
    db.commit()
    db.refresh(entity)
    return {
        "id": entity.id,
        "name": entity.name,
        "instance_count": 0,
        "created_at": entity.created_at,
    }


@router.get("/entities")
def list_entities(
    include_deleted: bool = False,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    db: Session = Depends(get_db),
):
    query = db.query(RecognitionEntity)
    if not include_deleted:
        query = query.filter(RecognitionEntity.is_deleted.is_(False))

    total = query.count()
    entities = query.order_by(RecognitionEntity.id).offset((page - 1) * limit).limit(limit).all()

    return {
        "result": [
            {
                "id": entity.id,
                "name": entity.name,
                "instance_count": _instance_count(db, entity.id),
            }
            for entity in entities
        ],
        "total": total,
    }


@router.get("/entities/{entity_id}")
def get_entity(entity_id: int, db: Session = Depends(get_db)):
    entity = _get_entity_or_404(db, entity_id)
    return {
        "id": entity.id,
        "name": entity.name,
        "instance_count": _instance_count(db, entity.id),
        "is_deleted": entity.is_deleted,
        "created_at": entity.created_at,
    }


@router.delete("/entities/{entity_id}")
def delete_entity(entity_id: int, db: Session = Depends(get_db)):
    entity = _get_entity_or_404(db, entity_id)
    entity.is_deleted = True
    db.query(RecognitionInstance).filter(RecognitionInstance.entity_id == entity_id).update(
        {"is_deleted": True}
    )
    db.commit()
    return {"id": entity.id, "name": entity.name, "is_deleted": True}
