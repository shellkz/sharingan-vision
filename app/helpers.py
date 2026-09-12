from sqlalchemy.orm import Session

from .errors import AppError


def get_or_404(db: Session, model, obj_id: int, label: str):
    obj = db.get(model, obj_id)
    if obj is None:
        raise AppError(404, "not_found", f"{label} {obj_id} 不存在")
    return obj
