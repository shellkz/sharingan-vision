import onnxruntime as ort
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..auth import require_api_key
from ..config import SAMPLES_DIR
from ..db import get_db
from ..helpers import get_or_404
from ..models import RecognitionInstance, RecognitionSample
from ..sample_ops import create_sample, recrop_sample
from ..schemas import SampleBboxPatch
from ..sessions import get_embedding_session, get_objectness_session

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/instances/{instance_id}/samples", status_code=201)
async def add_sample(
    instance_id: int,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    objectness_session: ort.InferenceSession = Depends(get_objectness_session),
    embedding_session: ort.InferenceSession = Depends(get_embedding_session),
):
    get_or_404(db, RecognitionInstance, instance_id, "instance")
    raw = await image.read()
    sample = create_sample(db, instance_id, raw, objectness_session, embedding_session)
    db.commit()
    db.refresh(sample)
    return {"id": sample.id, "instance_id": sample.instance_id, "bbox": sample.final_bbox, "created_at": sample.created_at}


@router.get("/instances/{instance_id}/samples")
def list_samples(instance_id: int, db: Session = Depends(get_db)):
    get_or_404(db, RecognitionInstance, instance_id, "instance")
    samples = (
        db.query(RecognitionSample)
        .filter(RecognitionSample.instance_id == instance_id)
        .order_by(RecognitionSample.id)
        .all()
    )
    return {
        "result": [
            {"id": s.id, "instance_id": s.instance_id, "bbox": s.final_bbox, "created_at": s.created_at}
            for s in samples
        ]
    }


@router.get("/samples/{sample_id}/image")
def get_sample_image(sample_id: int, db: Session = Depends(get_db)):
    sample = get_or_404(db, RecognitionSample, sample_id, "sample")
    return FileResponse(SAMPLES_DIR / sample.image_path, media_type="image/jpeg")


@router.patch("/samples/{sample_id}")
def patch_sample(
    sample_id: int,
    body: SampleBboxPatch,
    db: Session = Depends(get_db),
    embedding_session: ort.InferenceSession = Depends(get_embedding_session),
):
    sample = get_or_404(db, RecognitionSample, sample_id, "sample")
    recrop_sample(sample, body.bbox, embedding_session)
    db.commit()
    db.refresh(sample)
    return {"id": sample.id, "instance_id": sample.instance_id, "bbox": sample.final_bbox}


@router.delete("/samples/{sample_id}", status_code=204)
def delete_sample(sample_id: int, db: Session = Depends(get_db)):
    sample = get_or_404(db, RecognitionSample, sample_id, "sample")
    (SAMPLES_DIR / sample.image_path).unlink(missing_ok=True)
    db.delete(sample)
    db.commit()
