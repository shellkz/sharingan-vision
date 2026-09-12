import os
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from alembic import command
from alembic.config import Config
from fastapi import FastAPI

from .errors import register_error_handlers
from .routers import config, entities, instances, predictions, recognize, samples
from .vision import embedding as vision_embedding
from .vision import objectness as vision_objectness

SERVICE_ROOT = Path(__file__).resolve().parent.parent


def _memory_mb() -> float:
    """目前 process 的常駐記憶體(RSS), MB。量測用, 跟業務邏輯無關。"""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _run_migrations() -> None:
    """開機時自動跑到最新 schema, 部署端不用另外執行 alembic upgrade head,
    見 docs/辨識服務部署方案.md 對 SQLite 方案的說明。"""
    cfg = Config(str(SERVICE_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(SERVICE_ROOT / "alembic"))
    command.upgrade(cfg, "head")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _run_migrations()

    print(f"[metric] 任何模型載入前記憶體用量(基準): {_memory_mb():.1f} MB")

    app.state.fastsam_session = vision_objectness.load_session()
    app.state.dinov2_session = vision_embedding.load_session()

    print(f"[metric] 模型載入後記憶體用量: {_memory_mb():.1f} MB")

    yield


app = FastAPI(lifespan=lifespan)

register_error_handlers(app)

for router in (
    recognize.router,
    entities.router,
    instances.router,
    samples.router,
    predictions.router,
    config.router,
):
    app.include_router(router, prefix="/v1")
