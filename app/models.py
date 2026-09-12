"""辨識服務自己的資料表(recognition_entities/instances/samples/predictions/config)。
對應 docs/schema.md「辨識服務自己的表」章節,是這份設計文件的實作。這個服務完全不知道
consumer(例如庫存 demo app)的 items/inventory 這些表存在,見同一份文件的說明。
"""
import enum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class AnnotationStatus(str, enum.Enum):
    PENDING_REVIEW = "pending_review"
    CONFIRMED = "confirmed"


def _enum_values(py_enum):
    return Enum(py_enum, values_callable=lambda e: [member.value for member in e])


class RecognitionEntity(Base):
    """概念層(同一個東西),例如「led」。見 docs/schema.md。"""

    __tablename__ = "recognition_entities"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RecognitionInstance(Base):
    """變體(具體實例),例如「led_front」。不需要區分變體時可以跟 entity 同名。"""

    __tablename__ = "recognition_instances"
    __table_args__ = (UniqueConstraint("entity_id", "name", name="uq_instance_entity_name"),)

    id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey("recognition_entities.id"), nullable=False)
    name = Column(String, nullable=False)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RecognitionSample(Base):
    """特徵池:一批向量,不是單一平均後的 prototype。見 docs/schema.md 的討論。"""

    __tablename__ = "recognition_samples"

    id = Column(Integer, primary_key=True)
    instance_id = Column(Integer, ForeignKey("recognition_instances.id"), nullable=False)
    vector = Column(JSON, nullable=False)  # list[float],SQLite 沒有原生向量型態
    predicted_bbox = Column(JSON, nullable=True)  # FastSAM 自動偵測的原始結果, 寫入後不變
    final_bbox = Column(JSON, nullable=True)  # 目前實際生效、拿去算 vector 的區域, 可被 PATCH 修正
    image_path = Column(String, nullable=False)  # 相對檔名,實體檔案在 SAMPLES_DIR,存完整原圖不是裁切後的結果
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RecognitionPrediction(Base):
    """預測紀錄,兼未來 fine-tune 資料池。見 docs/schema.md。"""

    __tablename__ = "recognition_predictions"

    id = Column(Integer, primary_key=True)
    image_path = Column(String, nullable=False)  # 相對檔名,實體檔案在 SCANS_DIR
    predicted_instance_id = Column(Integer, ForeignKey("recognition_instances.id"), nullable=True)
    predicted_score = Column(Float, nullable=True)
    predicted_bbox = Column(JSON, nullable=True)  # [x1, y1, x2, y2]
    final_instance_id = Column(Integer, ForeignKey("recognition_instances.id"), nullable=True)
    final_bbox = Column(JSON, nullable=True)
    annotation_status = Column(
        _enum_values(AnnotationStatus), nullable=False, default=AnnotationStatus.PENDING_REVIEW
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RecognitionConfig(Base):
    """執行期可調參數,永遠只有一列(id 固定為 1)。見 docs/schema.md。"""

    __tablename__ = "recognition_config"

    id = Column(Integer, primary_key=True)
    score_threshold = Column(Float, nullable=False, default=0.6)
    crop_padding_ratio = Column(Float, nullable=False, default=0.1)
    max_retained_scan_images = Column(Integer, nullable=False, default=10000)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
