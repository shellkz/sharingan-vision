"""JSON request body 驗證用的 Pydantic models。回應形狀直接在各路由組成 dict 回傳
(不用 response_model),因為多數輸出欄位是查詢算出來的(instance_count/sample_count),
不是 ORM 物件本身的欄位,硬套 response_model 反而要多繞一層。
"""
from pydantic import BaseModel


class EntityCreate(BaseModel):
    name: str


class SampleBboxPatch(BaseModel):
    bbox: list[float]


class PredictionPatch(BaseModel):
    final_instance_id: int | None = None
    final_bbox: list[float] | None = None


class ConfigPatch(BaseModel):
    score_threshold: float | None = None
    crop_padding_ratio: float | None = None
    max_retained_scan_images: int | None = None
