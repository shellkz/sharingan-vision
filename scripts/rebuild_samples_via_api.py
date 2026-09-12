"""把既有的 recognition_samples 全部透過真正的 API(DELETE + POST)重建一次,
套用新版「FastSAM 抓最大候選框 + 裁切後才 embed」的邏輯。適用情境:辦法生效之前
建立的舊樣本, schema migration 不會幫忙重算, 需要主動重建。

只用 API 操作 db 的方式做重建動作(DELETE /samples/{id}、POST /instances/{id}/samples),
但因為現在沒有「列出某個 instance 底下所有 sample id」的端點, 列清單這一步直接讀
資料庫(唯讀, 不寫入)。

用法:
    python scripts/rebuild_samples_via_api.py
"""
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

from fastapi.testclient import TestClient  # noqa: E402

from app.config import SAMPLES_DIR, VISION_API_KEY  # noqa: E402
from app.main import app  # noqa: E402

HEADERS = {"Authorization": f"Bearer {VISION_API_KEY}"}


def _area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


con = sqlite3.connect(ROOT / "data" / "vision.db")
cur = con.cursor()
cur.execute(
    """
    SELECT s.id, s.instance_id, s.image_path, i.name, e.name
    FROM recognition_samples s
    JOIN recognition_instances i ON i.id = s.instance_id
    JOIN recognition_entities e ON e.id = i.entity_id
    ORDER BY s.id
    """
)
rows = cur.fetchall()
con.close()

print(f"共 {len(rows)} 筆既有樣本要重建\n")

flagged = []

with TestClient(app) as client:
    for old_id, instance_id, image_path, instance_name, entity_name in rows:
        image_bytes = (SAMPLES_DIR / image_path).read_bytes()

        r = client.delete(f"/v1/samples/{old_id}", headers=HEADERS)
        assert r.status_code == 204, f"DELETE 失敗: {r.status_code} {r.text}"

        r = client.post(
            f"/v1/instances/{instance_id}/samples",
            headers=HEADERS,
            files={"image": (image_path, image_bytes, "image/jpeg")},
        )
        assert r.status_code == 201, f"POST 失敗: {r.status_code} {r.text}"
        new_sample = r.json()
        bbox = new_sample["bbox"]

        area_note = ""
        if bbox is not None and _area(bbox) < 50000:
            area_note = "  <-- 面積偏小, 建議肉眼確認裁切結果"
            flagged.append((new_sample["id"], entity_name, instance_name))

        print(
            f"[{entity_name}/{instance_name}] old_id={old_id} -> new_id={new_sample['id']}, "
            f"bbox={bbox}{area_note}"
        )

print(f"\n重建完成。{len(flagged)} 筆面積偏小, 建議檢查:")
for sample_id, entity_name, instance_name in flagged:
    print(f"  sample_id={sample_id} ({entity_name}/{instance_name})")
