"""透過真正在跑的 container(走 HTTP, 不是 TestClient in-process)重建 peg/bottle_cap
這組展示資料。data/vision.db 之前不小心被清空過一次(圖片檔案被誤刪, 見對話紀錄),
這隻腳本從 test/fixtures/ 的原始照片重新建一次 entity/instance/samples, 順便驗證
container 真的能正常回應。

用法(container 要先跑起來, 見 docs/辨識服務部署方案.md):
    python scripts/seed_demo_data.py
"""
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "test" / "fixtures"

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

API_KEY = os.environ["VISION_API_KEY"]
BASE_URL = "http://localhost:8000/v1"

client = httpx.Client(base_url=BASE_URL, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=60.0)


def create_entity(name: str) -> int:
    r = client.post("/entities", json={"name": name})
    r.raise_for_status()
    entity_id = r.json()["id"]
    print(f"entity {name} -> id={entity_id}")
    return entity_id


def create_instance(entity_id: int, name: str, image_paths: list[Path]) -> int:
    files = [("images", (p.name, p.open("rb"), "image/jpeg")) for p in image_paths]
    r = client.post(f"/entities/{entity_id}/instances", data={"name": name}, files=files)
    r.raise_for_status()
    body = r.json()
    print(f"  instance {name} -> id={body['id']}")
    for s in body["samples"]:
        note = "  <-- 面積偏小, 建議肉眼確認" if s["bbox"] and _small(s["bbox"]) else ""
        print(f"    sample id={s['id']} bbox={s['bbox']}{note}")
    return body["id"]


def _small(bbox: list[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1) < 50000


peg_id = create_entity("peg")
create_instance(peg_id, "blue", sorted(FIXTURES.glob("peg_blue_*.jpg")))
create_instance(peg_id, "purple", sorted(FIXTURES.glob("peg_purple_*.jpg")))

cap_id = create_entity("bottle_cap")
create_instance(cap_id, "green", sorted(FIXTURES.glob("bottle_cap_green_*.jpg")))
create_instance(cap_id, "red", sorted(FIXTURES.glob("bottle_cap_red_*.jpg")))
create_instance(cap_id, "white", sorted(FIXTURES.glob("bottle_cap_white_*.jpg")))

print("\n重建完成。發現面積偏小的 sample 記得用 PATCH /samples/{id} 修正(見 peg_purple_1.jpg 那個已知案例)。")
