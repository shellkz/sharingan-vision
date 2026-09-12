"""手動跑一次的煙霧測試, 確認新的 API surface 串得起來(entities -> instances(+samples,
含自動裁切 bbox + PATCH 修正) -> recognize -> predictions -> config)。不是正式測試套件,
用完即丟, 之後可以刪除或搬進 tests/ 正式化。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("FASTSAM_ONNX_PATH", str(ROOT / "models" / "FastSAM-s_640_fp32.onnx"))
os.environ.setdefault("DINOV2_ONNX_PATH", str(ROOT / "models" / "dinov2_vits14_batch_int8.onnx"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{ROOT.as_posix()}/data/smoke_test.db")

db_file = ROOT / "data" / "smoke_test.db"
db_file.unlink(missing_ok=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

FIXTURES = ROOT / "test" / "fixtures"

# 使用者實際量過的正確答案(peg_purple_1.jpg), 見對話紀錄/docs/影像辨識.md 第 6 節
PEG_PURPLE_1_CORRECT_BBOX = [800.0, 1920.0, 2125.0, 2550.0]

# 不用手動建表: lifespan 開機時會自動跑 alembic upgrade head(見 app/main.py)

with TestClient(app) as client:
    print("== POST /v1/entities (peg) ==")
    r = client.post("/v1/entities", json={"name": "peg"})
    print(r.status_code, r.json())
    assert r.status_code == 201
    entity_id = r.json()["id"]

    print("== POST /v1/entities (duplicate) ==")
    r = client.post("/v1/entities", json={"name": "peg"})
    print(r.status_code, r.json())
    assert r.status_code == 409

    print("== POST /v1/entities/{id}/instances (blue, 2 張圖) ==")
    blue_files = [
        ("images", (p.name, open(p, "rb"), "image/jpeg"))
        for p in sorted(FIXTURES.glob("peg_blue_*.jpg"))[:2]
    ]
    r = client.post(f"/v1/entities/{entity_id}/instances", data={"name": "blue"}, files=blue_files)
    print(r.status_code, r.json())
    assert r.status_code == 201
    blue_instance_id = r.json()["id"]
    assert len(r.json()["samples"]) == 2
    for s in r.json()["samples"]:
        print("  blue sample bbox:", s["bbox"])

    print("== POST /v1/entities/{id}/instances (purple, 1 張圖, 已知這張的自動偵測會框錯) ==")
    purple_files = [("images", ("peg_purple_1.jpg", open(FIXTURES / "peg_purple_1.jpg", "rb"), "image/jpeg"))]
    r = client.post(f"/v1/entities/{entity_id}/instances", data={"name": "purple"}, files=purple_files)
    print(r.status_code, r.json())
    assert r.status_code == 201
    purple_instance_id = r.json()["id"]
    purple_sample = r.json()["samples"][0]
    purple_sample_id = purple_sample["id"]
    print("  purple sample 自動偵測 bbox(預期是錯的, 面積偏小):", purple_sample["bbox"])

    print("== PATCH /v1/samples/{id} (修正成使用者量測的正確 bbox) ==")
    r = client.patch(f"/v1/samples/{purple_sample_id}", json={"bbox": PEG_PURPLE_1_CORRECT_BBOX})
    print(r.status_code, r.json())
    assert r.status_code == 200
    assert r.json()["bbox"] == PEG_PURPLE_1_CORRECT_BBOX

    print("== GET /v1/entities/{entity_id}/instances ==")
    r = client.get(f"/v1/entities/{entity_id}/instances")
    print(r.status_code, r.json())
    assert r.status_code == 200
    assert len(r.json()["result"]) == 2

    print("== POST /v1/recognize (scene_1.jpg) ==")
    with open(FIXTURES / "scene_1.jpg", "rb") as f:
        r = client.post("/v1/recognize", files={"file": ("scene_1.jpg", f, "image/jpeg")})
    print(r.status_code, "result count:", len(r.json()["result"]))
    assert r.status_code == 200
    for item in r.json()["result"]:
        if item["entity_name"] != "unknown":
            print("  ", item["entity_name"], item["instance_name"], round(item["score"], 2))

    print("== GET /v1/predictions ==")
    r = client.get("/v1/predictions")
    print(r.status_code, "total:", r.json()["total"])
    assert r.status_code == 200
    first_prediction = r.json()["result"][0]

    print("== PATCH /v1/predictions/{id} ==")
    r = client.patch(
        f"/v1/predictions/{first_prediction['id']}",
        json={"final_instance_id": blue_instance_id, "final_bbox": first_prediction["predicted_bbox"]},
    )
    print(r.status_code, r.json())
    assert r.status_code == 200
    assert r.json()["annotation_status"] == "confirmed"

    print("== GET /v1/config ==")
    r = client.get("/v1/config")
    print(r.status_code, r.json())
    assert r.status_code == 200

    print("== PATCH /v1/config ==")
    r = client.patch("/v1/config", json={"score_threshold": 0.35})
    print(r.status_code, r.json())
    assert r.status_code == 200
    assert r.json()["score_threshold"] == 0.35

    print("== DELETE /v1/instances/{id} (purple) ==")
    r = client.delete(f"/v1/instances/{purple_instance_id}")
    print(r.status_code, r.json())
    assert r.status_code == 200
    assert r.json()["is_deleted"] is True

    print("== DELETE /v1/entities/{id} ==")
    r = client.delete(f"/v1/entities/{entity_id}")
    print(r.status_code, r.json())
    assert r.status_code == 200

    print("== GET /v1/entities/{id} after soft delete (should still be 200) ==")
    r = client.get(f"/v1/entities/{entity_id}")
    print(r.status_code, r.json())
    assert r.status_code == 200
    assert r.json()["is_deleted"] is True

    print("== GET /v1/entities/99999 (never existed, should 404) ==")
    r = client.get("/v1/entities/99999")
    print(r.status_code, r.json())
    assert r.status_code == 404

print("\nALL SMOKE TESTS PASSED")
