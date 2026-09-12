"""DINOv2 embedding, 純 ONNX Runtime + PIL/numpy 推論與前處理, 完全不依賴
PyTorch/torchvision (執行期不用裝 torch, 只有當初匯出 .onnx 那次性動作需要)。

resize/center_crop 手刻邏輯對齊 torchvision 的計算方式(short-side resize 用
int() 無條件捨去、center crop 用 round()), 已跟 embedding.py(torch 版本)
的輸出驗證過幾乎一致。
"""

import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# 模型路徑決定順序: 呼叫 load_session() 時明確傳入 > DINOV2_ONNX_PATH 環境變數。
# 沒有內建預設路徑 —— 兩者都沒給就直接報錯, 不要悄悄 fallback 到可能不對的位置。

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_session(onnx_path: str | Path | None = None) -> ort.InferenceSession:
    """單純讀取並回傳一顆新的 session, 不快取。呼叫端(例如 FastAPI lifespan)
    自己決定要不要只建立一次、要放哪裡, 這裡不再用模組全域變數藏生命週期。"""
    path = onnx_path or os.environ.get("DINOV2_ONNX_PATH")
    if not path:
        raise ValueError(
            "找不到 DINOv2 ONNX 模型路徑: 沒有明確傳入 onnx_path, "
            "也沒有設定 DINOV2_ONNX_PATH 環境變數"
        )
    # 正式模型是動態 INT8 量化過的(QOperator 格式: DynamicQuantizeLinear/QLinearMatMul),
    # OpenVINOExecutionProvider 讀這個格式會直接炸掉("the graph is not acyclic"),不是
    # 優雅 fallback, 所以這裡固定只用 CPUExecutionProvider(取捨評估見 docs/影像辨識.md)。
    print(f"載入 DINOv2 ONNX Runtime session ({path}) ...")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"DINOv2 session 實際使用的 providers: {session.get_providers()}")
    return session


def to_rgb(img: Image.Image) -> Image.Image:
    """RGBA 一律合成到白底再轉 RGB, 跟參考圖的乾淨白底風格一致"""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def _resize_shorter_side(img: Image.Image, size: int = 256) -> Image.Image:
    """等同 torchvision transforms.Resize(size): 短邊縮到 size, 長邊等比例縮放"""
    w, h = img.size
    if w <= h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)
    return img.resize((new_w, new_h), Image.BICUBIC)


def _center_crop(img: Image.Image, size: int = 224) -> Image.Image:
    """等同 torchvision transforms.CenterCrop(size)"""
    w, h = img.size
    left = round((w - size) / 2)
    top = round((h - size) / 2)
    return img.crop((left, top, left + size, top + size))


def preprocess(img: Image.Image) -> np.ndarray:
    img = _resize_shorter_side(to_rgb(img), 256)
    img = _center_crop(img, 224)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    return arr[None].astype(np.float32)  # (1, 3, 224, 224)


def embed(img: Image.Image, session: ort.InferenceSession) -> np.ndarray:
    """session 由呼叫端明確傳入(例如 FastAPI lifespan 建立好的那顆), 這裡不負責建立或快取。"""
    x = preprocess(img)
    out = session.run(["embedding"], {"input": x})[0]
    return out[0]


def embed_batch(imgs: list[Image.Image], session: ort.InferenceSession) -> np.ndarray:
    """跟 embed() 同一套前處理, 但把多張圖疊成一個 batch 只呼叫一次 session.run(),
    用來 A/B 比較跟逐張呼叫 embed() 的耗時差異。回傳 shape (len(imgs), 384),
    沒有圖片就回傳 shape (0, 384) 的空陣列, 不呼叫 session。
    """
    if not imgs:
        return np.empty((0, 384), dtype=np.float32)

    t0 = time.perf_counter()
    x = np.concatenate([preprocess(img) for img in imgs], axis=0)  # (N, 3, 224, 224)
    t1 = time.perf_counter()
    out = session.run(["embedding"], {"input": x})[0]
    t2 = time.perf_counter()
    print(f"[metric]   前處理(resize+crop+normalize, {len(imgs)}張) 耗時: {(t1 - t0) * 1000:.1f} ms")
    print(f"[metric]   session.run() ONNX 推論 耗時: {(t2 - t1) * 1000:.1f} ms")

    return out
