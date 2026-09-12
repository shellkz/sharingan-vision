"""Objectness: FastSAM ONNX Runtime 推論 + 後處理, 找出圖片中的候選物件。
執行期完全不依賴 PyTorch/ultralytics, 只靠 onnxruntime + numpy + PIL。

流程: letterbox 前處理 -> 信心過濾 -> NMS -> proto 解析度 mask 解碼 -> containment
去重(NMS 抓不到巢狀包含的重複, 額外用「交集/較小面積」比例判斷) -> 座標換算回原圖。
設計細節與效能實驗數字見 docs/影像辨識.md。
"""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# 模型路徑決定順序: 呼叫 load_session() 時明確傳入 > FASTSAM_ONNX_PATH 環境變數。
# 沒有內建預設路徑 —— 兩者都沒給就直接報錯, 不要悄悄 fallback 到可能不對的位置。
IMGSZ = 640
CONF_THRESH = 0.4
IOU_THRESH = 0.6
CONTAINMENT_THRESH = 0.85

@dataclass
class DetectedObject:
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2), 原圖座標系
    objectness_conf: float  # FastSAM 自己對這個候選框的信心分數


def load_session(onnx_path: str | Path | None = None) -> ort.InferenceSession:
    """單純讀取並回傳一顆新的 session, 不快取。呼叫端(例如 FastAPI lifespan)
    自己決定要不要只建立一次、要放哪裡, 這裡不再用模組全域變數藏生命週期。"""
    path = onnx_path or os.environ.get("FASTSAM_ONNX_PATH")
    if not path:
        raise ValueError(
            "找不到 FastSAM ONNX 模型路徑: 沒有明確傳入 onnx_path, "
            "也沒有設定 FASTSAM_ONNX_PATH 環境變數"
        )
    print(f"載入 FastSAM ONNX Runtime session ({path}) ...")
    session = ort.InferenceSession(
        str(path), providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    )
    print(f"FastSAM session 實際使用的 providers: {session.get_providers()}")
    return session


def _letterbox(img: Image.Image, new_size: int = IMGSZ, color=(114, 114, 114)):
    """等比例縮放 + 補邊, 回傳 (letterbox後的圖, scale, pad_x, pad_y, 縮放後不含padding的寬高)"""
    w, h = img.size
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (new_size, new_size), color)
    pad_x, pad_y = (new_size - nw) // 2, (new_size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y, nw, nh


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """貪婪 NMS, 邏輯跟 torchvision.ops.nms 一致: 分數高的留, 跟它 IoU 超過閾值的低分丟掉"""
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]
    return keep


def _decode_mask_lowres(mask_proto: np.ndarray, box_letterbox, imgsz: int) -> np.ndarray:
    """在 proto 解析度(例如 160x160)上直接 threshold + 裁到 box 範圍, 不放大到原圖
    解析度——containment dedup 只需要尺度無關的「交集/較小面積」比例, 不用放大就能算
    (為什麼不放大、效能數字見 docs/影像辨識.md)。

    box_letterbox 是 letterbox 畫布(IMGSZ)座標系的 bbox(NMS 後、還沒轉回原圖座標的那份),
    跟 proto 同一個座標系只是解析度不同, 用比例縮放就能對應, 不用先還原成原圖座標。"""
    proto_h, proto_w = mask_proto.shape
    sx, sy = proto_w / imgsz, proto_h / imgsz
    x1, y1, x2, y2 = box_letterbox
    px1 = max(0, int(round(x1 * sx)))
    py1 = max(0, int(round(y1 * sy)))
    px2 = min(proto_w, int(round(x2 * sx)))
    py2 = min(proto_h, int(round(y2 * sy)))

    mask_bin = mask_proto > 0.5
    crop_mask = np.zeros_like(mask_bin)
    crop_mask[py1:py2, px1:px2] = True
    return mask_bin & crop_mask


def _containment_dedup(masks_np: np.ndarray, confs: np.ndarray) -> list[int]:
    """巢狀包含時留信心分數較高的那個(highscore 規則, 比較實驗見 docs/影像辨識.md)"""
    areas = masks_np.reshape(len(masks_np), -1).sum(axis=1)
    keep = np.ones(len(masks_np), dtype=bool)
    for i in range(len(masks_np)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(masks_np)):
            if not keep[j]:
                continue
            smaller_area = min(areas[i], areas[j])
            if smaller_area == 0:
                continue
            inter = np.logical_and(masks_np[i], masks_np[j]).sum()
            ratio = inter / smaller_area
            if ratio > CONTAINMENT_THRESH:
                loser = j if confs[i] >= confs[j] else i
                keep[loser] = False
                if loser == i:
                    break
    return np.where(keep)[0].tolist()


def find_objects(image: Image.Image, session: ort.InferenceSession) -> list[DetectedObject]:
    """對圖片跑 FastSAM 找候選物件, 回傳 list[DetectedObject]。
    已經做完信心過濾、NMS、containment 去重, 回傳的是最終候選, 不用再額外處理。
    image 必須是已經載入好的 PIL Image(RGB), 讀檔案是呼叫端的責任, 這裡只處理圖片資料。
    session 由呼叫端明確傳入(例如 FastAPI lifespan 建立好的那顆), 這裡不負責建立或快取。
    """
    W, H = image.size
    canvas, scale, pad_x, pad_y, _nw, _nh = _letterbox(image, IMGSZ)
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]

    output0, output1 = session.run(None, {"images": x})

    pred = output0[0].T  # (num_anchors, 37)
    proto = output1[0]  # (32, proto_h, proto_w), proto_h/w 通常是 IMGSZ 的 1/4, 不寫死

    boxes_cxcywh, conf, mask_coeffs = pred[:, :4], pred[:, 4], pred[:, 5:]
    keep_conf = conf > CONF_THRESH
    boxes_cxcywh, conf, mask_coeffs = boxes_cxcywh[keep_conf], conf[keep_conf], mask_coeffs[keep_conf]

    cx, cy, w_, h_ = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    boxes_xyxy = np.stack([cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2], axis=1)

    keep_idx = _nms(boxes_xyxy, conf, IOU_THRESH)
    boxes_xyxy, conf, mask_coeffs = boxes_xyxy[keep_idx], conf[keep_idx], mask_coeffs[keep_idx]
    print(f"FastSAM 候選物件 (信心過濾+NMS 後): {len(conf)}")

    proto_h, proto_w = proto.shape[1], proto.shape[2]
    proto_flat = proto.reshape(32, -1)
    masks_proto = _sigmoid(mask_coeffs @ proto_flat).reshape(-1, proto_h, proto_w)

    boxes_orig = boxes_xyxy.copy()
    boxes_orig[:, [0, 2]] = (boxes_orig[:, [0, 2]] - pad_x) / scale
    boxes_orig[:, [1, 3]] = (boxes_orig[:, [1, 3]] - pad_y) / scale
    boxes_orig[:, [0, 2]] = np.clip(boxes_orig[:, [0, 2]], 0, W)
    boxes_orig[:, [1, 3]] = np.clip(boxes_orig[:, [1, 3]], 0, H)

    masks_lowres = np.stack([
        _decode_mask_lowres(masks_proto[i], boxes_xyxy[i], IMGSZ)
        for i in range(len(conf))
    ])

    dedup_idx = _containment_dedup(masks_lowres, conf)
    print(f"containment 過濾後: {len(masks_lowres)} -> {len(dedup_idx)}")

    return [
        DetectedObject(
            bbox=tuple(float(v) for v in boxes_orig[i]),
            objectness_conf=float(conf[i]),
        )
        for i in dedup_idx
    ]
