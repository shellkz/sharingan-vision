"""開發用的一次性工具: 把 dinov2_vits14.onnx 動態量化成 INT8, 並驗證量化前後
DINOv2 embedding 的 cosine similarity / 分類結果是否還可信賴。

需要 onnxruntime(quantization 工具內建, 不需要額外套件), 只在開發環境跑,
不要打包進 Docker/production 環境。量化完的 dinov2_vits14_int8.onnx 要這支腳本
驗證過關才能換成正式檔名(見 NOTES.md 的驗證慣例)。
"""
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

FP32_PATH = "../models/dinov2_vits14.onnx"
PREPROCESSED_PATH = "../models/dinov2_vits14_preprocessed.onnx"
INT8_PATH = "../models/dinov2_vits14_int8.onnx"

# 涵蓋全部 4 個已知類別, led 額外挑一張跟 export_dino.py 用過的同一張圖(3.png)交叉核對
TEST_IMAGES = [
    "../app/vision/images/led/1.png",
    "../app/vision/images/led/3.png",
    "../app/vision/images/button_switch/1.jpg",
    "../app/vision/images/rotary_potentiometer/1.jpg",
    "../app/vision/images/button_cap/1.png",
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def resize_shorter_side(img: Image.Image, size: int = 256) -> Image.Image:
    w, h = img.size
    if w <= h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_h, new_w = size, int(size * w / h)
    return img.resize((new_w, new_h), Image.BICUBIC)


def center_crop(img: Image.Image, size: int = 224) -> Image.Image:
    w, h = img.size
    left = round((w - size) / 2)
    top = round((h - size) / 2)
    return img.crop((left, top, left + size, top + size))


def preprocess(img: Image.Image) -> np.ndarray:
    img = center_crop(resize_shorter_side(to_rgb(img), 256), 224)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return arr[None].astype(np.float32)


print(f"量化前 pre-processing(shape inference + 模型優化) {FP32_PATH} -> {PREPROCESSED_PATH} ...")
quant_pre_process(FP32_PATH, PREPROCESSED_PATH)
print("pre-processing 完成")

print(f"動態量化 {PREPROCESSED_PATH} -> {INT8_PATH} ...")
quantize_dynamic(
    model_input=PREPROCESSED_PATH,
    model_output=INT8_PATH,
    weight_type=QuantType.QInt8,
)
print("量化完成")

# ---- 驗證: 用 CPUExecutionProvider 當基準(QOperator 格式的量化節點保證支援),
# 跟正式環境用 OpenVINOExecutionProvider 執行時的節點分派可能不同, 那是另一件事,
# 這裡只驗證「量化後數值還可不可信」。----
fp32_session = ort.InferenceSession(FP32_PATH, providers=["CPUExecutionProvider"])
int8_session = ort.InferenceSession(INT8_PATH, providers=["CPUExecutionProvider"])

print(f"\n{'測試圖':<45} {'cosine similarity':>18} {'最大絕對誤差':>14}")
for path in TEST_IMAGES:
    img = Image.open(path)
    x = preprocess(img)

    fp32_out = fp32_session.run(["embedding"], {"input": x})[0][0]
    int8_out = int8_session.run(["embedding"], {"input": x})[0][0]

    cos_sim = float(
        np.dot(fp32_out, int8_out)
        / (np.linalg.norm(fp32_out) * np.linalg.norm(int8_out))
    )
    max_abs_diff = float(np.abs(fp32_out - int8_out).max())
    print(f"{path:<45} {cos_sim:>18.6f} {max_abs_diff:>14.6f}")
