"""開發用的一次性工具: 把 DINOv2 (dinov2_vits14) 匯出成 ONNX, 並驗證
ONNX Runtime 推論結果跟 PyTorch 原本的結果是否一致 (cosine similarity 應接近 1)。

需要 torch/torchvision, 只在開發環境跑, 不要打包進 Docker/production 環境。
匯出完的 dinov2_vits14.onnx 才是實際會被 embedding.py 使用的產物。

輸入  input      [1, 3, 224, 224]  float32  (前處理後的裁切圖: resize 短邊 256 + center crop 224 + normalize)
輸出  embedding  [1, 384]          float32  (ViT 的 [CLS] token, 直接就是這張圖的外觀特徵向量, 不用額外解碼)
"""

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

ONNX_PATH = "dinov2_vits14.onnx"
TEST_IMAGE = "images/led/3.png"

_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


class DinoWrapper(torch.nn.Module):
    """強制只留一個乾淨的 x 輸入, 避免原始 forward(*args, **kwargs) 簽名
    在 tracing 時把 masks=None 也捕捉成一個多餘的 ONNX 圖輸入。"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


print("載入 DINOv2 (dinov2_vits14) ...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model.eval()

wrapped = DinoWrapper(model)
dummy_input = torch.randn(1, 3, 224, 224)

print(f"匯出 ONNX -> {ONNX_PATH} ...")
torch.onnx.export(
    wrapped,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["embedding"],
    opset_version=17,
    dynamo=False,
    dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
)
print("匯出完成")

# ---- 驗證 1: 同一張圖, PyTorch vs ONNX Runtime(batch=1) 的 embedding 應該幾乎一樣 ----
img = to_rgb(Image.open(TEST_IMAGE))
x = _transform(img).unsqueeze(0)

with torch.no_grad():
    torch_out = model(x).numpy()

session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
onnx_out = session.run(["embedding"], {"input": x.numpy()})[0]

cos_sim = float(
    np.dot(torch_out[0], onnx_out[0])
    / (np.linalg.norm(torch_out[0]) * np.linalg.norm(onnx_out[0]))
)
max_abs_diff = float(np.abs(torch_out - onnx_out).max())

print(f"\n測試圖: {TEST_IMAGE}")
print(f"cosine similarity (應接近 1.0): {cos_sim:.6f}")
print(f"最大絕對誤差 (應接近 0):        {max_abs_diff:.6f}")

# ---- 驗證 2: dynamic_axes 之後, batch>1 一次跑 vs 逐張跑, 每張的結果應該完全對得上 ----
BATCH_TEST_IMAGES = [
    "images/led/1.png",
    "images/button_switch/1.jpg",
    "images/rotary_potentiometer/1.jpg",
    "images/button_cap/1.png",
]

batch_x = torch.cat(
    [_transform(to_rgb(Image.open(p))).unsqueeze(0) for p in BATCH_TEST_IMAGES], dim=0
)
batch_out = session.run(["embedding"], {"input": batch_x.numpy()})[0]

print(f"\nbatch 推論驗證(batch size={len(BATCH_TEST_IMAGES)}):")
for i, p in enumerate(BATCH_TEST_IMAGES):
    single_out = session.run(["embedding"], {"input": batch_x[i : i + 1].numpy()})[0]
    cos_sim_batch = float(
        np.dot(batch_out[i], single_out[0])
        / (np.linalg.norm(batch_out[i]) * np.linalg.norm(single_out[0]))
    )
    max_abs_diff_batch = float(np.abs(batch_out[i] - single_out[0]).max())
    print(
        f"  [{i}] {p}: cosine similarity (應接近 1.0) = {cos_sim_batch:.6f}, "
        f"最大絕對誤差 (應接近 0) = {max_abs_diff_batch:.6f}"
    )
