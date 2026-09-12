"""開發用的一次性工具: 把 FastSAM-s 匯出成 ONNX。

需要 torch/ultralytics, 只在開發環境跑, 不要打包進 Docker/production 環境。
匯出完的 FastSAM-s.onnx 才是實際會被 objectness.py 使用的產物。

輸入解析度預設 1024, 可用 FASTSAM_EXPORT_IMGSZ 環境變數覆蓋(例如匯出 640 版本時)。
匯出格式跟 PyTorch 版本的驗證結果、解析度選擇的取捨見 docs/影像辨識.md:
  輸入  images  [1, 3, IMGSZ, IMGSZ]      float32
  輸出0 output0 [1, 37, num_anchors]      float32  (4 bbox + 1 conf + 32 mask 係數)
  輸出1 output1 [1, 32, IMGSZ/4, IMGSZ/4] float32  (32 個 mask prototype)
"""
import os

from ultralytics import FastSAM

imgsz = int(os.environ.get("FASTSAM_EXPORT_IMGSZ", "1024"))

model = FastSAM("FastSAM-s.pt")
path = model.export(format="onnx", imgsz=imgsz)
print(f"匯出完成(imgsz={imgsz}): {path}")
