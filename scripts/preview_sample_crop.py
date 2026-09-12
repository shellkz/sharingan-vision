"""離線預覽:對圖片跑一次 create_sample() 會用到的「FastSAM 抓最大候選框 + 裁切」
邏輯,但只輸出裁切後的圖片本身(就是實際會丟給 DINOv2 的那個畫面),不寫資料庫、
不建立任何 sample、不影響現有系統。用來在真的呼叫 API 之前先確認自動裁切結果
好不好、值不值得直接用。

用法:
    python scripts/preview_sample_crop.py <圖片路徑或資料夾> [-o 輸出資料夾]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from app.sample_ops import _detect_and_crop  # noqa: E402
from app.vision.objectness import load_session  # noqa: E402
from app.vision.recognize import CROP_PADDING_RATIO  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def preview(image_path: Path, session, output_dir: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    crop, bbox = _detect_and_crop(image, session, CROP_PADDING_RATIO)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_crop.jpg"
    crop.save(output_path, format="JPEG", quality=92)

    if bbox is None:
        print(f"{image_path.name}: 沒有偵測到候選框, fallback 用整張圖 -> {output_path}")
    else:
        print(f"{image_path.name}: bbox={bbox} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="預覽 create_sample() 的自動裁切結果, 不寫資料庫")
    parser.add_argument("path", type=Path, help="單張圖片路徑, 或包含圖片的資料夾")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="輸出資料夾(預設: 輸入資料夾底下的 crop_preview/)",
    )
    parser.add_argument("--model-path", type=str, default=None, help="FastSAM onnx 路徑, 不給就吃 FASTSAM_ONNX_PATH 環境變數")
    args = parser.parse_args()

    if args.path.is_dir():
        image_paths = sorted(p for p in args.path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        image_paths = [args.path]

    if not image_paths:
        print("找不到任何圖片")
        return

    output_dir = args.output_dir or (image_paths[0].parent / "crop_preview")

    session = load_session(args.model_path)
    for image_path in image_paths:
        preview(image_path, session, output_dir)


if __name__ == "__main__":
    main()
