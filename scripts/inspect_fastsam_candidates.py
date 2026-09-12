"""跑一次 FastSAM, 只畫候選框(不比對類別、不用 DINOv2), 用來驗證「挑面積最大的
候選框當樣本真正物件」這個啟發式規則站不站得住腳——尤其是背景木紋雜訊會不會
被誤判成面積最大的候選框, 蓋過真正的物件。

用法:
    python scripts/inspect_fastsam_candidates.py <圖片路徑或資料夾> [-o 輸出資料夾]

需要 FASTSAM_ONNX_PATH 環境變數(或用 --model-path 明確指定)。
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.vision.objectness import find_objects, load_session  # noqa: E402

CANDIDATE_COLOR = (150, 150, 150)
LARGEST_COLOR = (220, 30, 30)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def inspect(image_path: Path, session, output_dir: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    detections = find_objects(image, session)

    if not detections:
        print(f"{image_path.name}: 沒有偵測到任何候選框")
        return

    largest = max(detections, key=lambda d: _area(d.bbox))

    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 400))
    font_size = max(14, round(min(image.size) / 80))
    font = _load_font(font_size)

    for det in detections:
        is_largest = det is largest
        color = LARGEST_COLOR if is_largest else CANDIDATE_COLOR
        width = line_width * 2 if is_largest else line_width
        x1, y1, x2, y2 = det.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        label = f"{'LARGEST ' if is_largest else ''}conf={det.objectness_conf:.2f} area={int(_area(det.bbox))}"
        text_pos = (x1 + 2, max(0, y1 - font_size - 4))
        text_bg = draw.textbbox(text_pos, label, font=font)
        draw.rectangle(text_bg, fill=color)
        draw.text(text_pos, label, fill=(255, 255, 255), font=font)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_fastsam.jpg"
    image.save(output_path, format="JPEG", quality=92)
    print(f"{image_path.name}: {len(detections)} 個候選框, 最大面積框已標紅 -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="畫出 FastSAM 原始候選框(不比對類別), 驗證挑最大面積框的啟發式"
    )
    parser.add_argument("path", type=Path, help="單張圖片路徑, 或包含圖片的資料夾")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="輸出資料夾(預設: 輸入資料夾底下的 fastsam_check/)",
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

    output_dir = args.output_dir or (image_paths[0].parent / "fastsam_check")

    session = load_session(args.model_path)
    for image_path in image_paths:
        inspect(image_path, session, output_dir)


if __name__ == "__main__":
    main()
