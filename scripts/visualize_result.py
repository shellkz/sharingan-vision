"""吃一張原圖 + `POST /recognize` 回傳的 JSON(存成檔案的 `{"result": [...]}`),
畫出每個候選框的 bbox + entity_name/instance_name + score, 存成新圖方便肉眼核對。

純 PIL, 不依賴 app/ 底下任何模組、不用連服務——離線分析工具, 專門用來對照
test/ 底下存下來的辨識結果 JSON(例如 scene_1_result.json)。

用法:
    python scripts/visualize_result.py <圖片路徑> <result json 路徑> [-o 輸出路徑]
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

UNKNOWN_COLOR = (150, 150, 150)
PALETTE = [
    (220, 30, 30),
    (30, 100, 220),
    (30, 160, 60),
    (230, 150, 20),
    (160, 30, 220),
    (30, 200, 200),
]


def _color_for_entity(entity_name: str, assigned: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    """同一個 entity_name 一律用同一個顏色, 顏色照第一次出現的順序從 PALETTE 分配。
    unknown 固定灰色, 一眼就能跟「有比對到已知類別」的框分開。"""
    if entity_name == "unknown":
        return UNKNOWN_COLOR
    if entity_name not in assigned:
        assigned[entity_name] = PALETTE[len(assigned) % len(PALETTE)]
    return assigned[entity_name]


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def visualize(image_path: Path, result_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    results = payload["result"] if isinstance(payload, dict) else payload

    # 線寬/字級跟著圖片解析度縮放, 手機拍的原圖(3000-4000px)用固定 2px/預設字級會小到看不見
    line_width = max(2, round(min(image.size) / 400))
    font_size = max(14, round(min(image.size) / 80))
    font = _load_font(font_size)

    draw = ImageDraw.Draw(image)
    assigned_colors: dict[str, tuple[int, int, int]] = {}

    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        score = r["score"]
        # meets_threshold=false 一律當 unknown 畫(灰色), 不管底下猜的身份是誰
        # ——沒有這個欄位的舊格式 JSON(改版前存的結果)當作沒有門檻限制, 照原樣畫
        if r.get("meets_threshold", True):
            entity_name = r["entity_name"]
            instance_name = r.get("instance_name")
        else:
            entity_name = "unknown"
            instance_name = None
        color = _color_for_entity(entity_name, assigned_colors)

        label = f"{entity_name}/{instance_name} {score:.2f}" if instance_name else f"{entity_name} {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        text_pos = (x1 + 2, max(0, y1 - font_size - 4))
        text_bg = draw.textbbox(text_pos, label, font=font)
        draw.rectangle(text_bg, fill=color)
        draw.text(text_pos, label, fill=(255, 255, 255), font=font)

    image.save(output_path, format="JPEG", quality=92)
    print(f"寫入 {output_path}({len(results)} 個候選框)")


def main() -> None:
    parser = argparse.ArgumentParser(description="畫出 /recognize 結果的 bbox 標註圖, 方便肉眼核對")
    parser.add_argument("image", type=Path, help="原圖路徑")
    parser.add_argument("result_json", type=Path, help="/recognize 回傳的 JSON 檔案路徑")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="輸出圖片路徑(預設: <原圖檔名>_annotated.jpg, 跟原圖同資料夾)",
    )
    args = parser.parse_args()

    output = args.output or args.image.with_name(f"{args.image.stem}_annotated.jpg")
    visualize(args.image, args.result_json, output)


if __name__ == "__main__":
    main()
