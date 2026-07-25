"""
ขั้นตอนที่ 1 — ตรวจสอบชุดข้อมูลก่อนเทรน

เหตุผลที่ต้องมีสคริปต์นี้: ปัญหาโมเดลไม่แม่นส่วนใหญ่ไม่ได้เกิดจากโค้ดเทรน
แต่เกิดจากชุดข้อมูล เช่น จำนวนภาพวัชพืชน้อยกว่าสตรอว์เบอร์รี 10 เท่า
หรือลืมใส่ไฟล์ label ตรวจตั้งแต่ตอนนี้ประหยัดเวลาเทรนไปหลายชั่วโมง

วิธีใช้:
    python tools/dataset_check.py
    python tools/dataset_check.py --data datasets/data.yaml
"""

import _bootstrap  # noqa: F401  (ต้องมาก่อนเพื่อให้ import src ได้)

import argparse
from collections import Counter
from pathlib import Path

from src.config import load_config
from src.stats import format_table

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(path: Path) -> dict:
    import yaml

    if not path.is_file():
        raise SystemExit(
            f"ไม่พบไฟล์ {path}\n"
            "  ดาวน์โหลดชุดข้อมูลจาก Roboflow แบบ 'YOLOv8' แล้วแตกไฟล์ไว้ในโฟลเดอร์ datasets/"
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_split_dir(data_yaml_path: Path, data: dict, split: str) -> Path | None:
    """หาโฟลเดอร์ภาพของแต่ละ split โดยรองรับรูปแบบ path ที่ Roboflow ใช้"""
    value = data.get(split)
    if not value:
        return None

    base = Path(data.get("path", ".")) if data.get("path") else Path(".")
    candidates = [
        Path(value),
        data_yaml_path.parent / value,
        data_yaml_path.parent / base / value,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate
        # data.yaml บางไฟล์ชี้ไปที่ ../train/images ซึ่ง normalize แล้วอาจหลุด
        if candidate.name != "images" and (candidate / "images").is_dir():
            return (candidate / "images").resolve()
    return None


def labels_dir_for(images_dir: Path) -> Path:
    """โครงสร้างมาตรฐานของ YOLO: .../images/xxx.jpg คู่กับ .../labels/xxx.txt"""
    parts = list(images_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels"


def check_split(images_dir: Path, class_names: list[str]) -> dict:
    labels_dir = labels_dir_for(images_dir)
    image_files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)

    class_counts: Counter[int] = Counter()
    missing_labels: list[str] = []
    empty_labels: list[str] = []
    malformed: list[str] = []
    unknown_class_ids: set[int] = set()
    total_boxes = 0
    tiny_boxes = 0

    for image_path in image_files:
        relative = image_path.relative_to(images_dir)
        label_path = (labels_dir / relative).with_suffix(".txt")
        if not label_path.is_file():
            missing_labels.append(str(relative))
            continue

        lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            # ภาพที่ไม่มีวัตถุเลย (background image) มีประโยชน์ในการลด False Positive
            empty_labels.append(str(relative))
            continue

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                malformed.append(f"{relative}: {line}")
                continue
            try:
                class_id = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                malformed.append(f"{relative}: {line}")
                continue

            if class_id < 0 or class_id >= len(class_names):
                unknown_class_ids.add(class_id)
            if not all(0.0 <= v <= 1.0 for v in (x, y)) or not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                malformed.append(f"{relative}: พิกัดอยู่นอกช่วง 0-1 -> {line}")
                continue

            # กล่องที่เล็กกว่า 2% ของภาพ จะเล็กมากเมื่อย่อเหลือ 320x320 (ราว 6 พิกเซล)
            if w < 0.02 or h < 0.02:
                tiny_boxes += 1

            class_counts[class_id] += 1
            total_boxes += 1

    return {
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "num_images": len(image_files),
        "num_boxes": total_boxes,
        "class_counts": class_counts,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "malformed": malformed,
        "unknown_class_ids": unknown_class_ids,
        "tiny_boxes": tiny_boxes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ตรวจสอบความถูกต้องของชุดข้อมูล YOLO ก่อนเทรน")
    parser.add_argument("--config", default=None, help="พาธ config.yaml")
    parser.add_argument("--data", default=None, help="พาธ data.yaml (แทนค่าใน config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_yaml_path = Path(args.data).resolve() if args.data else cfg.resolve_path(cfg.train.data)
    data = load_data_yaml(data_yaml_path)

    names = data.get("names")
    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names)]
    elif isinstance(names, list):
        class_names = list(names)
    else:
        raise SystemExit(f"ไฟล์ {data_yaml_path} ไม่มีคีย์ 'names'")

    print("=" * 72)
    print(" ตรวจสอบชุดข้อมูล")
    print("=" * 72)
    print(f" ไฟล์ data.yaml : {data_yaml_path}")
    print(f" คลาสในชุดข้อมูล : {class_names}")
    print(f" คลาสใน config   : {cfg.classes.names}")

    problems: list[str] = []
    if class_names != cfg.classes.names:
        problems.append(
            "classes.names ใน config.yaml ไม่ตรงกับ names ใน data.yaml\n"
            f"    แก้ config.yaml เป็น:  names: {class_names}"
        )

    rows = []
    grand_total = Counter()
    for split in ("train", "val", "test"):
        images_dir = resolve_split_dir(data_yaml_path, data, split)
        if images_dir is None:
            if split != "test":
                problems.append(f"หาโฟลเดอร์ของ split '{split}' ไม่เจอ")
            continue

        result = check_split(images_dir, class_names)
        grand_total.update(result["class_counts"])

        row = {"split": split, "images": result["num_images"], "boxes": result["num_boxes"]}
        for index, name in enumerate(class_names):
            row[name] = result["class_counts"].get(index, 0)
        rows.append(row)

        if result["missing_labels"]:
            problems.append(
                f"[{split}] ไม่มีไฟล์ label {len(result['missing_labels'])} ภาพ "
                f"(เช่น {result['missing_labels'][0]})"
            )
        if result["malformed"]:
            problems.append(
                f"[{split}] ไฟล์ label ผิดรูปแบบ {len(result['malformed'])} รายการ "
                f"(เช่น {result['malformed'][0]})"
            )
        if result["unknown_class_ids"]:
            problems.append(
                f"[{split}] พบ class id ที่ไม่มีในรายการคลาส: {sorted(result['unknown_class_ids'])}"
            )
        if result["empty_labels"]:
            print(
                f"\n [{split}] มีภาพพื้นหลัง (ไม่มีวัตถุ) {len(result['empty_labels'])} ภาพ "
                "— ปกติดี ช่วยลดการตรวจจับผิดพลาด"
            )
        if result["tiny_boxes"]:
            print(
                f" [{split}] มีกล่องขนาดเล็กมาก {result['tiny_boxes']} กล่อง "
                "— เมื่อย่อเหลือ 320x320 อาจเล็กจนโมเดลเรียนรู้ไม่ได้"
            )

    print("\n" + format_table(rows, ["split", "images", "boxes", *class_names]))

    # ตรวจความสมดุลของจำนวนตัวอย่างแต่ละคลาส
    if grand_total:
        most = max(grand_total.values())
        least = min(grand_total.values())
        if least == 0:
            zero = [class_names[i] for i in range(len(class_names)) if grand_total.get(i, 0) == 0]
            problems.append(f"คลาสเหล่านี้ไม่มีตัวอย่างเลย: {zero}")
        elif most / least > 5:
            worst = min(grand_total, key=lambda k: grand_total[k])
            best = max(grand_total, key=lambda k: grand_total[k])
            problems.append(
                f"จำนวนตัวอย่างไม่สมดุล: '{class_names[best]}' มี {most} กล่อง "
                f"แต่ '{class_names[worst]}' มีแค่ {least} กล่อง (ต่างกัน {most/least:.1f} เท่า)\n"
                "    ควรเก็บภาพคลาสที่น้อยเพิ่ม ไม่งั้นโมเดลจะเอนเอียงไปทางคลาสที่เยอะ"
            )

    print()
    if problems:
        print("พบข้อควรแก้ไข:")
        for index, problem in enumerate(problems, 1):
            print(f"  {index}. {problem}")
        return 1

    print("ชุดข้อมูลผ่านการตรวจสอบ — พร้อมเทรนแล้ว")
    print("  ขั้นตอนถัดไป:  python tools/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
