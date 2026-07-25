"""
ประเมินประสิทธิภาพโมเดล — ใช้เขียนผลการทดลองในบทที่ 4

ทำ 2 อย่าง:
  1. วัดค่ามาตรฐาน (Precision, Recall, mAP50, mAP50-95) ทั้งภาพรวมและรายคลาส
  2. ไล่ทดสอบค่า confidence หลายระดับ เพื่อหาค่าที่เหมาะที่สุดสำหรับคลาส "วัชพืช"

ข้อ 2 สำคัญมากสำหรับงานนี้ เพราะการตรวจผิดสองแบบมีราคาไม่เท่ากัน:
  - พ่นทั้งที่ไม่ใช่วัชพืช (False Positive) = เปลืองยา และอาจโดนพืชหลัก
  - ไม่พ่นทั้งที่เป็นวัชพืช (False Negative) = วัชพืชเหลือรอด
งานนี้ให้น้ำหนักกับการลด False Positive มากกว่า จึงควรตั้ง conf ค่อนข้างสูง

วิธีใช้:
    python tools/evaluate.py
    python tools/evaluate.py --quick        # ข้ามการไล่ค่า conf
    python tools/evaluate.py --split test
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.stats import format_table, write_summary_csv


def extract_metrics(results, class_names: list[str]) -> dict:
    """ดึงตัวเลขที่ต้องใช้ออกจากผลลัพธ์ของ ultralytics (โครงสร้างต่างกันตามเวอร์ชัน)"""
    box = getattr(results, "box", None)
    if box is None:
        return {"overall": {}, "per_class": []}

    overall = {
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "mAP50": float(getattr(box, "map50", 0.0)),
        "mAP50-95": float(getattr(box, "map", 0.0)),
    }
    precision = overall["precision"]
    recall = overall["recall"]
    overall["f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_class = []
    class_indices = list(getattr(box, "ap_class_index", []) or [])
    for position, class_index in enumerate(class_indices):
        class_index = int(class_index)
        name = class_names[class_index] if class_index < len(class_names) else f"id_{class_index}"

        def value_at(attribute: str) -> float:
            values = getattr(box, attribute, None)
            try:
                return float(values[position])
            except (TypeError, IndexError, ValueError):
                return 0.0

        p = value_at("p")
        r = value_at("r")
        per_class.append(
            {
                "class": name,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
                "mAP50": round(value_at("ap50"), 4),
                "mAP50-95": round(value_at("ap"), 4),
            }
        )
    return {"overall": {k: round(v, 4) for k, v in overall.items()}, "per_class": per_class}


def interpret(map50: float) -> str:
    """แปลตัวเลข mAP50 เป็นระดับคุณภาพแบบภาษาคน (ใช้อ้างอิงสมมติฐานงานวิจัย)"""
    if map50 >= 0.90:
        return "ดีมาก — ใช้งานจริงได้"
    if map50 >= 0.80:
        return "ดี — ใช้งานได้ในสภาพแวดล้อมที่ควบคุมได้"
    if map50 >= 0.70:
        return "พอใช้ — ควรเก็บภาพเพิ่มหรือเทรนต่อ"
    if map50 >= 0.50:
        return "ต่ำ — ชุดข้อมูลน่าจะยังน้อยหรือ label ไม่สม่ำเสมอ"
    return "ต่ำมาก — ตรวจสอบว่า label ถูกต้องและคลาสตรงกันหรือไม่"


def main() -> int:
    parser = argparse.ArgumentParser(description="ประเมินประสิทธิภาพโมเดลตรวจจับวัชพืช")
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights", default=None, help="พาธโมเดล (ค่าเริ่มต้น models/best.pt)")
    parser.add_argument("--data", default=None, help="พาธ data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="ข้ามขั้นตอนไล่ค่า confidence")
    parser.add_argument(
        "--conf-range",
        default="0.25,0.75,0.05",
        help="ช่วงค่า conf ที่จะไล่ทดสอบ รูปแบบ เริ่ม,จบ,ก้าว",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights = Path(args.weights).resolve() if args.weights else PROJECT_ROOT / "models" / "best.pt"
    data = Path(args.data).resolve() if args.data else cfg.resolve_path(cfg.train.data)
    imgsz = args.imgsz if args.imgsz is not None else cfg.model.imgsz

    if not weights.is_file():
        print(f"[ผิดพลาด] ไม่พบโมเดล {weights} — เทรนก่อนด้วย: python tools/train.py")
        return 1
    if not data.is_file():
        print(f"[ผิดพลาด] ไม่พบ {data}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ผิดพลาด] ไม่พบไลบรารี ultralytics — ติดตั้งด้วย:  pip install -r requirements-pc.txt")
        return 1

    model = YOLO(str(weights))
    model_names = getattr(model, "names", {}) or {}
    class_names = [model_names[k] for k in sorted(model_names)] if model_names else cfg.classes.names

    print("=" * 72)
    print(" ประเมินประสิทธิภาพโมเดล")
    print("=" * 72)
    print(f" โมเดล      : {weights.name}")
    print(f" ชุดข้อมูล   : {data}  (split = {args.split})")
    print(f" ขนาดภาพ    : {imgsz}x{imgsz}")
    print(f" คลาส       : {class_names}")
    print("-" * 72)

    results = model.val(
        data=str(data), imgsz=imgsz, split=args.split, conf=0.001, iou=cfg.model.iou, verbose=False
    )
    metrics = extract_metrics(results, class_names)

    print("\n[ภาพรวมทุกคลาส]")
    overall = metrics["overall"]
    for key, label in (
        ("precision", "Precision (ความแม่นยำของที่ทายว่าใช่)"),
        ("recall", "Recall (สัดส่วนของจริงที่จับได้)"),
        ("f1", "F1-score (ค่าเฉลี่ยถ่วงของสองตัวบน)"),
        ("mAP50", "mAP@0.5"),
        ("mAP50-95", "mAP@0.5:0.95"),
    ):
        print(f"  {label:<42} {overall.get(key, 0.0):.4f}")
    print(f"\n  ระดับคุณภาพจาก mAP@0.5 : {interpret(overall.get('mAP50', 0.0))}")

    print("\n[แยกรายคลาส]")
    print(format_table(metrics["per_class"], ["class", "precision", "recall", "f1", "mAP50", "mAP50-95"]))

    output_dir = PROJECT_ROOT / "logs"
    output_dir.mkdir(exist_ok=True)
    write_summary_csv(output_dir / "eval_per_class.csv", metrics["per_class"])

    sweep_rows: list[dict] = []
    if not args.quick:
        try:
            start, stop, step = (float(v) for v in args.conf_range.split(","))
        except ValueError:
            print(f"[เตือน] รูปแบบ --conf-range ไม่ถูกต้อง: {args.conf_range} -> ข้ามขั้นตอนนี้")
            start = stop = step = 0.0

        if step > 0:
            print("\n" + "-" * 72)
            print(" ไล่ทดสอบค่า confidence เพื่อหาจุดที่เหมาะที่สุด (ใช้เวลาสักครู่)")
            print("-" * 72)

            conf_value = start
            while conf_value <= stop + 1e-9:
                sweep_result = model.val(
                    data=str(data),
                    imgsz=imgsz,
                    split=args.split,
                    conf=round(conf_value, 4),
                    iou=cfg.model.iou,
                    verbose=False,
                )
                sweep_metrics = extract_metrics(sweep_result, class_names)
                row = {
                    "conf": round(conf_value, 2),
                    "precision": sweep_metrics["overall"].get("precision", 0.0),
                    "recall": sweep_metrics["overall"].get("recall", 0.0),
                    "f1": round(sweep_metrics["overall"].get("f1", 0.0), 4),
                }
                for entry in sweep_metrics["per_class"]:
                    if entry["class"] in cfg.classes.weed:
                        row["weed_precision"] = entry["precision"]
                        row["weed_recall"] = entry["recall"]
                        row["weed_f1"] = entry["f1"]
                        break
                sweep_rows.append(row)
                print(
                    f"  conf {row['conf']:.2f} -> P {row['precision']:.3f}  "
                    f"R {row['recall']:.3f}  F1 {row['f1']:.3f}"
                )
                conf_value += step

            if sweep_rows:
                columns = ["conf", "precision", "recall", "f1"]
                if "weed_f1" in sweep_rows[0]:
                    columns += ["weed_precision", "weed_recall", "weed_f1"]
                print("\n" + format_table(sweep_rows, columns))
                write_summary_csv(output_dir / "eval_conf_sweep.csv", sweep_rows, columns)

                score_key = "weed_f1" if "weed_f1" in sweep_rows[0] else "f1"
                best = max(sweep_rows, key=lambda r: r.get(score_key, 0.0))
                print(f"\n  ค่า confidence ที่ให้ {score_key} สูงสุด = {best['conf']:.2f}")
                print("  นำไปตั้งใน config.yaml:")
                if score_key == "weed_f1":
                    weed_name = next(
                        (n for n in cfg.classes.weed if n in class_names), cfg.classes.weed[0]
                    )
                    print(f"    classes.per_class_conf.{weed_name}: {best['conf']:.2f}")
                else:
                    print(f"    model.conf: {best['conf']:.2f}")

    report = {
        "weights": str(weights),
        "data": str(data),
        "split": args.split,
        "imgsz": imgsz,
        "overall": metrics["overall"],
        "per_class": metrics["per_class"],
        "conf_sweep": sweep_rows,
    }
    report_path = output_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f" บันทึกผลไว้ที่ {output_dir}")
    print("   - eval_report.json     สรุปทั้งหมด")
    print("   - eval_per_class.csv   ตารางรายคลาส (เปิดด้วย Excel ได้)")
    if sweep_rows:
        print("   - eval_conf_sweep.csv  ตารางไล่ค่า confidence")
    print(f" กราฟและ Confusion Matrix อยู่ที่ {getattr(results, 'save_dir', '(ดูในโฟลเดอร์ runs/)')}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
