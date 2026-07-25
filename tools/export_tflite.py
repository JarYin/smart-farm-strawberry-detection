"""
ขั้นตอนที่ 3 (ต่อ) — แปลงโมเดล .pt เป็น .tflite สำหรับรันบน Raspberry Pi

ทำไมต้องแปลง: ไฟล์ .pt ต้องใช้ PyTorch ซึ่งกินพื้นที่ราว 2 GB และช้ามากบน Pi
ส่วน .tflite ใช้ tflite-runtime ขนาดไม่กี่ MB และเร็วกว่าอย่างชัดเจน

รูปแบบที่แปลงได้:
    float32 : แม่นยำเท่าต้นฉบับ ไฟล์ใหญ่กว่า   <- แนะนำให้เริ่มจากตัวนี้
    int8    : เล็กลง ~4 เท่า เร็วขึ้น แต่ความแม่นยำลดลงเล็กน้อย
              ต้องมีภาพตัวอย่างจากชุดข้อมูลจริงไว้ใช้ปรับเทียบ (calibration)

วิธีใช้:
    python tools/export_tflite.py
    python tools/export_tflite.py --int8
"""

import _bootstrap  # noqa: F401

import argparse
import shutil
from pathlib import Path

from src.config import PROJECT_ROOT, load_config


def find_exported_file(source_weights: Path, want_int8: bool) -> Path | None:
    """ค้นหาไฟล์ .tflite ที่ ultralytics สร้างไว้

    ultralytics จะสร้างโฟลเดอร์ <ชื่อโมเดล>_saved_model/ แล้ววางไฟล์ .tflite ไว้ข้างใน
    ชื่อไฟล์ต่างกันไปตามเวอร์ชัน จึงต้องค้นแบบยืดหยุ่น
    """
    saved_model_dir = source_weights.parent / f"{source_weights.stem}_saved_model"
    search_dirs = [saved_model_dir, source_weights.parent]

    keyword = "int8" if want_int8 else "float32"
    fallback: Path | None = None
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.glob("*.tflite")):
            if keyword in candidate.name:
                return candidate
            fallback = fallback or candidate
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="แปลงโมเดล YOLOv8 เป็น TensorFlow Lite")
    parser.add_argument("--config", default=None, help="พาธ config.yaml")
    parser.add_argument("--weights", default=None, help="พาธไฟล์ .pt (ค่าเริ่มต้น models/best.pt)")
    parser.add_argument("--imgsz", type=int, default=None, help="ขนาดภาพเข้าโมเดล (ค่าเริ่มต้น 320)")
    parser.add_argument("--int8", action="store_true", help="แปลงเป็นแบบ int8 (เล็กและเร็วกว่า)")
    parser.add_argument("--data", default=None, help="data.yaml สำหรับปรับเทียบ int8")
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights = Path(args.weights).resolve() if args.weights else PROJECT_ROOT / "models" / "best.pt"
    imgsz = args.imgsz if args.imgsz is not None else cfg.model.imgsz

    if not weights.is_file():
        print(f"[ผิดพลาด] ไม่พบไฟล์โมเดล {weights}")
        print("  เทรนก่อนด้วย:  python tools/train.py")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ผิดพลาด] ไม่พบไลบรารี ultralytics — ติดตั้งด้วย:  pip install -r requirements-pc.txt")
        return 1

    print("=" * 72)
    print(" แปลงโมเดลเป็น TensorFlow Lite")
    print("=" * 72)
    print(f" โมเดลต้นทาง : {weights}")
    print(f" ขนาดภาพ     : {imgsz}x{imgsz}")
    print(f" รูปแบบ      : {'int8 (quantized)' if args.int8 else 'float32'}")
    print("-" * 72)
    print(" หมายเหตุ: ครั้งแรกจะช้าเพราะต้องดาวน์โหลดเครื่องมือแปลง (onnx2tf ฯลฯ)")
    print()

    model = YOLO(str(weights))
    export_kwargs = {"format": "tflite", "imgsz": imgsz, "int8": args.int8}
    if args.int8:
        data = Path(args.data).resolve() if args.data else cfg.resolve_path(cfg.train.data)
        if not data.is_file():
            print(f"[ผิดพลาด] การแปลงแบบ int8 ต้องใช้ภาพปรับเทียบจาก {data} แต่ไม่พบไฟล์")
            return 1
        export_kwargs["data"] = str(data)

    try:
        model.export(**export_kwargs)
    except Exception as exc:
        print(f"\n[ผิดพลาด] แปลงไม่สำเร็จ: {exc}\n")
        print("สาเหตุที่พบบ่อย:")
        print("  - ยังไม่ได้ติดตั้งเครื่องมือแปลง ให้ลอง:")
        print("      pip install tensorflow onnx onnx2tf onnxslim onnx_graphsurgeon sng4onnx")
        print("  - Python 3.13 ยังมีปัญหากับสายการแปลงนี้ในบางเวอร์ชัน")
        print("    แนะนำให้สร้าง environment ด้วย Python 3.11 สำหรับขั้นตอนแปลงโดยเฉพาะ")
        print("  - ทางเลือกที่ง่ายที่สุด: แปลงบน Google Colab แล้วดาวน์โหลดไฟล์ .tflite มาใช้")
        return 1

    exported = find_exported_file(weights, args.int8)
    if exported is None:
        print("[เตือน] แปลงเสร็จแล้วแต่หาไฟล์ .tflite ไม่เจอ ลองมองหาในโฟลเดอร์เดียวกับโมเดลต้นทาง")
        return 1

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    suffix = "int8" if args.int8 else "float32"
    destination = models_dir / f"best_{suffix}.tflite"
    shutil.copy2(exported, destination)

    size_mb = destination.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 72)
    print(" แปลงเสร็จแล้ว")
    print("=" * 72)
    print(f" ไฟล์ผลลัพธ์ : {destination}  ({size_mb:.1f} MB)")
    print("\n ขั้นตอนถัดไป:")
    print(f"   ทดสอบบน PC :  python -m src.main --weights models/best_{suffix}.tflite")
    print(f"   คัดลอกไป Pi :  scp {destination.name} pi@<ไอพีของ Pi>:~/smartfarm/models/")
    print("\n อย่าลืมแก้ config.yaml บน Pi เป็น:")
    print(f"   model.weights: models/best_{suffix}.tflite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
