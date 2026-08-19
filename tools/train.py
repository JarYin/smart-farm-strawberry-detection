"""
ขั้นตอนที่ 3 — เทรนโมเดล YOLOv8 Nano ที่ขนาดภาพ 320x320

ต้องเทรนที่ 320 ตั้งแต่แรก ไม่ใช่เทรนที่ 640 แล้วค่อยลดตอนใช้งาน
เพราะโมเดลจะเรียนรู้รายละเอียดในระดับความละเอียดที่ใช้ตอนเทรน
ถ้าเทรน 640 แล้วรัน 320 ความแม่นยำจะตกอย่างเห็นได้ชัด

วิธีใช้:
    python tools/train.py                       # ใช้ค่าจาก config.yaml
    python tools/train.py --epochs 200
    python tools/train.py --device cpu          # ไม่มีการ์ดจอ (ช้ามาก แนะนำใช้ Colab แทน)
"""

import _bootstrap  # noqa: F401

import argparse
import shutil
from pathlib import Path

from src.config import PROJECT_ROOT, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="เทรนโมเดล YOLOv8 Nano สำหรับแยกพืชเป้าหมายกับวัชพืช")
    parser.add_argument("--config", default=None, help="พาธ config.yaml")
    parser.add_argument("--data", default=None, help="พาธ data.yaml")
    parser.add_argument("--model", default=None, help="โมเดลตั้งต้น (ค่าเริ่มต้น yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None, help='"" = อัตโนมัติ, "0" = GPU, "cpu" = ซีพียู')
    parser.add_argument("--name", default=None, help="ชื่อโฟลเดอร์ผลลัพธ์")
    parser.add_argument("--resume", action="store_true", help="เทรนต่อจากรอบที่ค้างไว้")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg.train

    data = Path(args.data).resolve() if args.data else cfg.resolve_path(tcfg.data)
    if not data.is_file():
        print(f"[ผิดพลาด] ไม่พบไฟล์ {data}")
        print("  ดาวน์โหลดชุดข้อมูลรูปแบบ YOLOv8 จาก Roboflow แล้วแตกไฟล์ไว้ที่ datasets/")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ผิดพลาด] ไม่พบไลบรารี ultralytics — ติดตั้งด้วย:  pip install -r requirements-pc.txt")
        return 1

    base_model = args.model or tcfg.base_model
    epochs = args.epochs if args.epochs is not None else tcfg.epochs
    batch = args.batch if args.batch is not None else tcfg.batch
    imgsz = args.imgsz if args.imgsz is not None else tcfg.imgsz
    device = args.device if args.device is not None else tcfg.device
    run_name = args.name or tcfg.name

    print("=" * 72)
    print(" เริ่มเทรนโมเดล")
    print("=" * 72)
    print(f" ชุดข้อมูล      : {data}")
    print(f" โมเดลตั้งต้น    : {base_model}")
    print(f" ขนาดภาพ        : {imgsz}x{imgsz}")
    print(f" จำนวนรอบ       : {epochs}  (หยุดก่อนถ้าไม่ดีขึ้นภายใน {tcfg.patience} รอบ)")
    print(f" batch          : {batch}")
    print(f" อุปกรณ์         : {device or 'อัตโนมัติ'}")
    print("-" * 72)

    model = YOLO(base_model)
    results = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=tcfg.patience,
        device=device if device else None,
        project=str(cfg.resolve_path(tcfg.project)),
        name=run_name,
        exist_ok=True,
        resume=args.resume,
        # การเพิ่มความหลากหลายของข้อมูล (Data Augmentation)
        # โรงเรือนมีแสงเปลี่ยนตลอดวัน จึงเพิ่มการสุ่มปรับความสว่าง/สีให้มากขึ้น
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=10.0,      # กล้องติดบนรถ อาจเอียงเล็กน้อยเวลาวิ่งผ่านร่องขรุขระ
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,        # ไม่พลิกบน-ล่าง เพราะต้นพืชขึ้นจากพื้นเสมอ
        mosaic=1.0,
        close_mosaic=10,   # ปิด mosaic ใน 10 รอบสุดท้าย ให้โมเดลปรับตัวกับภาพจริง
        # เพิ่ม 2026-08-07: ทดสอบ best.pt รุ่นก่อนพบว่า false positive ง่ายกับฉากที่ไม่ใช่
        # สตรอว์เบอร์รี (มือ/คน/พืชอื่น) สูงถึง 90%+ เพราะโมเดลน่าจะจับ "พื้นผิวซับซ้อนทั่วไป"
        # เป็นสัญญาณแทนลักษณะเฉพาะของสตรอว์เบอร์รี — mixup/copy_paste ผสม/แปะวัตถุจากภาพอื่น
        # เข้าไปในภาพเทรน บังคับให้โมเดลต้องเรียนรู้ลักษณะเฉพาะจริงๆ แทนพื้นผิวทั่วไป
        # ค่าที่ตั้งเป็นค่ากลางๆ ที่ ultralytics แนะนำ ยังไม่เคยรันจริง ต้องดูผลตอนเทรนรอบหน้า
        mixup=0.15,
        copy_paste=0.1,
    )

    save_dir = Path(getattr(results, "save_dir", cfg.resolve_path(tcfg.project) / run_name))
    best_weights = save_dir / "weights" / "best.pt"

    print("\n" + "=" * 72)
    print(" เทรนเสร็จแล้ว")
    print("=" * 72)
    print(f" ผลลัพธ์ทั้งหมด : {save_dir}")
    print(f" กราฟผลการเทรน  : {save_dir / 'results.png'}")
    print(f" Confusion Matrix: {save_dir / 'confusion_matrix.png'}  <- ใช้ใส่ในรายงานได้เลย")

    if best_weights.is_file():
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        destination = models_dir / "best.pt"
        shutil.copy2(best_weights, destination)
        print(f" คัดลอกโมเดลที่ดีที่สุดไปที่ : {destination}")
        print("\n ขั้นตอนถัดไป:")
        print("   1) ประเมินผล      :  python tools/evaluate.py")
        print("   2) แปลงเป็น TFLite :  python tools/export_tflite.py")
    else:
        print(f" [เตือน] ไม่พบไฟล์ {best_weights}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
