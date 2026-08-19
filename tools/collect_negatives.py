"""
เก็บภาพ "hard negative" — ใบไม้/พืชชนิดอื่นที่ไม่ใช่สตรอว์เบอร์รี แต่หน้าตาคล้ายกันพอจะหลอกโมเดล

ที่มา: ทดสอบ best.pt แล้วพบว่าโมเดลให้ความมั่นใจสูง (70-80%) กับใบไม้พืชอื่นที่ไม่ใช่
สตรอว์เบอร์รี เพราะชุดข้อมูลตอนเทรนมีแต่ภาพ background เป็นพื้นดิน/พื้นเปล่า ไม่มีภาพ
"ใบไม้พืชอื่นที่คล้ายกัน" เลย โมเดลเลยไม่เคยเรียนรู้ว่าใบไม้ทรงคล้ายกันไม่ใช่สตรอว์เบอร์รีเสมอไป

เครื่องมือนี้ถ่ายภาพแล้วบันทึกคู่กับไฟล์ label ว่างเปล่า (ตามรูปแบบ YOLOv8: ภาพที่ไม่มี
label = background) พร้อมรัน best.pt ปัจจุบันทันทีเพื่อบอกว่าภาพนี้ "หลอกโมเดลได้จริง" หรือไม่
ช่วยให้เล็งเก็บภาพที่มีประโยชน์กับการเทรนรอบหน้า ไม่ใช่ถ่ายสุ่มไปเรื่อยๆ

วิธีใช้:
    python tools/collect_negatives.py --count 30                  # ถ่ายจากเว็บแคม 30 ใบ
    python tools/collect_negatives.py --count 30 --interval 1.5   # หน่วงเวลาระหว่างภาพ
    python tools/collect_negatives.py --count 10 --no-check        # ไม่ต้องรันโมเดลเช็ค (เร็วกว่า)

ปุ่มลัด (ต้องมีหน้าต่างแสดงผล): s = บันทึกทันที | q/ESC = ออก

เอาโฟลเดอร์ที่ได้ (images/ + labels/) ไปรวมกับ datasets/train ก่อนเทรนใหม่ด้วย tools/train.py
"""

import _bootstrap  # noqa: F401

import argparse
from datetime import datetime
from pathlib import Path

from src.camera import CameraError, open_source
from src.config import PROJECT_ROOT, load_config
from src.vision_utils import imwrite


def main() -> int:
    parser = argparse.ArgumentParser(description="เก็บภาพ hard-negative สำหรับเทรนรอบหน้า")
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", choices=["webcam", "picamera2", "video", "images"])
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--count", type=int, default=20, help="จำนวนภาพที่จะเก็บ")
    parser.add_argument("--interval", type=float, default=1.0, help="ถ่ายทุกกี่วินาที (โหมดอัตโนมัติ)")
    parser.add_argument("--out", default="datasets/hard_negatives", help="โฟลเดอร์ปลายทาง")
    parser.add_argument("--weights", default=None, help="โมเดลที่ใช้เช็ค (ค่าเริ่มต้นตาม config)")
    parser.add_argument("--no-check", action="store_true", help="ไม่ต้องรันโมเดลเช็คความมั่นใจ")
    parser.add_argument("--no-window", action="store_true", help="ไม่เปิดหน้าต่าง ถ่ายอัตโนมัติล้วนๆ")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg.camera.source = args.source
    if args.index is not None:
        cfg.camera.index = args.index
    if args.path:
        cfg.camera.path = args.path
    cfg.validate()

    out_dir = cfg.resolve_path(args.out)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    checker = None
    if not args.no_check:
        weights = Path(args.weights).resolve() if args.weights else PROJECT_ROOT / "models" / "best.pt"
        if weights.is_file():
            try:
                from ultralytics import YOLO

                checker = YOLO(str(weights))
            except ImportError:
                print("[เตือน] ไม่พบ ultralytics -> ข้ามการเช็คความมั่นใจ (ใช้ --no-check ปิดเตือนนี้ได้)")
        else:
            print(f"[เตือน] ไม่พบโมเดล {weights} -> ข้ามการเช็คความมั่นใจ")

    print("=" * 72)
    print(" เก็บภาพ Hard-Negative")
    print("=" * 72)
    print(f" ปลายทาง : {out_dir}")
    print(f" เป้าหมาย : {args.count} ภาพ")
    print(" ถ่ายอะไร : ใบไม้/พืชชนิดอื่นที่ไม่ใช่สตรอว์เบอร์รี ยิ่งหน้าตาคล้ายยิ่งมีประโยชน์")
    print(" ปุ่มลัด  : s = บันทึกทันที | q/ESC = ออกก่อนครบ")
    print("-" * 72)

    try:
        source = open_source(cfg.camera, PROJECT_ROOT)
    except CameraError as exc:
        print(f"[ผิดพลาด] {exc}")
        return 1

    show_window = not args.no_window
    cv2 = None
    if show_window:
        try:
            import cv2 as _cv2

            cv2 = _cv2
        except ImportError:
            show_window = False

    saved = 0
    fooled = 0
    window = "Collect Hard-Negative — s=save now  q=quit"

    import time

    try:
        source.warmup()
        last_capture = 0.0
        while saved < args.count:
            frame = source.read()
            if frame is None:
                print("[camera] ไม่มีภาพแล้ว")
                break

            now = time.monotonic()
            should_save = (now - last_capture) >= args.interval

            key = 255
            if show_window:
                canvas = frame.copy()
                cv2.putText(
                    canvas, f"saved {saved}/{args.count}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    should_save = True

            if should_save:
                last_capture = now
                stem = f"neg_{datetime.now():%Y%m%d_%H%M%S_%f}"
                img_path = images_dir / f"{stem}.jpg"
                label_path = labels_dir / f"{stem}.txt"
                imwrite(img_path, frame)
                label_path.write_text("", encoding="utf-8")  # ว่างเปล่า = background ตามรูปแบบ YOLO
                saved += 1

                note = ""
                if checker is not None:
                    result = checker.predict(source=frame, imgsz=cfg.model.imgsz, conf=0.25, verbose=False)[0]
                    if len(result.boxes):
                        best_conf = float(result.boxes.conf.max())
                        fooled += 1
                        note = f"  [หลอกโมเดลได้! conf={best_conf:.2f} -> ภาพนี้มีประโยชน์มาก]"
                    else:
                        note = "  [โมเดลไม่หลง -> เป็น negative ธรรมดา ยังเก็บไว้ได้]"

                print(f"  ({saved}/{args.count}) {img_path.name}{note}")

            if key in (ord("q"), 27):
                print("[collect] ผู้ใช้ออกก่อนครบ")
                break

    except KeyboardInterrupt:
        print("\n[collect] ผู้ใช้กด Ctrl+C")
    finally:
        source.release()
        if show_window:
            cv2.destroyAllWindows()

    print("\n" + "=" * 72)
    print(f" เก็บได้ {saved} ภาพ ที่ {out_dir}")
    if checker is not None:
        print(f" ในจำนวนนี้หลอกโมเดลปัจจุบันได้ {fooled} ภาพ ({fooled}/{saved if saved else 1})")
    print(" ขั้นตอนถัดไป:")
    print(f"   1. ตรวจภาพใน {images_dir} คร่าวๆ ทิ้งภาพเบลอ/ซ้ำมากเกินไป")
    print(f"   2. copy images/*.jpg และ labels/*.txt ไปรวมกับ datasets/train/images และ train/labels")
    print("   3. เทรนใหม่ด้วย  python tools/train.py")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
