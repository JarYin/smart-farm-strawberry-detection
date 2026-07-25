"""
ทดสอบกล้อง — ใช้ตรวจว่ากล้องเปิดได้ ภาพชัดพอ และมุมกล้องครอบคลุมพื้นที่ที่ต้องการ

มีฟังก์ชันวัดความคมชัดของภาพ (Sharpness) ด้วย เพราะเอกสารโครงงานระบุว่า
Motion Blur คือความท้าทายหลักของงานนี้ ตัวเลขนี้ช่วยตัดสินว่าแสงในโรงเรือน
เพียงพอหรือไม่ และต้องเพิ่มไฟ LED หรือเปล่า

วิธีใช้:
    python tools/camera_test.py                 # เปิดกล้องพร้อมกรอบ ROI
    python tools/camera_test.py --list          # ไล่หาว่ามีกล้องหมายเลขไหนใช้ได้บ้าง
    python tools/camera_test.py --source picamera2
    python tools/camera_test.py --capture 20    # ถ่ายภาพเก็บไว้ 20 ใบ (ใช้สร้างชุดข้อมูล)
"""

import _bootstrap  # noqa: F401

import argparse
from datetime import datetime
from pathlib import Path

from src.camera import CameraError, open_source
from src.config import PROJECT_ROOT, load_config
from src.overlay import COLOR_ROI, COLOR_TEXT
from src.vision_utils import imwrite


def list_cameras(max_index: int = 5) -> None:
    """ไล่เปิดกล้องหมายเลข 0..max_index เพื่อดูว่าตัวไหนใช้ได้"""
    import cv2
    import sys

    print("กำลังค้นหากล้องที่ใช้งานได้...\n")
    found = []
    for index in range(max_index + 1):
        capture = (
            cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
        )
        if capture.isOpened():
            ok, frame = capture.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                print(f"  index {index} : ใช้ได้  ({width}x{height})")
                found.append(index)
            else:
                print(f"  index {index} : เปิดได้แต่อ่านภาพไม่ได้")
        capture.release()

    if not found:
        print("  ไม่พบกล้องเลย — ตรวจสายเชื่อมต่อ หรือปิดโปรแกรมอื่นที่ใช้กล้องอยู่")
    else:
        print(f"\nตั้งค่าใน config.yaml เป็น  camera.index: {found[0]}")


def sharpness_score(frame) -> float:
    """วัดความคมชัดด้วยความแปรปรวนของ Laplacian

    ยิ่งค่าสูง ภาพยิ่งคม โดยประมาณ:
      < 50   เบลอมาก (แสงน้อยหรือรถสั่น) — ควรเพิ่มไฟ LED หรือให้รถวิ่งช้าลง
      50-150 พอใช้
      > 150  คมชัดดี
    """
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> int:
    parser = argparse.ArgumentParser(description="ทดสอบกล้องและวัดคุณภาพภาพ")
    parser.add_argument("--config", default=None)
    parser.add_argument("--list", action="store_true", help="ค้นหากล้องที่ใช้งานได้")
    parser.add_argument("--source", choices=["webcam", "picamera2", "video", "images"])
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--capture", type=int, default=0, help="ถ่ายภาพเก็บไว้กี่ใบ (0 = ไม่ถ่าย)")
    parser.add_argument("--interval", type=float, default=1.0, help="ถ่ายทุกกี่วินาที")
    parser.add_argument(
        "--out", default="datasets/captured", help="โฟลเดอร์เก็บภาพที่ถ่าย"
    )
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return 0

    import cv2

    cfg = load_config(args.config)
    if args.source:
        cfg.camera.source = args.source
    if args.index is not None:
        cfg.camera.index = args.index
    if args.path:
        cfg.camera.path = args.path
    cfg.validate()

    output_dir = cfg.resolve_path(args.out)
    if args.capture:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" ทดสอบกล้อง")
    print("=" * 72)
    print(f" แหล่งภาพ : {cfg.camera.source}")
    print(f" ขนาดที่ขอ : {cfg.camera.width}x{cfg.camera.height}")
    print("-" * 72)
    print(" ปุ่มลัด: q = ออก | s = บันทึกภาพทันที")
    if args.capture:
        print(f" โหมดถ่ายภาพ: ถ่าย {args.capture} ใบ ทุก {args.interval} วินาที -> {output_dir}")
    print()

    try:
        source = open_source(cfg.camera, PROJECT_ROOT)
    except CameraError as exc:
        print(f"[ผิดพลาด] {exc}")
        return 1

    captured = 0
    frame_count = 0
    last_capture = 0.0
    sharpness_values: list[float] = []
    window = "Camera Test — q=quit  s=save"

    import time

    try:
        source.warmup()
        while True:
            frame = source.read()
            if frame is None:
                print("[camera] ไม่มีภาพแล้ว")
                break

            frame_count += 1
            sharpness = sharpness_score(frame)
            sharpness_values.append(sharpness)

            canvas = frame.copy()
            height, width = canvas.shape[:2]

            if cfg.roi.enabled:
                x1, y1, x2, y2 = cfg.roi.pixel_box(width, height)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_ROI, 2)
                cv2.putText(
                    canvas, "SPRAY ZONE", (x1 + 5, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ROI, 1
                )

            quality = "SHARP" if sharpness > 150 else ("OK" if sharpness > 50 else "BLURRY!")
            color = (80, 220, 80) if sharpness > 150 else ((60, 200, 235) if sharpness > 50 else (60, 60, 235))
            cv2.putText(
                canvas, f"{width}x{height}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2
            )
            cv2.putText(
                canvas,
                f"sharpness {sharpness:6.1f}  {quality}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            if args.capture:
                cv2.putText(
                    canvas,
                    f"captured {captured}/{args.capture}",
                    (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_TEXT,
                    2,
                )

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF

            now = time.monotonic()
            should_save = key == ord("s")
            if args.capture and captured < args.capture and (now - last_capture) >= args.interval:
                should_save = True
                last_capture = now

            if should_save:
                path = output_dir / f"img_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                imwrite(path, frame)
                captured += 1
                print(f"  บันทึก {path.name}  (ความคมชัด {sharpness:.1f})")
                if args.capture and captured >= args.capture:
                    print("  ถ่ายครบตามจำนวนแล้ว")
                    break

            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        cv2.destroyAllWindows()

    if sharpness_values:
        average = sum(sharpness_values) / len(sharpness_values)
        minimum = min(sharpness_values)
        print("\n" + "=" * 72)
        print(f" จำนวนเฟรมที่ทดสอบ : {frame_count}")
        print(f" ความคมชัดเฉลี่ย    : {average:.1f}")
        print(f" ความคมชัดต่ำสุด    : {minimum:.1f}")
        if average < 50:
            print("\n [เตือน] ภาพเบลอมาก — เพิ่มไฟ LED ที่ตัวรถ หรือให้รถวิ่งช้าลง")
            print("         (เอกสารโครงงานระบุว่า Motion Blur คือความท้าทายหลักของงานนี้)")
        elif average < 150:
            print("\n ภาพพอใช้ได้ แต่ถ้าเพิ่มไฟ LED จะช่วยให้ AI แม่นยำขึ้น")
        else:
            print("\n ภาพคมชัดดี เหมาะกับการนำไปเทรนและใช้งานจริง")
        print("=" * 72)

    if captured:
        print(f"\n ภาพที่ถ่ายไว้ {captured} ใบ อยู่ที่ {output_dir}")
        print(" นำไป label ต่อได้ที่ https://roboflow.com หรือใช้โปรแกรม labelImg")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
