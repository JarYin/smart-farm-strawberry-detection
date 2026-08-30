"""
เครื่องมือจูนค่าสีสำหรับโหมดตรวจจับด้วยสี (model.backend: color)

ทำไมต้องจูน: ค่าสีที่จับได้ขึ้นกับ "แสงหน้างาน" อย่างมาก แผ่นสีแดงใบเดียวกัน
ใต้ไฟ LED ขาวกับใต้ไฟหลืองในห้องประชุมให้ค่า HSV ต่างกันชัดเจน ค่าเริ่มต้นใน
config.yaml ตั้งไว้กว้างพอสำหรับแสงทั่วไป แต่ก่อนขึ้นนำเสนอจริงควรมาลากแถบเลื่อน
ในที่ที่จะสาธิตจริงสัก 2 นาที แล้วคัดลอกค่ากลับไปใส่ config.yaml

หน้าต่างที่เปิดขึ้นมา 3 บาน:
    camera : ภาพจากกล้อง พร้อมกรอบของก้อนสีที่ "ผ่านเกณฑ์ทั้งหมดแล้ว"
    mask   : ภาพขาวดำของสีที่กำลังจูน (ขาว = ผ่าน) — ดูบานนี้เป็นหลัก
    control: แถบเลื่อนปรับค่า

วิธีจูนให้เร็ว:
    1. เริ่มที่ mask ให้แผ่นสีขึ้นเป็นก้อนขาวทึบ ไม่มีรู
    2. ค่อยๆ เพิ่ม S ขั้นต่ำ จนพื้นหลัง/เงา/ผิวมือดำสนิท แต่แผ่นสียังขาวอยู่
    3. เพิ่ม V ขั้นต่ำ ถ้ามุมมืดของภาพยังขาวหลอกอยู่
    4. ปรับ H ให้แคบที่สุดเท่าที่แผ่นสียังขาวครบทั้งแผ่น

วิธีใช้:
    python tools/color_tune.py                 # จูนสีแดง (ค่าเริ่มต้น)
    python tools/color_tune.py --band green    # จูนสีเขียว
    python tools/color_tune.py --source video --path test_images/webcam_test.mp4

ปุ่มลัด:
    q / ESC = ออก (พิมพ์ค่าที่ได้ในรูปแบบ YAML ให้คัดลอกไปวางใน config.yaml)
    b       = สลับระหว่างสีแดง/สีเขียว
    p       = คลิกภาพเพื่ออ่านค่า HSV ตรงจุดนั้น (เปิด/ปิดโหมดหยอด)
    s       = บันทึกภาพ camera + mask ไว้ที่ logs/
"""

import _bootstrap  # noqa: F401

import argparse
from datetime import datetime

import numpy as np

from src import camera as camera_module
from src.config import PROJECT_ROOT, ColorBand, load_config
from src.detector import ColorDetector, effective_conf
from src.vision_utils import imwrite

WINDOW_CAMERA = "camera"
WINDOW_MASK = "mask"
WINDOW_CONTROL = "control"

# ชื่อแถบเลื่อน — สีแดงมีสองช่วง hue จึงมีแถบมากกว่าสีอื่น
TRACKBARS = ("H1 min", "H1 max", "H2 min", "H2 max", "S min", "V min", "min area x1000")


def parse_args():
    parser = argparse.ArgumentParser(description="จูนค่า HSV สำหรับโหมดตรวจจับด้วยสี")
    parser.add_argument("--config", default=None, help="พาธไฟล์ตั้งค่า (ค่าเริ่มต้น: config.yaml)")
    parser.add_argument("--band", choices=["red", "green"], default="red", help="สีที่จะจูนก่อน")
    parser.add_argument("--source", choices=["webcam", "picamera2", "video", "images"], help="แหล่งภาพ")
    parser.add_argument("--path", help="พาธไฟล์วิดีโอหรือโฟลเดอร์ภาพ")
    parser.add_argument("--index", type=int, help="หมายเลขกล้องเว็บแคม")
    return parser.parse_args()


def band_to_sliders(cfg, band_key: str) -> dict:
    """แปลงค่าในไฟล์ตั้งค่าเป็นตำแหน่งเริ่มต้นของแถบเลื่อน

    สีเขียวมีช่วง hue เดียว จึงตั้งช่วงที่สองให้ "ปิด" ด้วยการทำให้ min > max
    ซึ่ง cv2.inRange จะคืนค่าว่างเปล่าเอง (ไม่ต้องมี if พิเศษในลูปวาด)
    """
    ranges = list(getattr(cfg.color, f"{band_key}_hue_ranges"))
    first = ranges[0]
    second = ranges[1] if len(ranges) > 1 else (179, 0)
    return {
        "H1 min": int(first[0]),
        "H1 max": int(first[1]),
        "H2 min": int(second[0]),
        "H2 max": int(second[1]),
        "S min": int(getattr(cfg.color, f"{band_key}_sat_min")),
        "V min": int(getattr(cfg.color, f"{band_key}_val_min")),
        "min area x1000": int(round(cfg.color.min_area_frac * 1000)),
    }


def sliders_to_band(cv2, class_id: int, name: str) -> tuple[ColorBand, float]:
    """อ่านตำแหน่งแถบเลื่อนปัจจุบันกลับมาเป็นแถบสีพร้อมใช้"""
    read = lambda label: cv2.getTrackbarPos(label, WINDOW_CONTROL)  # noqa: E731

    hue_ranges = []
    for lo_label, hi_label in (("H1 min", "H1 max"), ("H2 min", "H2 max")):
        low, high = read(lo_label), read(hi_label)
        if low <= high:  # ช่วงที่ min > max ถือว่าปิดใช้งาน
            hue_ranges.append((low, high))
    if not hue_ranges:
        hue_ranges = [(0, 0)]

    band = ColorBand(
        class_id=class_id,
        name=name,
        hue_ranges=hue_ranges,
        sat_min=read("S min"),
        val_min=read("V min"),
    )
    return band, max(read("min area x1000"), 1) / 1000.0


def yaml_snippet(band_key: str, band: ColorBand, min_area_frac: float) -> str:
    """สร้างข้อความ YAML ให้คัดลอกไปวางใน config.yaml ได้ตรงๆ"""
    ranges = "[" + ", ".join(f"[{lo}, {hi}]" for lo, hi in band.hue_ranges) + "]"
    return (
        "color:\n"
        f"  {band_key}_hue_ranges: {ranges}\n"
        f"  {band_key}_sat_min: {band.sat_min}\n"
        f"  {band_key}_val_min: {band.val_min}\n"
        f"  min_area_frac: {min_area_frac:.3f}"
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.source:
        cfg.camera.source = args.source
    if args.path:
        cfg.camera.path = args.path
    if args.index is not None:
        cfg.camera.index = args.index
    cfg.validate()

    try:
        import cv2
    except ImportError:
        print("[ผิดพลาด] ต้องติดตั้ง opencv ก่อน:  pip install opencv-python")
        return 1

    # เครื่องมือนี้ต้องเปิดหน้าต่างได้ ไม่งั้นไม่มีแถบเลื่อนให้ลาก
    if not hasattr(cv2, "namedWindow") or not hasattr(cv2, "createTrackbar"):
        print("[ผิดพลาด] opencv รุ่นที่ติดตั้งไม่มีส่วนแสดงผลหน้าต่าง (headless)")
        print("          ติดตั้ง opencv-python (ไม่ใช่ opencv-python-headless) แล้วลองใหม่")
        return 1

    detector = ColorDetector(cfg.color, conf=effective_conf(cfg), max_det=cfg.model.max_det)
    source = camera_module.open_source(cfg.camera, PROJECT_ROOT)

    band_key = args.band
    class_id = 0 if band_key == "red" else 1
    band_name = getattr(cfg.color, f"{band_key}_name")

    cv2.namedWindow(WINDOW_CONTROL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROL, 460, 260)
    initial = band_to_sliders(cfg, band_key)
    for label in TRACKBARS:
        upper = 1000 if label == "min area x1000" else (255 if label in ("S min", "V min") else 179)
        cv2.createTrackbar(label, WINDOW_CONTROL, initial[label], upper, lambda _value: None)

    probe = False
    hsv_holder = {"frame": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and probe and hsv_holder["frame"] is not None:
            hsv = hsv_holder["frame"]
            if 0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1]:
                h, s, v = hsv[y, x]
                print(f"[probe] ({x:>4},{y:>4})  H={h:>3}  S={s:>3}  V={v:>3}")

    cv2.namedWindow(WINDOW_CAMERA)
    cv2.setMouseCallback(WINDOW_CAMERA, on_mouse)

    print("=" * 72)
    print(" จูนค่าสีสำหรับโหมดตรวจจับด้วยสี")
    print("=" * 72)
    print(f" แหล่งภาพ : {cfg.camera.source}")
    print(f" กำลังจูน : {band_key} ('{band_name}')")
    print(" ปุ่มลัด   : q=ออก  b=สลับสี  p=โหมดคลิกอ่านค่า  s=บันทึกภาพ")
    print("-" * 72)

    source.warmup()
    band, min_area_frac = sliders_to_band(cv2, class_id, band_name)

    try:
        while True:
            frame = source.read()
            if frame is None:
                print("[color_tune] ไม่มีภาพแล้ว -> จบการทำงาน")
                break

            band, min_area_frac = sliders_to_band(cv2, class_id, band_name)
            detector.min_area_frac = min_area_frac

            hsv = detector.to_hsv(frame)
            hsv_holder["frame"] = hsv
            mask = detector.band_mask(hsv, band)
            detections = detector.boxes_from_mask(mask, band, frame.shape[:2])

            canvas = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det.as_int_box()
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 255), 2)
                cv2.putText(
                    canvas,
                    f"{det.confidence:.2f} {det.width:.0f}x{det.height:.0f}",
                    (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (60, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

            coverage = float(np.count_nonzero(mask)) / mask.size * 100.0
            hud = f"{band_key}  boxes {len(detections)}  mask {coverage:5.1f}%  fill>={detector.conf:.2f}"
            cv2.putText(canvas, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 255), 2, cv2.LINE_AA)
            if probe:
                cv2.putText(
                    canvas, "PROBE ON - click to read HSV", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 255), 1, cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_CAMERA, canvas)
            cv2.imshow(WINDOW_MASK, mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                band_key = "green" if band_key == "red" else "red"
                class_id = 0 if band_key == "red" else 1
                band_name = getattr(cfg.color, f"{band_key}_name")
                for label, value in band_to_sliders(cfg, band_key).items():
                    cv2.setTrackbarPos(label, WINDOW_CONTROL, value)
                print(f"[color_tune] สลับไปจูน {band_key} ('{band_name}')")
            elif key == ord("p"):
                probe = not probe
                print(f"[color_tune] โหมดคลิกอ่านค่า HSV: {'เปิด' if probe else 'ปิด'}")
            elif key == ord("s"):
                stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
                out_dir = PROJECT_ROOT / "logs"
                imwrite(out_dir / f"colortune_{band_key}_{stamp}.jpg", canvas)
                imwrite(out_dir / f"colortune_{band_key}_{stamp}_mask.jpg", mask)
                print(f"[color_tune] บันทึกภาพไว้ที่ {out_dir}")
    except KeyboardInterrupt:
        print("\n[color_tune] ผู้ใช้กด Ctrl+C")
    finally:
        source.release()
        cv2.destroyAllWindows()

    print("-" * 72)
    print(" คัดลอกค่าด้านล่างไปวางทับในหมวด color ของ config.yaml:")
    print("-" * 72)
    print(yaml_snippet(band_key, band, min_area_frac))
    print("-" * 72)
    print(" อย่าลืมจูนอีกสีหนึ่งด้วย (กด b ระหว่างรัน หรือรันใหม่ด้วย --band)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
