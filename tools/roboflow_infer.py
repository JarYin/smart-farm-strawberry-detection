"""
เรียกใช้โมเดลที่ฝากไว้บน Roboflow ผ่าน API เพื่อทดสอบภาพนิ่งทีละภาพ

เหมาะกับ: ทดสอบเร็วๆ ก่อนมีโมเดลของตัวเอง หรือเทียบผลกับโมเดลที่เทรนเอง
ไม่เหมาะกับ: การควบคุมปั๊มแบบเรียลไทม์บน Raspberry Pi
  (ต้องพึ่งอินเทอร์เน็ต + มี latency เครือข่ายเพิ่มทุกเฟรม ดูรายละเอียดใน
  src/detector.py -> RoboflowDetector)

ก่อนใช้งานต้องตั้งค่า API key ก่อน (ห้ามใส่ค่านี้ลงไฟล์ใดๆ ในโปรเจกต์):
    วิธีที่ 1 (ง่ายสุด) — สร้างไฟล์ .env ที่รากโปรเจกต์ (คัดลอกจาก .env.example):
        ROBOFLOW_API_KEY=คีย์ของคุณ
    วิธีที่ 2 — ตั้งตัวแปรสภาพแวดล้อมในเชลล์ก่อนรัน:
        PowerShell:  $env:ROBOFLOW_API_KEY = "คีย์ของคุณ"
        Bash      :  export ROBOFLOW_API_KEY="คีย์ของคุณ"

วิธีใช้:
    python tools/roboflow_infer.py --image photo.jpg
    python tools/roboflow_infer.py --image photo.jpg --save-annotated logs/annotated.jpg
    python tools/roboflow_infer.py --image photo.jpg --model-id strawberry-detection-msf0m/3
"""

import _bootstrap  # noqa: F401

import argparse
import json
import os
from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.detector import DetectorError, RoboflowDetector
from src.overlay import ascii_safe
from src.vision_utils import imread, imwrite


def draw_predictions(frame, detections) -> None:
    """วาดกรอบสี่เหลี่ยมและป้ายชื่อ+ความเชื่อมั่นลงบนภาพ (แก้ไข frame ในที่)"""
    import cv2

    for det in detections:
        x1, y1, x2, y2 = det.as_int_box()
        color = (60, 220, 60)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = ascii_safe(f"{det.class_name} {det.confidence:.2f}")
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + baseline + 4), color, -1)
        cv2.putText(
            frame, label, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="เรียกใช้โมเดล Roboflow ผ่าน API เพื่อทดสอบภาพนิ่ง")
    parser.add_argument("--config", default=None, help="พาธ config.yaml")
    parser.add_argument("--image", required=True, help="พาธไฟล์ภาพที่จะวิเคราะห์")
    parser.add_argument("--model-id", default=None, help="รูปแบบ ชื่อโมเดล/เวอร์ชัน (แทนค่าใน config)")
    parser.add_argument("--api-url", default=None, help="แทนค่า roboflow.api_url ใน config")
    parser.add_argument("--conf", type=float, default=None, help="ค่า confidence ขั้นต่ำ (แทนค่าใน config)")
    parser.add_argument("--save-annotated", default=None, help="บันทึกภาพพร้อมกรอบผลลัพธ์ไปที่พาธนี้")
    parser.add_argument("--json", action="store_true", help="พิมพ์ผลลัพธ์ดิบเป็น JSON แทนตารางอ่านง่าย")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_id = args.model_id or cfg.roboflow.model_id
    api_url = args.api_url or cfg.roboflow.api_url
    conf = args.conf if args.conf is not None else cfg.model.conf

    api_key = os.environ.get(cfg.roboflow.api_key_env, "")
    if not api_key:
        print(f"[ผิดพลาด] ไม่พบตัวแปรสภาพแวดล้อม {cfg.roboflow.api_key_env}")
        print("  ตั้งค่าอย่างใดอย่างหนึ่ง:")
        print(f"    1) สร้างไฟล์ .env ที่รากโปรเจกต์ (คัดลอกจาก .env.example) แล้วใส่ {cfg.roboflow.api_key_env}=...")
        print(f"    2) PowerShell:  $env:{cfg.roboflow.api_key_env} = \"คีย์ของคุณ\"")
        return 1

    image_path = Path(args.image)
    if not image_path.is_absolute():
        candidate = PROJECT_ROOT / image_path
        image_path = candidate if candidate.is_file() else image_path

    frame = imread(image_path)
    if frame is None:
        print(f"[ผิดพลาด] อ่านภาพไม่ได้: {image_path}")
        return 1

    print("=" * 72)
    print(" เรียกใช้โมเดล Roboflow (hosted inference)")
    print("=" * 72)
    print(f" ภาพ      : {image_path}  ({frame.shape[1]}x{frame.shape[0]})")
    print(f" model_id : {model_id}")
    print(f" api_url  : {api_url}")
    print(f" conf     : {conf}")
    print("-" * 72)

    try:
        detector = RoboflowDetector(
            model_id=model_id,
            api_url=api_url,
            api_key=api_key,
            class_names=cfg.classes.names,
            conf=conf,
            iou=cfg.model.iou,
            max_det=cfg.model.max_det,
            timeout=cfg.roboflow.timeout_seconds,
            api_key_env_name=cfg.roboflow.api_key_env,
        )
        detections = detector.infer(frame)
    except DetectorError as exc:
        print(f"\n[ผิดพลาด] {exc}")
        return 1

    timing = detector.last_timing_ms
    total_ms = sum(timing.values())

    if args.json:
        payload = [
            {
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "box_xyxy": [round(v, 1) for v in d.box],
            }
            for d in detections
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"\nพบวัตถุ {len(detections)} ชิ้น  (รวมเวลา {total_ms:.0f} ms — ส่วนใหญ่คือเวลารอเครือข่าย)")
        for det in detections:
            print(f"  {det.class_name:<20} conf={det.confidence:.2f}  กล่อง={tuple(round(v) for v in det.box)}")

        if not detections:
            print("\n  ไม่พบวัตถุใดเลย ลองเช็ค:")
            print("    - conf ตั้งไว้สูงไปหรือไม่ (--conf 0.1 เพื่อทดสอบ)")
            print("    - model_id ถูกต้องหรือไม่ และโมเดลนี้รู้จักวัตถุในภาพนี้จริงหรือไม่")

    if args.save_annotated:
        canvas = frame.copy()
        draw_predictions(canvas, detections)
        out_path = Path(args.save_annotated)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        if imwrite(out_path, canvas):
            print(f"\nบันทึกภาพพร้อมกรอบผลลัพธ์ที่ {out_path}")
        else:
            print(f"\n[เตือน] บันทึกภาพไม่สำเร็จ: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
