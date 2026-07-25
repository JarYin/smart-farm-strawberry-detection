"""
วัดความเร็วการประมวลผลจริงบนเครื่องที่รันอยู่

ตัวเลข 7-10 FPS ที่ระบุในเอกสารโครงงานคือค่าที่คาดหวังบน Raspberry Pi 4
สคริปต์นี้ใช้พิสูจน์ว่าของจริงทำได้เท่าไร แล้วนำไปใส่ตารางผลการทดลอง

วัดแยก 3 ขั้นเพื่อให้รู้ว่าคอขวดอยู่ตรงไหน:
    preprocess  ย่อภาพ + letterbox
    inference   ตัวโมเดลเอง
    postprocess ถอดผลลัพธ์ + NMS

วิธีใช้ (รันบน Pi ด้วยคำสั่งเดียวกัน):
    python tools/benchmark.py
    python tools/benchmark.py --weights models/best_float32.tflite --runs 100
    python tools/benchmark.py --images datasets/valid/images
"""

import _bootstrap  # noqa: F401

import argparse
import platform
import statistics
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.detector import build_detector
from src.stats import format_table, write_summary_csv
from src.vision_utils import imread

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def describe_machine() -> str:
    """อธิบายเครื่องที่กำลังรัน — ต้องระบุในรายงานว่าตัวเลขนี้วัดบนอะไร"""
    parts = [platform.system(), platform.machine(), f"Python {platform.python_version()}"]
    try:
        with open("/proc/device-tree/model", "r", encoding="utf-8", errors="ignore") as fh:
            parts.insert(0, fh.read().strip("\x00").strip())
    except OSError:
        processor = platform.processor()
        if processor:
            parts.insert(0, processor)
    return " | ".join(parts)


def load_frames(images_dir: Path | None, count: int, size: tuple[int, int]) -> list[np.ndarray]:
    """เตรียมภาพสำหรับทดสอบ — ใช้ภาพจริงถ้ามี ไม่งั้นสุ่มภาพขึ้นมา

    ภาพจริงให้ตัวเลขที่น่าเชื่อถือกว่า เพราะจำนวนวัตถุที่ตรวจพบมีผลต่อเวลา NMS
    """
    if images_dir is not None:
        files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        frames = []
        for path in files[:count]:
            image = imread(path)
            if image is not None:
                frames.append(image)
        if frames:
            print(f"[benchmark] ใช้ภาพจริง {len(frames)} ภาพจาก {images_dir}")
            return frames
        print(f"[benchmark] เตือน: อ่านภาพจาก {images_dir} ไม่ได้ -> ใช้ภาพสุ่มแทน")

    print(f"[benchmark] ใช้ภาพสุ่มขนาด {size[0]}x{size[1]} (ตัวเลขจะใกล้เคียงของจริงพอสมควร)")
    rng = np.random.default_rng(seed=42)
    return [rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8) for _ in range(min(count, 20))]


def summarize(name: str, values_ms: list[float]) -> dict:
    if not values_ms:
        return {"stage": name, "mean_ms": 0, "median_ms": 0, "min_ms": 0, "max_ms": 0, "p95_ms": 0}
    ordered = sorted(values_ms)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "stage": name,
        "mean_ms": round(statistics.fmean(values_ms), 2),
        "median_ms": round(statistics.median(values_ms), 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
        "p95_ms": round(ordered[p95_index], 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="วัดความเร็วการประมวลผลของโมเดล")
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights", default=None, help="พาธโมเดล (แทนค่าใน config)")
    parser.add_argument("--runs", type=int, default=60, help="จำนวนรอบที่วัด (ไม่รวมรอบอุ่นเครื่อง)")
    parser.add_argument("--warmup", type=int, default=10, help="จำนวนรอบอุ่นเครื่องที่ไม่นำมาคิด")
    parser.add_argument("--images", default=None, help="โฟลเดอร์ภาพจริงสำหรับทดสอบ")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.weights:
        cfg.model.weights = args.weights
        cfg.validate()

    images_dir = Path(args.images).resolve() if args.images else None
    if images_dir is not None and not images_dir.is_dir():
        print(f"[เตือน] ไม่พบโฟลเดอร์ {images_dir} -> จะใช้ภาพสุ่มแทน")
        images_dir = None

    print("=" * 72)
    print(" วัดความเร็วการประมวลผล (Benchmark)")
    print("=" * 72)
    print(f" เครื่อง  : {describe_machine()}")
    print(f" โมเดล   : {cfg.model.weights}  (เอนจิ้น {cfg.model.resolved_backend()})")
    print(f" ขนาดภาพ : {cfg.model.imgsz}x{cfg.model.imgsz}")
    print("-" * 72)

    detector = build_detector(cfg)
    frames = load_frames(images_dir, max(args.runs, 20), (cfg.camera.width, cfg.camera.height))

    print(f"\n อุ่นเครื่อง {args.warmup} รอบ...")
    for index in range(args.warmup):
        detector.infer(frames[index % len(frames)])

    print(f" วัดผล {args.runs} รอบ...\n")
    stages = {"preprocess": [], "inference": [], "postprocess": []}
    totals: list[float] = []
    detection_counts: list[int] = []

    for index in range(args.runs):
        frame = frames[index % len(frames)]
        detections = detector.infer(frame)
        timing = detector.last_timing_ms
        for stage in stages:
            stages[stage].append(float(timing.get(stage, 0.0)))
        totals.append(sum(float(v) for v in timing.values()))
        detection_counts.append(len(detections))

    detector.close()

    rows = [summarize(stage, values) for stage, values in stages.items()]
    rows.append(summarize("TOTAL", totals))
    print(format_table(rows, ["stage", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms"]))

    mean_total = statistics.fmean(totals) if totals else 0.0
    fps = 1000.0 / mean_total if mean_total > 0 else 0.0
    # รถต้องวิ่งช้าพอที่ AI จะประมวลผลทันในระยะที่หัวฉีดครอบคลุม
    # สมมติหัวฉีดครอบคลุมช่วง 15 ซม. และต้องเห็นวัชพืชอย่างน้อย 3 เฟรมก่อนพ่น
    max_speed_cm_s = fps * 15.0 / max(1, cfg.spray.confirm_frames)

    print(f"\n เวลาเฉลี่ยต่อเฟรม : {mean_total:.1f} มิลลิวินาที")
    print(f" ความเร็ว          : {fps:.1f} FPS")
    print(f" วัตถุที่ตรวจพบเฉลี่ย : {statistics.fmean(detection_counts):.1f} ชิ้นต่อเฟรม")
    print(
        f"\n ประเมินความเร็วรถสูงสุด: ~{max_speed_cm_s:.0f} ซม./วินาที"
        f"\n   (คำนวณจาก {fps:.1f} FPS x ระยะหัวฉีด 15 ซม. หาร {cfg.spray.confirm_frames} เฟรมที่ต้องยืนยัน)"
        "\n   ถ้ารถวิ่งเร็วกว่านี้ ระบบอาจตรวจไม่ทันหรือพ่นช้าเกินจุด"
    )

    if fps < 5:
        print("\n [เตือน] ต่ำกว่า 5 FPS — ลองใช้โมเดล int8, ลด imgsz หรือเพิ่ม model.num_threads")
    elif fps >= 7:
        print("\n ผ่านเกณฑ์ 7-10 FPS ตามที่ระบุไว้ในเอกสารโครงงาน")

    output_path = PROJECT_ROOT / "logs" / "benchmark.csv"
    write_summary_csv(output_path, rows, ["stage", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms"])
    print(f"\n บันทึกตารางไว้ที่ {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
