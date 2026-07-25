"""
สรุปไฟล์บันทึกเหตุการณ์ (logs/events.csv) เป็นตารางสำหรับรายงานผลการทดลอง

ระบบจะบันทึกทุกครั้งที่ปั๊มเปลี่ยนสถานะ สคริปต์นี้จับคู่ SPRAY_START กับ
SPRAY_STOP ให้กลายเป็น "เหตุการณ์การพ่น 1 ครั้ง" พร้อมระยะเวลาและความเชื่อมั่น
แล้วสรุปเป็นตัวเลขที่นำไปใส่บทที่ 4 ได้โดยตรง

วิธีใช้:
    python tools/report_events.py
    python tools/report_events.py --csv logs/events.csv --out logs/spray_report.csv
"""

import _bootstrap  # noqa: F401

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT
from src.stats import format_table, write_summary_csv


def parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def build_spray_events(rows: list[dict]) -> list[dict]:
    """จับคู่เหตุการณ์เริ่มพ่นกับหยุดพ่นให้เป็นรายการเดียว"""
    events: list[dict] = []
    pending: dict | None = None

    for row in rows:
        event = (row.get("event") or "").strip()
        timestamp = parse_timestamp(row.get("timestamp", ""))

        if event == "SPRAY_START":
            if pending is not None:
                # เจอ START ซ้อนโดยไม่มี STOP — เกิดได้ถ้าโปรแกรมถูกฆ่ากลางคัน
                pending["duration_s"] = ""
                pending["note"] = "ไม่พบเหตุการณ์หยุดพ่น (โปรแกรมอาจถูกปิดกลางคัน)"
                events.append(pending)
            pending = {
                "no": len(events) + 1,
                "start": row.get("timestamp", ""),
                "stop": "",
                "duration_s": "",
                "class": row.get("class_name", ""),
                "confidence": row.get("confidence", ""),
                "weed_count": row.get("weed_count", ""),
                "fps": row.get("fps", ""),
                "note": "",
                "_start_dt": timestamp,
            }
        elif event == "SPRAY_STOP" and pending is not None:
            pending["stop"] = row.get("timestamp", "")
            start_dt = pending.pop("_start_dt", None)
            if start_dt and timestamp:
                pending["duration_s"] = round((timestamp - start_dt).total_seconds(), 2)
            pending["note"] = row.get("reason", "")
            events.append(pending)
            pending = None

    if pending is not None:
        pending.pop("_start_dt", None)
        pending["note"] = "ไม่พบเหตุการณ์หยุดพ่น (โปรแกรมอาจถูกปิดกลางคัน)"
        events.append(pending)

    for event in events:
        event.pop("_start_dt", None)
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="สรุปผลการทดลองจากไฟล์บันทึกเหตุการณ์")
    parser.add_argument("--csv", default=None, help="พาธไฟล์ events.csv")
    parser.add_argument("--out", default=None, help="พาธไฟล์สรุปที่จะเขียน")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve() if args.csv else PROJECT_ROOT / "logs" / "events.csv"
    if not csv_path.is_file():
        print(f"[ผิดพลาด] ไม่พบไฟล์ {csv_path}")
        print("  รันระบบก่อนเพื่อสร้างไฟล์บันทึก:  python -m src.main")
        return 1

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print(f"ไฟล์ {csv_path} ยังไม่มีข้อมูล")
        return 1

    events = build_spray_events(rows)

    print("=" * 72)
    print(" สรุปผลการทดลองจากไฟล์บันทึกเหตุการณ์")
    print("=" * 72)
    print(f" ไฟล์ต้นทาง        : {csv_path}")
    print(f" จำนวนบรรทัดทั้งหมด : {len(rows)}")
    print(f" จำนวนครั้งที่พ่น    : {len(events)}")

    timestamps = [parse_timestamp(r.get("timestamp", "")) for r in rows]
    timestamps = [t for t in timestamps if t]
    if len(timestamps) >= 2:
        span = (max(timestamps) - min(timestamps)).total_seconds()
        print(f" ช่วงเวลาที่บันทึก   : {min(timestamps):%Y-%m-%d %H:%M:%S} ถึง {max(timestamps):%H:%M:%S}")
        print(f" ระยะเวลารวม       : {span:.1f} วินาที")

    durations = [float(e["duration_s"]) for e in events if isinstance(e["duration_s"], (int, float))]
    if durations:
        print(f" เวลาพ่นรวม        : {sum(durations):.2f} วินาที")
        print(f" เวลาพ่นเฉลี่ยต่อครั้ง : {sum(durations)/len(durations):.2f} วินาที")
        print(f" พ่นสั้นสุด / นานสุด : {min(durations):.2f} / {max(durations):.2f} วินาที")

    class_counts = Counter(e["class"] for e in events if e["class"])
    if class_counts:
        print("\n คลาสที่เป็นเหตุให้พ่น:")
        for name, count in class_counts.most_common():
            print(f"   {name:<20} {count} ครั้ง")

    fps_values = []
    for row in rows:
        try:
            fps_values.append(float(row.get("fps") or 0))
        except ValueError:
            continue
    fps_values = [v for v in fps_values if v > 0]
    if fps_values:
        print(f"\n FPS เฉลี่ยขณะเกิดเหตุการณ์ : {sum(fps_values)/len(fps_values):.2f}")

    if events:
        columns = ["no", "start", "stop", "duration_s", "class", "confidence", "weed_count", "fps"]
        print("\n[รายการการพ่นทั้งหมด]")
        print(format_table(events, columns))

        out_path = Path(args.out).resolve() if args.out else PROJECT_ROOT / "logs" / "spray_report.csv"
        write_summary_csv(out_path, events, columns + ["note"])
        print(f"\n บันทึกตารางสรุปไว้ที่ {out_path}")
        print(" เปิดด้วย Excel แล้วคัดลอกไปใส่บทที่ 4 ได้เลย")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
