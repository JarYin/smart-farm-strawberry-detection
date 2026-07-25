"""
วัดความเร็วและบันทึกเหตุการณ์

ไฟล์ CSV ที่ได้จากโมดูลนี้เอาไปทำตารางผลการทดลองในบทที่ 4 ได้โดยตรง
(เปิดด้วย Excel ได้ทันที เพราะเขียนด้วย utf-8-sig)
"""

from __future__ import annotations

import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterable


class FpsMeter:
    """วัดเฟรมต่อวินาทีแบบเฉลี่ยเคลื่อนที่

    ใช้ค่าเฉลี่ยจากหน้าต่างล่าสุดแทนค่าเฉลี่ยรวมทั้งหมด เพราะเราต้องการเห็น
    ความเร็วปัจจุบัน ไม่ใช่ค่าที่ถูกถ่วงด้วยช่วงอุ่นเครื่องตอนเริ่มโปรแกรม
    """

    def __init__(self, window: int = 30) -> None:
        self.window = max(2, int(window))
        self._timestamps: deque[float] = deque(maxlen=self.window)
        self.total_frames = 0
        self._started_at: float | None = None

    def tick(self) -> float:
        """เรียก 1 ครั้งต่อ 1 เฟรม คืนค่า FPS ปัจจุบัน"""
        now = time.perf_counter()
        if self._started_at is None:
            self._started_at = now
        self._timestamps.append(now)
        self.total_frames += 1
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span

    @property
    def average_fps(self) -> float:
        """FPS เฉลี่ยตลอดการทำงาน (ใช้รายงานผลรวม)"""
        if self._started_at is None or self.total_frames < 2:
            return 0.0
        span = time.perf_counter() - self._started_at
        return (self.total_frames - 1) / span if span > 0 else 0.0

    @property
    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.perf_counter() - self._started_at


EVENT_FIELDS = [
    "timestamp",
    "frame",
    "event",
    "state",
    "pump",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "weed_count",
    "target_count",
    "fps",
    "infer_ms",
    "reason",
]


class EventLogger:
    """เขียนบันทึกเหตุการณ์ลงไฟล์ CSV

    บันทึกเฉพาะ "ตอนสถานะเปลี่ยน" ไม่ใช่ทุกเฟรม เพราะที่ 10 FPS การเขียนทุกเฟรม
    จะได้ไฟล์ 36,000 แถวต่อชั่วโมงโดยไม่ได้ข้อมูลเพิ่มขึ้นเลย
    """

    def __init__(self, path: str | Path, enabled: bool = True) -> None:
        self.enabled = bool(enabled) and bool(path)
        self.path = Path(path) if path else None
        self._file = None
        self._writer = None
        self.rows_written = 0

        if not self.enabled or self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        # utf-8-sig ทำให้ Excel บน Windows อ่านภาษาไทยได้ถูกต้อง
        self._file = self.path.open("a", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._file, fieldnames=EVENT_FIELDS, extrasaction="ignore")
        if is_new:
            self._writer.writeheader()
            self._file.flush()

    def log(self, **row) -> None:
        if not self.enabled or self._writer is None:
            return
        row.setdefault("timestamp", datetime.now().isoformat(timespec="milliseconds"))
        self._writer.writerow(row)
        self.rows_written += 1
        # flush ทุกแถว เพราะการทดลองภาคสนามอาจถูกตัดไฟกลางคัน ข้อมูลต้องไม่หาย
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def write_summary_csv(path: str | Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> Path:
    """เขียนตารางสรุปลงไฟล์ CSV (ใช้โดยเครื่องมือใน tools/)"""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return path

    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def format_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """จัดตารางข้อความสำหรับพิมพ์ในเทอร์มินัล"""
    if not rows:
        return "(ไม่มีข้อมูล)"

    columns = columns or list(rows[0].keys())
    widths = {c: len(str(c)) for c in columns}
    for row in rows:
        for c in columns:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))

    header = "  ".join(str(c).ljust(widths[c]) for c in columns)
    separator = "  ".join("-" * widths[c] for c in columns)
    body = "\n".join("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns) for row in rows)
    return f"{header}\n{separator}\n{body}"
