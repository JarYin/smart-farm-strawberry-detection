"""
วาดผลลัพธ์ลงบนภาพ — ใช้ตอนพัฒนาและใช้ทำภาพประกอบรายงาน

สีที่ใช้ (รูปแบบ BGR ตามมาตรฐาน OpenCV) — สื่อ "การกระทำ" ไม่ใช่สีของวัตถุ:
  เขียว  = พืชเป้าหมาย (ปล่อยผ่าน)
  แดง    = วัชพืชที่สั่งพ่น/ยิง
  เหลือง = วัชพืชที่ถูกระงับ (นอก ROI หรือชิดพืชหลัก)
  ฟ้า    = กรอบ ROI
ใน prototype-v1 ที่ตรวจจับด้วยสี กรอบสีแดงจึงอาจล้อมแผ่น "สีเขียว" ได้ — ไม่ใช่ความผิดพลาด
กรอบบอกว่าระบบจะทำอะไรกับวัตถุนั้น ส่วนป้ายกำกับบอกว่ามันคือคลาสอะไร
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .detection import Detection
from .logic import STATE_COOLDOWN, STATE_IDLE, STATE_SPRAYING, Decision

COLOR_TARGET = (80, 220, 80)
COLOR_WEED = (60, 60, 235)
COLOR_SUPPRESSED = (60, 200, 235)
COLOR_ROI = (235, 180, 60)
COLOR_PANEL = (28, 28, 28)
COLOR_TEXT = (245, 245, 245)

STATE_COLORS = {
    STATE_IDLE: (200, 200, 200),
    STATE_SPRAYING: (60, 60, 235),
    STATE_COOLDOWN: (60, 200, 235),
}

# แปลสถานะเป็นข้อความอังกฤษ เพราะ OpenCV วาดฟอนต์ไทยไม่ได้
STATE_LABELS = {
    STATE_IDLE: "IDLE",
    STATE_SPRAYING: "SPRAYING",
    STATE_COOLDOWN: "COOLDOWN",
}


@dataclass(frozen=True)
class ActuatorLabels:
    """คำที่ใช้เรียกอุปกรณ์ปลายทางของรีเลย์ — เปลี่ยนแค่ "คำ" ไม่เปลี่ยนตรรกะใดๆ

    ระบบจริงต่อรีเลย์เข้ากับปั๊มพ่นยา ส่วน prototype-v1 ที่ใช้นำเสนอผลงานต่อกับเลเซอร์
    ชื่อภายในโค้ด (สถานะ SPRAYING, ฟิลด์ spray_on, เหตุการณ์ SPRAY_START ในไฟล์ CSV)
    ยังใช้คำเดิมทั้งหมดเพื่อไม่ให้เครื่องมือวิเคราะห์ผลและไฟล์ log เก่าใช้ไม่ได้ —
    เปลี่ยนเฉพาะข้อความที่ "คนดู" เห็นบนหน้าจอกับคอนโซลเท่านั้น

    ฟิลด์ที่ลงท้ายด้วย _th ใช้กับคอนโซล (พิมพ์ภาษาไทยได้) ที่เหลือเป็น ASCII
    สำหรับวาดลงภาพ เพราะฟอนต์ในตัวของ OpenCV ไม่มีตัวอักษรไทย
    """

    device: str        # ป้ายในแผง HUD           เช่น "PUMP" / "LASER"
    state: str         # ชื่อสถานะขณะทำงาน       เช่น "SPRAYING" / "FIRING"
    verb: str          # คำบนกรอบวัตถุที่จะจัดการ  เช่น "SPRAY" / "FIRE"
    zone: str          # ป้ายกำกับกรอบ ROI       เช่น "SPRAY ZONE" / "LASER ZONE"
    device_th: str     # ชื่ออุปกรณ์ภาษาไทย       เช่น "ปั๊ม" / "เลเซอร์"
    action_th: str     # คำกริยาภาษาไทย          เช่น "พ่น" / "ยิง"


PUMP_LABELS = ActuatorLabels(
    device="PUMP", state="SPRAYING", verb="SPRAY", zone="SPRAY ZONE",
    device_th="ปั๊ม", action_th="พ่น",
)
LASER_LABELS = ActuatorLabels(
    device="LASER", state="FIRING", verb="FIRE", zone="LASER ZONE",
    device_th="เลเซอร์", action_th="ยิง",
)

ACTUATOR_LABELS = {"pump": PUMP_LABELS, "laser": LASER_LABELS}


def labels_for(actuator: str) -> ActuatorLabels:
    """คืนชุดคำของอุปกรณ์ที่ระบุ (relay.actuator ใน config.yaml)"""
    return ACTUATOR_LABELS.get(actuator, PUMP_LABELS)


def ascii_safe(text: str) -> str:
    """แทนอักขระที่ไม่ใช่ ASCII ด้วย '?'

    ฟอนต์ในตัวของ OpenCV (Hershey) ไม่มีตัวอักษรไทย ถ้าวาดข้อความไทยลงภาพตรงๆ
    จะได้สี่เหลี่ยมหรือขยะ ข้อความภาษาไทยจึงส่งไปที่คอนโซลและไฟล์ CSV แทน
    """
    return text.encode("ascii", "replace").decode("ascii")


def _draw_box(frame, det: Detection, color, label: str, thickness: int = 2) -> None:
    import cv2

    label = ascii_safe(label)
    x1, y1, x2, y2 = det.as_int_box()
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    # วางป้ายไว้เหนือกรอบ ถ้าชนขอบบนให้ย้ายลงมาไว้ในกรอบแทน
    top = y1 - text_h - baseline - 4
    if top < 0:
        top = y1 + 2
    cv2.rectangle(frame, (x1, top), (x1 + text_w + 6, top + text_h + baseline + 4), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 3, top + text_h + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_roi(
    frame: np.ndarray,
    roi_box: tuple[int, int, int, int],
    labels: ActuatorLabels = PUMP_LABELS,
) -> None:
    """วาดกรอบขอบเขตการทำงาน (ระยะที่หัวฉีด/ลำแสงครอบคลุม)"""
    import cv2

    x1, y1, x2, y2 = roi_box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_ROI, -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ROI, 1)
    cv2.putText(
        frame, labels.zone, (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ROI, 1, cv2.LINE_AA
    )


def draw_detections(
    frame: np.ndarray, decision: Decision, labels: ActuatorLabels = PUMP_LABELS
) -> None:
    """วาดกรอบวัตถุทั้งหมดตามบทบาทที่ตรรกะจัดไว้"""
    analysis = decision.analysis

    for det in analysis.targets:
        _draw_box(frame, det, COLOR_TARGET, f"{det.class_name} {det.confidence:.2f} PASS")

    for det in analysis.actionable_weeds:
        _draw_box(
            frame, det, COLOR_WEED, f"{det.class_name} {det.confidence:.2f} {labels.verb}", thickness=3
        )

    for det, _reason in analysis.suppressed_weeds:
        _draw_box(frame, det, COLOR_SUPPRESSED, f"{det.class_name} {det.confidence:.2f} HOLD")


def draw_hud(
    frame: np.ndarray,
    decision: Decision,
    fps: float,
    timing_ms: dict | None = None,
    extra_lines: Sequence[str] = (),
    labels: ActuatorLabels = PUMP_LABELS,
) -> None:
    """วาดแผงข้อมูลสถานะมุมบนซ้าย"""
    import cv2

    state_color = STATE_COLORS.get(decision.state, COLOR_TEXT)
    state_text = STATE_LABELS.get(decision.state, decision.state)
    if decision.state == STATE_SPRAYING:
        state_text = labels.state
    lines = [
        f"FPS {fps:5.1f}",
        f"STATE {state_text}",
        f"{labels.device:<5} {'ON' if decision.spray_on else 'OFF'}",
        f"weed {len(decision.analysis.actionable_weeds)}  crop {len(decision.analysis.targets)}",
    ]
    if timing_ms:
        total = sum(timing_ms.values())
        lines.append(f"infer {timing_ms.get('inference', 0.0):.0f}ms / total {total:.0f}ms")
    lines.extend(extra_lines)

    padding = 8
    line_height = 20
    panel_w = 240
    panel_h = padding * 2 + line_height * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for index, text in enumerate(lines):
        color = state_color if index in (1, 2) else COLOR_TEXT
        cv2.putText(
            frame,
            text,
            (padding, padding + line_height * (index + 1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    # ขณะพ่นจริง ตีกรอบแดงรอบภาพให้เห็นชัดในวิดีโอบันทึก
    if decision.spray_on:
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), COLOR_WEED, 4)
        text = f"{labels.state} {decision.spray_elapsed:.1f}s"
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            frame,
            text,
            (width - text_w - 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_WEED,
            2,
            cv2.LINE_AA,
        )


def draw_footer(frame: np.ndarray, text: str) -> None:
    """วาดข้อความอธิบายเหตุผลของคำตัดสินไว้ด้านล่างภาพ"""
    import cv2

    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 26), (width, height), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(
        frame,
        ascii_safe(text)[:110],
        (8, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )


def render(
    frame: np.ndarray,
    decision: Decision,
    fps: float,
    roi_box: tuple[int, int, int, int] | None = None,
    timing_ms: dict | None = None,
    footer: str = "",
    labels: ActuatorLabels = PUMP_LABELS,
) -> np.ndarray:
    """วาดทุกอย่างลงบนสำเนาของเฟรมแล้วคืนภาพผลลัพธ์"""
    canvas = frame.copy()
    if roi_box is not None:
        draw_roi(canvas, roi_box, labels)
    draw_detections(canvas, decision, labels)
    draw_hud(canvas, decision, fps, timing_ms, labels=labels)
    if footer:
        draw_footer(canvas, footer)
    return canvas
