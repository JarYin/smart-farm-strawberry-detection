"""
ทดสอบกฎการทำงานของ prototype-v1 ทั้งสายตั้งแต่ภาพจนถึงรีเลย์

กฎที่ต้องเป็นจริงเสมอ (โจทย์ของเวอร์ชันนำเสนอผลงาน):
    เจอสีแดง (= สตรอว์เบอร์รี)   -> เลเซอร์ "ไม่" ทำงาน
    เจอสีเขียว (= วัชพืช)        -> เลเซอร์ยิง
    ตรวจจับอะไรไม่ได้เลย          -> เลเซอร์ยิง

เทสต์ชุดนี้อ่านค่าจาก config.yaml ตัวจริงของ branch นี้ ไม่ใช่ค่าที่ตั้งขึ้นเองในเทสต์
จึงทำหน้าที่เป็น "ยามเฝ้าไฟล์ตั้งค่า" ด้วย — ถ้ามีใครแก้ config จนกฎข้อใดข้อหนึ่งเพี้ยน
(เช่น เผลอปิด inverse_mode หรือสลับ target/weed) เทสต์จะฟ้องทันทีก่อนถึงวันนำเสนอ

รันด้วย:  pytest tests/test_prototype_color_logic.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2", reason="ต้องติดตั้ง opencv ก่อนจึงจะรันเทสต์นี้ได้")

from src import overlay
from src.config import load_config
from src.detector import build_detector, effective_conf
from src.logic import STATE_COOLDOWN, STATE_SPRAYING, SprayController
from src.relay import MockRelay

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAME_H, FRAME_W = 480, 640

BGR_RED = (40, 40, 220)
BGR_GREEN = (60, 200, 60)

# กล่องกลาง-ล่างของเฟรม อยู่ใน ROI ของ config.yaml (y1 = 0.45 -> 216 พิกเซล)
IN_ZONE_LEFT = (120, 280, 260, 420)
IN_ZONE_RIGHT = (380, 280, 520, 420)
# กล่องครึ่งบนของเฟรม อยู่นอก ROI
OUT_OF_ZONE = (260, 30, 400, 170)


class FakeClock:
    """นาฬิกาปลอมที่เราสั่งให้เดินหน้าได้เอง (แบบเดียวกับ tests/test_logic.py)"""

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def scene(red=None, green=None) -> np.ndarray:
    """สร้างภาพฉากจำลอง: พื้นหลังมืด + แผ่นสีตามตำแหน่งที่ระบุ"""
    frame = np.full((FRAME_H, FRAME_W, 3), (30, 30, 30), dtype=np.uint8)
    for box, color in ((red, BGR_RED), (green, BGR_GREEN)):
        if box is not None:
            x1, y1, x2, y2 = box
            frame[y1:y2, x1:x2] = color
    return frame


@pytest.fixture(scope="module")
def cfg():
    return load_config(PROJECT_ROOT / "config.yaml")


@pytest.fixture(scope="module")
def detector(cfg):
    return build_detector(cfg)


@pytest.fixture
def controller(cfg):
    """ตัวควบคุมที่ประกอบเหมือน src/main.py ทุกพารามิเตอร์ แต่ใช้นาฬิกาปลอม"""
    clock = FakeClock()
    labels = overlay.labels_for(cfg.relay.actuator)
    ctrl = SprayController(
        spray_cfg=cfg.spray,
        roi_cfg=cfg.roi,
        class_cfg=cfg.classes,
        base_conf=effective_conf(cfg),
        clock=clock,
        geom_cfg=cfg.geometry_filter,
        action_word=labels.action_th,
        device_word=labels.device_th,
    )
    ctrl.test_clock = clock
    return ctrl


def run_scene(detector, controller, frame, frames: int = 8):
    """ป้อนภาพเดิมซ้ำๆ จนพ้นช่วงยืนยันหลายเฟรม แล้วคืนคำตัดสินล่าสุด"""
    decision = None
    for _ in range(frames):
        decision = controller.update(detector.infer(frame), frame.shape[:2])
    return decision


# ---------------------------------------------------------------------------
# กฎหลักสามข้อ
# ---------------------------------------------------------------------------


def test_เจอสีแดงแล้วเลเซอร์ไม่ทำงาน(detector, controller):
    decision = run_scene(detector, controller, scene(red=IN_ZONE_LEFT))

    assert decision.spray_on is False, "สีแดง = สตรอว์เบอร์รี ห้ามยิงเด็ดขาด"
    assert len(decision.analysis.targets) == 1
    assert decision.analysis.targets[0].class_name == "red-object"


def test_เจอสีเขียวแล้วเลเซอร์ยิง(detector, controller):
    decision = run_scene(detector, controller, scene(green=IN_ZONE_LEFT))

    assert decision.spray_on is True
    assert decision.state == STATE_SPRAYING


def test_ตรวจจับอะไรไม่ได้เลยก็ยิงเหมือนกัน(detector, controller):
    decision = run_scene(detector, controller, scene())

    assert decision.spray_on is True, "ฉากว่างเปล่าถือว่าไม่ใช่พืชหลัก -> ต้องยิง"
    assert decision.state == STATE_SPRAYING


def test_เจอทั้งแดงและเขียวพร้อมกันให้สีแดงชนะ(detector, controller):
    """ความปลอดภัยของพืชหลักมาก่อนเสมอ — เจอแดงในเขตเมื่อไหร่ งดยิงทั้งเขต"""
    decision = run_scene(detector, controller, scene(red=IN_ZONE_LEFT, green=IN_ZONE_RIGHT))

    assert decision.spray_on is False
    assert len(decision.analysis.targets) == 1
    assert decision.analysis.actionable_weeds == []
    reasons = " ".join(reason for _det, reason in decision.analysis.suppressed_weeds)
    assert "พืชเป้าหมาย" in reasons


# ---------------------------------------------------------------------------
# ตำแหน่งที่รายงานออกไป (ใช้วาดกรอบและบันทึก log)
# ---------------------------------------------------------------------------


def test_เจอสีเขียวแล้วชี้เป้าไปที่กล่องสีเขียวจริง(detector, controller):
    """ไม่ใช่กล่องสังเคราะห์คลุมทั้งเขต — log และภาพต้องบอกตำแหน่งจริงได้"""
    decision = run_scene(detector, controller, scene(green=IN_ZONE_RIGHT))

    assert len(decision.analysis.actionable_weeds) == 1
    weed = decision.analysis.actionable_weeds[0]
    assert weed.class_name == "green-object"
    assert weed.x1 == pytest.approx(IN_ZONE_RIGHT[0], abs=6)
    assert weed.y1 == pytest.approx(IN_ZONE_RIGHT[1], abs=6)


def test_ไม่เจออะไรเลยจึงใช้กล่องสังเคราะห์ครอบทั้งเขต(detector, controller, cfg):
    decision = run_scene(detector, controller, scene())

    assert len(decision.analysis.actionable_weeds) == 1
    zone = decision.analysis.actionable_weeds[0]
    assert zone.class_name == "non-crop-area"
    assert zone.as_int_box() == cfg.roi.pixel_box(FRAME_W, FRAME_H)


# ---------------------------------------------------------------------------
# ขอบเขตการทำงาน (ROI)
# ---------------------------------------------------------------------------


def test_สีแดงที่อยู่นอกเขตไม่นับว่าเป็นพืชหลัก(detector, controller):
    """สตรอว์เบอร์รีแถวข้างๆ ที่ลำแสงไปไม่ถึง ไม่ควรมีผลต่อการตัดสินใจ"""
    decision = run_scene(detector, controller, scene(red=OUT_OF_ZONE))

    assert decision.spray_on is True
    assert len(decision.analysis.targets) == 1, "ยังตรวจพบและวาดกรอบให้ แต่ไม่นับในการตัดสิน"


def test_สีเขียวที่อยู่นอกเขตถูกระงับแต่เขตยังว่างจึงยิงอยู่ดี(detector, controller):
    decision = run_scene(detector, controller, scene(green=OUT_OF_ZONE))

    assert decision.spray_on is True
    reasons = " ".join(reason for _det, reason in decision.analysis.suppressed_weeds)
    assert "ROI" in reasons
    assert decision.analysis.actionable_weeds[0].class_name == "non-crop-area"


# ---------------------------------------------------------------------------
# กลไกความปลอดภัยเดิมยังทำงานครบ
# ---------------------------------------------------------------------------


def test_ต้องยืนยันครบหลายเฟรมก่อนเริ่มยิง(detector, controller, cfg):
    """กันภาพเบลอ/สีแวบเดียวสั่งยิง — ยังเป็นกลไกเดียวกับระบบจริง"""
    frame = scene(green=IN_ZONE_LEFT)

    for _ in range(cfg.spray.confirm_frames - 1):
        decision = controller.update(detector.infer(frame), frame.shape[:2])
        assert decision.spray_on is False

    decision = controller.update(detector.infer(frame), frame.shape[:2])
    assert decision.spray_on is True
    assert decision.changed is True


def test_ยกแผ่นสีแดงเข้ามากลางคันแล้วหยุดยิง(detector, controller, cfg):
    """สถานการณ์สาธิตจริง: กำลังยิงอยู่ แล้วเอาสตรอว์เบอร์รีเข้ามาในเขต ต้องหยุดทันที"""
    clock = controller.test_clock
    empty = scene()
    with_red = scene(red=IN_ZONE_LEFT)

    run_scene(detector, controller, empty, frames=cfg.spray.confirm_frames)
    assert controller.state == STATE_SPRAYING

    clock.advance(cfg.spray.min_on_seconds + 0.1)  # พ้นเวลาทำงานขั้นต่ำแล้ว
    for _ in range(cfg.spray.release_frames):
        decision = controller.update(detector.infer(with_red), with_red.shape[:2])

    assert decision.spray_on is False
    assert decision.state == STATE_COOLDOWN


def test_ถือแผ่นสีแดงค้างไว้ทั้งคลิปแล้วรีเลย์ไม่เคยทำงานเลย(detector, controller):
    """เทสต์ความปลอดภัยข้อสำคัญที่สุด — ห้ามยิงโดนพืชหลักแม้แต่เฟรมเดียว"""
    relay = MockRelay(pin=17, active_high=False, verbose=False)
    frame = scene(red=IN_ZONE_LEFT)

    try:
        for _ in range(30):
            decision = controller.update(detector.infer(frame), frame.shape[:2])
            relay.set(decision.spray_on)
    finally:
        relay.close()

    assert relay.activation_count == 0
    assert relay.total_on_seconds == 0.0


def test_ระบบตัดการยิงเมื่อทำงานนานเกินกำหนด(detector, controller, cfg):
    """กลไกกันค้างยังอยู่ครบ แม้จะยืดเวลาไว้สำหรับการสาธิต"""
    clock = controller.test_clock
    frame = scene()

    run_scene(detector, controller, frame, frames=cfg.spray.confirm_frames)
    assert controller.state == STATE_SPRAYING

    clock.advance(cfg.spray.max_on_seconds + 0.5)
    decision = controller.update(detector.infer(frame), frame.shape[:2])

    assert decision.spray_on is False
    assert controller.cutoff_events == 1
    assert "ไฟ LED" in decision.reason, "ข้อความต้องเรียกอุปกรณ์ตาม relay.actuator"


# ---------------------------------------------------------------------------
# ไฟล์ตั้งค่าของ branch นี้ต้องอยู่ในโหมดต้นแบบจริงๆ
# ---------------------------------------------------------------------------


def test_ไฟล์ตั้งค่าจริงอยู่ในโหมดตรวจจับด้วยสี(cfg):
    assert cfg.model.resolved_backend() == "color"
    assert cfg.classes.inverse_mode is True, "กฎ 'ไม่เจออะไรเลยก็ยิง' ต้องพึ่งโหมดผกผัน"


def test_ชื่อสีในหมวด_color_ตรงกับบทบาทในหมวด_classes(cfg):
    """ถ้าไม่ตรง ทุกกล่องจะกลายเป็นคลาสที่ไม่รู้จัก แล้วระบบจะยิงตลอดเวลา"""
    assert cfg.color.red_name in cfg.classes.target, "สีแดงต้องเป็นพืชเป้าหมาย"
    assert cfg.color.green_name in cfg.classes.weed, "สีเขียวต้องเป็นวัชพืช"
    assert cfg.classes.role_of(cfg.color.red_name) == "target"
    assert cfg.classes.role_of(cfg.color.green_name) == "weed"


def test_ตัวกรองรูปทรงกล่องถูกปิดในโหมดสี(cfg):
    """ขอบเขตของตัวกรองนั้นวัดจากต้นสตรอว์เบอร์รีจริง ถ้าเปิดไว้จะตัดแผ่นสีทิ้งเกือบหมด"""
    assert cfg.geometry_filter.enabled is False


def test_อุปกรณ์ปลายทางถูกตั้งเป็นไฟ_led(cfg):
    """เปลี่ยนจาก laser เป็น led (2026-09-19) เพราะโมดูลเลเซอร์ไม่ติดที่ 3.3V จากขา GPIO
    ต้องใช้ 5V ผ่านรีเลย์/ทรานซิสเตอร์ ส่วน LED ต่อตรงได้เลย"""
    assert cfg.relay.actuator == "led"


# ---------------------------------------------------------------------------
# การแสดงผล
# ---------------------------------------------------------------------------


def test_คำที่แสดงเปลี่ยนตามอุปกรณ์ปลายทาง():
    assert overlay.labels_for("laser").device == "LASER"
    assert overlay.labels_for("laser").verb == "FIRE"
    assert overlay.labels_for("pump").device == "PUMP"
    assert overlay.labels_for("ไม่รู้จัก").device == "PUMP", "ค่าที่ไม่รู้จักต้องถอยไปใช้ค่าปั๊ม"


def test_วาดภาพผลลัพธ์ได้ทุกสถานการณ์โดยไม่พัง(detector, controller, cfg):
    labels = overlay.labels_for(cfg.relay.actuator)
    roi_box = cfg.roi.pixel_box(FRAME_W, FRAME_H)

    for frame in (scene(), scene(red=IN_ZONE_LEFT), scene(green=IN_ZONE_RIGHT),
                  scene(red=IN_ZONE_LEFT, green=IN_ZONE_RIGHT)):
        decision = controller.update(detector.infer(frame), frame.shape[:2])
        canvas = overlay.render(
            frame, decision, 25.0, roi_box, detector.last_timing_ms, "frame 1", labels
        )
        assert canvas.shape == frame.shape
        assert canvas is not frame, "ต้องวาดลงสำเนา ไม่ใช่ทับภาพต้นฉบับ"
