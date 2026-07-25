"""
ทดสอบทั้งสายการทำงานตั้งแต่ต้นจนจบ (End-to-End Smoke Test)

ต่อชิ้นส่วนจริงทุกตัวเข้าด้วยกัน — แหล่งภาพ, letterbox, ตรรกะพ่น, รีเลย์, ไฟล์ log —
โดยใช้ตัวตรวจจับปลอมแทนโมเดลจริง ทำให้ทดสอบได้โดยไม่ต้องมีไฟล์ .pt/.tflite

จุดประสงค์คือจับ "ท่อรั่ว" ระหว่างโมดูล ซึ่งเทสต์แยกทีละส่วนมองไม่เห็น
เช่น ส่งพิกัดผิดระบบ หรือเรียกฟังก์ชันด้วยลำดับพารามิเตอร์สลับกัน
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2", reason="ต้องติดตั้ง opencv ก่อนจึงจะรันเทสต์นี้ได้")

from src import overlay
from src.camera import ImageFolderSource
from src.config import CameraCfg, ClassCfg, RoiCfg, SprayCfg
from src.detection import Detection
from src.detector import BaseDetector
from src.logic import SprayController
from src.relay import MockRelay
from src.stats import EventLogger, FpsMeter
from src.vision_utils import imwrite, letterbox, scale_boxes_back

FRAME_W, FRAME_H = 640, 480


class ScriptedDetector(BaseDetector):
    """ตัวตรวจจับปลอมที่คืนผลลัพธ์ตามบทที่เขียนไว้ล่วงหน้า"""

    def __init__(self, script: list[list[Detection]]):
        super().__init__(["strawberry", "weed"], conf=0.25, iou=0.45, max_det=20)
        self.script = script
        self.calls = 0

    def infer(self, frame):
        result = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        self.last_timing_ms = {"preprocess": 1.0, "inference": 8.0, "postprocess": 1.0}
        return result


def weed_box(conf=0.9):
    return Detection(1, "weed", conf, 280.0, 300.0, 360.0, 400.0)


def crop_box(conf=0.9):
    return Detection(0, "strawberry", conf, 60.0, 300.0, 160.0, 400.0)


@pytest.fixture
def image_folder(tmp_path):
    """สร้างโฟลเดอร์ภาพจำลอง 3 ใบ"""
    folder = tmp_path / "frames"
    folder.mkdir()
    for index in range(3):
        image = np.full((FRAME_H, FRAME_W, 3), 40 * (index + 1), dtype=np.uint8)
        # ใช้ imwrite ของโปรเจกต์ เพราะพาธชั่วคราวของ pytest มีตัวอักษรไทยอยู่ด้วย
        assert imwrite(folder / f"frame_{index:02d}.jpg", image)
    return folder


def test_อ่านภาพจากโฟลเดอร์ได้ครบตามจำนวนที่กำหนด(image_folder):
    cfg = CameraCfg(source="images", path=str(image_folder), loop=False, hold_frames=2)
    source = ImageFolderSource(cfg, image_folder)

    frames = []
    while True:
        frame = source.read()
        if frame is None:
            break
        frames.append(frame)
    source.release()

    assert len(frames) == 6, "3 ภาพ x ค้างไว้ภาพละ 2 เฟรม"
    assert frames[0].shape == (FRAME_H, FRAME_W, 3)


def test_letterbox_แล้วแปลงพิกัดกลับได้ตำแหน่งเดิม():
    """ทดสอบไป-กลับ: กล่องที่รู้ตำแหน่งจริง ต้องกลับมาที่เดิมหลังผ่าน letterbox"""
    image = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    padded, ratio, pad = letterbox(image, 320)

    assert padded.shape[:2] == (320, 320)
    assert ratio == pytest.approx(0.5)

    original = np.array([[100.0, 120.0, 300.0, 400.0]], dtype=np.float32)
    # จำลองการแปลงไปยังระบบพิกัดของภาพ letterbox
    in_letterbox = original * ratio
    in_letterbox[:, [0, 2]] += pad[0]
    in_letterbox[:, [1, 3]] += pad[1]

    recovered = scale_boxes_back(in_letterbox, ratio, pad, (FRAME_H, FRAME_W))
    assert recovered[0].tolist() == pytest.approx(original[0].tolist(), abs=0.5)


def test_ทั้งระบบทำงานครบวงจรและบันทึกเหตุการณ์ถูกต้อง(image_folder, tmp_path):
    """จำลองรถวิ่งผ่าน: เจอสตรอว์เบอร์รี -> เจอวัชพืช -> วัชพืชหายไป"""
    script = (
        [[crop_box()]] * 3          # เฟรม 1-3  : พืชเป้าหมาย ไม่ต้องพ่น
        + [[weed_box()]] * 5        # เฟรม 4-8  : วัชพืช ต้องเริ่มพ่นที่เฟรม 6
        + [[]] * 8                  # เฟรม 9-16 : ไม่มีอะไร ต้องหยุดพ่น
    )
    detector = ScriptedDetector(script)

    camera_cfg = CameraCfg(source="images", path=str(image_folder), loop=True, hold_frames=1)
    source = ImageFolderSource(camera_cfg, image_folder)

    classes = ClassCfg(names=["strawberry", "weed"], target=["strawberry"], weed=["weed"])
    controller = SprayController(
        SprayCfg(confirm_frames=3, release_frames=2, min_on_seconds=0.0, cooldown_seconds=0.0),
        RoiCfg(enabled=True, x1=0.15, y1=0.45, x2=0.85, y2=1.0),
        classes,
        base_conf=0.45,
    )
    relay = MockRelay(pin=17, active_high=False, verbose=False)
    log_path = tmp_path / "events.csv"
    logger = EventLogger(log_path)
    fps_meter = FpsMeter()

    spray_frames = []
    try:
        for frame_no in range(1, len(script) + 1):
            frame = source.read()
            assert frame is not None

            detections = detector.infer(frame)
            decision = controller.update(detections, frame.shape[:2])
            relay.set(decision.spray_on)
            fps = fps_meter.tick()

            if decision.spray_on:
                spray_frames.append(frame_no)
            if decision.changed:
                logger.log(
                    frame=frame_no,
                    event="SPRAY_START" if decision.spray_on else "SPRAY_STOP",
                    state=decision.state,
                    pump="ON" if decision.spray_on else "OFF",
                    weed_count=len(decision.analysis.actionable_weeds),
                    target_count=len(decision.analysis.targets),
                    fps=f"{fps:.2f}",
                    reason=decision.reason,
                )

            # ต้องวาดภาพได้โดยไม่ error ทุกเฟรม (ทดสอบโค้ดวาดกรอบไปในตัว)
            canvas = overlay.render(
                frame, decision, fps, controller.roi.pixel_box(FRAME_W, FRAME_H), detector.last_timing_ms
            )
            assert canvas.shape == frame.shape
    finally:
        controller.force_stop()
        relay.close()
        logger.close()
        source.release()

    # เริ่มพ่นที่เฟรม 6 (เฟรมวัชพืชที่ 3) และหยุดหลังไม่เจอ 2 เฟรม
    assert spray_frames[0] == 6, f"ควรเริ่มพ่นที่เฟรม 6 แต่ได้ {spray_frames[0]}"
    assert 9 in spray_frames, "เฟรมแรกที่วัชพืชหายไป ยังต้องพ่นอยู่"
    assert 11 not in spray_frames, "ครบ 2 เฟรมที่ไม่เจอวัชพืชแล้ว ต้องหยุดพ่น"

    assert relay.is_on is False, "ปิดโปรแกรมแล้วปั๊มต้องไม่ค้างเปิด"
    assert relay.activation_count == 1

    with log_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["event"] for r in rows] == ["SPRAY_START", "SPRAY_STOP"]
    assert rows[0]["pump"] == "ON" and rows[1]["pump"] == "OFF"
    assert rows[0]["weed_count"] == "1"


def test_พืชเป้าหมายอย่างเดียวไม่ทำให้ปั๊มทำงานเลย(image_folder):
    detector = ScriptedDetector([[crop_box()]] * 20)
    camera_cfg = CameraCfg(source="images", path=str(image_folder), loop=True, hold_frames=1)
    source = ImageFolderSource(camera_cfg, image_folder)
    controller = SprayController(
        SprayCfg(),
        RoiCfg(),
        ClassCfg(names=["strawberry", "weed"], target=["strawberry"], weed=["weed"]),
    )
    relay = MockRelay(pin=17, active_high=False, verbose=False)

    try:
        for _ in range(20):
            frame = source.read()
            decision = controller.update(detector.infer(frame), frame.shape[:2])
            relay.set(decision.spray_on)
    finally:
        relay.close()
        source.release()

    assert relay.activation_count == 0, "ห้ามพ่นยาใส่พืชเป้าหมายเด็ดขาด"
    assert relay.total_on_seconds == 0.0


def test_ปั๊มถูกปิดเสมอแม้เกิดข้อผิดพลาดกลางคัน():
    """จำลองโปรแกรมพังขณะกำลังพ่น — รีเลย์ต้องถูกปิดโดย context manager"""
    relay = MockRelay(pin=17, active_high=False, verbose=False)

    with pytest.raises(RuntimeError):
        with relay:
            relay.on()
            assert relay.is_on is True
            raise RuntimeError("จำลองข้อผิดพลาดระหว่างพ่น")

    assert relay.is_on is False, "เกิด error แล้วปั๊มต้องถูกปิดอัตโนมัติ"
    assert relay.total_on_seconds > 0
