"""
ทดสอบตรรกะการตัดสินใจพ่น — ส่วนที่สำคัญที่สุดของระบบ

ทดสอบด้วยนาฬิกาปลอม จึงจำลองเหตุการณ์ "พ่นค้าง 6 วินาที" ได้ในเสี้ยววินาที
โดยไม่ต้องมีกล้อง โมเดล หรือฮาร์ดแวร์ใดๆ

รันด้วย:  pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ClassCfg, RoiCfg, SprayCfg
from src.detection import Detection
from src.logic import STATE_COOLDOWN, STATE_IDLE, STATE_SPRAYING, SprayController

FRAME_SHAPE = (480, 640)  # (สูง, กว้าง)


class FakeClock:
    """นาฬิกาปลอมที่เราสั่งให้เดินหน้าได้เอง"""

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_detection(name: str, conf: float = 0.9, box=(280, 300, 360, 400)) -> Detection:
    """สร้างวัตถุตรวจพบ 1 ชิ้น ตำแหน่งเริ่มต้นอยู่กลาง-ล่างของเฟรม (อยู่ใน ROI)"""
    return Detection(
        class_id=0,
        class_name=name,
        confidence=conf,
        x1=box[0],
        y1=box[1],
        x2=box[2],
        y2=box[3],
    )


@pytest.fixture
def classes() -> ClassCfg:
    return ClassCfg(
        names=["strawberry", "weed"],
        target=["strawberry"],
        weed=["weed"],
        unknown_policy="ignore",
        per_class_conf={"weed": 0.55},
    )


@pytest.fixture
def roi() -> RoiCfg:
    return RoiCfg(enabled=True, x1=0.15, y1=0.45, x2=0.85, y2=1.0, require="center")


@pytest.fixture
def spray() -> SprayCfg:
    return SprayCfg(
        confirm_frames=3,
        release_frames=2,
        min_on_seconds=0.5,
        max_on_seconds=3.0,
        cooldown_seconds=1.0,
        protect_crop=True,
        protect_margin_px=20,
    )


@pytest.fixture
def controller(spray, roi, classes):
    clock = FakeClock()
    ctrl = SprayController(spray, roi, classes, base_conf=0.45, clock=clock)
    ctrl.test_clock = clock  # ผูกนาฬิกาไว้กับตัวควบคุมเพื่อให้เทสต์เรียกใช้ง่าย
    return ctrl


# ---------------------------------------------------------------------------
# การจำแนกวัตถุ
# ---------------------------------------------------------------------------


def test_พืชเป้าหมายไม่ทำให้พ่น(controller):
    for _ in range(10):
        decision = controller.update([make_detection("strawberry")], FRAME_SHAPE)
    assert decision.spray_on is False
    assert decision.state == STATE_IDLE
    assert len(decision.analysis.targets) == 1


def test_วัชพืชความเชื่อมั่นต่ำถูกตัดทิ้ง(controller):
    # per_class_conf ของ weed = 0.55 ดังนั้น 0.50 ต้องไม่ผ่าน
    for _ in range(10):
        decision = controller.update([make_detection("weed", conf=0.50)], FRAME_SHAPE)
    assert decision.spray_on is False
    assert decision.analysis.actionable_weeds == []
    assert len(decision.analysis.rejected) == 1


def test_วัชพืชนอกขอบเขตหัวฉีดไม่ถูกพ่น(controller):
    # กล่องอยู่ครึ่งบนของภาพ (y < 0.45*480 = 216) จึงอยู่นอก ROI
    outside = make_detection("weed", box=(280, 20, 360, 100))
    for _ in range(10):
        decision = controller.update([outside], FRAME_SHAPE)
    assert decision.spray_on is False
    assert len(decision.analysis.suppressed_weeds) == 1
    assert "ROI" in decision.analysis.suppressed_weeds[0][1]


def test_วัชพืชชิดพืชเป้าหมายถูกระงับเพื่อป้องกันพืชหลัก(controller):
    crop = make_detection("strawberry", box=(300, 300, 400, 400))
    weed = make_detection("weed", box=(405, 300, 460, 400))  # ห่างแค่ 5 พิกเซล < margin 20
    for _ in range(10):
        decision = controller.update([crop, weed], FRAME_SHAPE)
    assert decision.spray_on is False
    assert len(decision.analysis.suppressed_weeds) == 1
    assert "พืชเป้าหมาย" in decision.analysis.suppressed_weeds[0][1]


def test_วัชพืชห่างจากพืชเป้าหมายพอสมควรพ่นได้(controller):
    crop = make_detection("strawberry", box=(100, 300, 200, 400))
    weed = make_detection("weed", box=(400, 300, 480, 400))
    for _ in range(3):
        decision = controller.update([crop, weed], FRAME_SHAPE)
    assert decision.spray_on is True


def test_คลาสที่ไม่รู้จักถูกมองข้ามตามนโยบาย(spray, roi):
    classes = ClassCfg(names=["strawberry", "weed", "stone"], target=["strawberry"], weed=["weed"])
    controller = SprayController(spray, roi, classes, base_conf=0.45, clock=FakeClock())
    for _ in range(10):
        decision = controller.update([make_detection("stone")], FRAME_SHAPE)
    assert decision.spray_on is False
    assert len(decision.analysis.rejected) == 1


def test_นโยบายถือคลาสแปลกเป็นวัชพืช(spray, roi):
    classes = ClassCfg(
        names=["strawberry", "weed", "stone"],
        target=["strawberry"],
        weed=["weed"],
        unknown_policy="weed",
    )
    controller = SprayController(spray, roi, classes, base_conf=0.45, clock=FakeClock())
    for _ in range(3):
        decision = controller.update([make_detection("stone")], FRAME_SHAPE)
    assert decision.spray_on is True


# ---------------------------------------------------------------------------
# เครื่องสถานะ
# ---------------------------------------------------------------------------


def test_ต้องเจอวัชพืชครบตามจำนวนเฟรมก่อนจึงเริ่มพ่น(controller):
    weed = make_detection("weed")

    decision = controller.update([weed], FRAME_SHAPE)
    assert decision.spray_on is False and decision.hit_streak == 1

    decision = controller.update([weed], FRAME_SHAPE)
    assert decision.spray_on is False and decision.hit_streak == 2

    decision = controller.update([weed], FRAME_SHAPE)  # ครบ 3 เฟรม
    assert decision.spray_on is True
    assert decision.state == STATE_SPRAYING
    assert decision.changed is True


def test_เจอวัชพืชสลับหายทำให้ตัวนับเริ่มใหม่(controller):
    weed = make_detection("weed")
    controller.update([weed], FRAME_SHAPE)
    controller.update([weed], FRAME_SHAPE)
    decision = controller.update([], FRAME_SHAPE)  # ขาดไป 1 เฟรม
    assert decision.hit_streak == 0

    decision = controller.update([weed], FRAME_SHAPE)
    assert decision.spray_on is False
    assert decision.hit_streak == 1


def test_พ่นครบเวลาขั้นต่ำก่อนจึงหยุดได้(controller):
    clock = controller.test_clock
    weed = make_detection("weed")

    for _ in range(3):
        controller.update([weed], FRAME_SHAPE)
    assert controller.state == STATE_SPRAYING

    # วัชพืชหายไปแล้ว 2 เฟรม (ครบ release_frames) แต่ยังไม่ถึง min_on_seconds
    clock.advance(0.1)
    controller.update([], FRAME_SHAPE)
    clock.advance(0.1)
    decision = controller.update([], FRAME_SHAPE)
    assert decision.spray_on is True, "ยังไม่ครบเวลาพ่นขั้นต่ำ ต้องพ่นต่อ"

    clock.advance(0.5)  # รวมแล้วเกิน 0.5 วินาที
    decision = controller.update([], FRAME_SHAPE)
    assert decision.spray_on is False
    assert decision.state == STATE_COOLDOWN


def test_ระบบตัดการพ่นเมื่อพ่นนานเกินกำหนด(controller):
    """กรณีเลวร้ายที่สุด: AI เห็นวัชพืชตลอดเวลาจนปั๊มเปิดค้าง ต้องถูกตัด"""
    clock = controller.test_clock
    weed = make_detection("weed")

    for _ in range(3):
        controller.update([weed], FRAME_SHAPE)
    assert controller.state == STATE_SPRAYING

    clock.advance(3.5)  # เกิน max_on_seconds = 3.0
    decision = controller.update([weed], FRAME_SHAPE)

    assert decision.spray_on is False
    assert decision.state == STATE_COOLDOWN
    assert controller.cutoff_events == 1
    assert "กันปั๊มค้าง" in decision.reason


def test_ต้องพักครบเวลาก่อนพ่นรอบถัดไป(controller):
    clock = controller.test_clock
    weed = make_detection("weed")

    for _ in range(3):
        controller.update([weed], FRAME_SHAPE)
    clock.advance(0.6)
    controller.update([], FRAME_SHAPE)
    controller.update([], FRAME_SHAPE)
    assert controller.state == STATE_COOLDOWN

    # ยังอยู่ในช่วงพัก แม้เจอวัชพืชครบ 3 เฟรมก็ยังไม่พ่น
    clock.advance(0.3)
    for _ in range(3):
        decision = controller.update([weed], FRAME_SHAPE)
    assert decision.spray_on is False
    assert decision.state == STATE_COOLDOWN

    # พ้นช่วงพักแล้ว พ่นได้อีกครั้ง
    clock.advance(1.0)
    decision = controller.update([weed], FRAME_SHAPE)
    assert decision.spray_on is True
    assert controller.spray_events == 2


def test_สั่งหยุดฉุกเฉินปิดปั๊มทันที(controller):
    weed = make_detection("weed")
    for _ in range(3):
        controller.update([weed], FRAME_SHAPE)
    assert controller.state == STATE_SPRAYING

    controller.force_stop()
    assert controller.state == STATE_COOLDOWN
    assert controller.hit_streak == 0


def test_สรุปสถิติถูกต้อง(controller):
    clock = controller.test_clock
    weed = make_detection("weed")

    for _ in range(3):
        controller.update([weed], FRAME_SHAPE)
    clock.advance(1.0)
    controller.update([], FRAME_SHAPE)
    controller.update([], FRAME_SHAPE)

    summary = controller.summary()
    assert summary["spray_events"] == 1
    assert summary["frames_processed"] == 5
    assert summary["frames_with_weed"] == 3
    assert summary["total_spray_seconds"] == pytest.approx(1.0, abs=0.01)


def test_ปิด_ROI_แล้ววัชพืชตำแหน่งไหนก็พ่น(spray, classes):
    roi = RoiCfg(enabled=False)
    controller = SprayController(spray, roi, classes, base_conf=0.45, clock=FakeClock())
    top_weed = make_detection("weed", box=(280, 20, 360, 100))
    for _ in range(3):
        decision = controller.update([top_weed], FRAME_SHAPE)
    assert decision.spray_on is True


def test_โหมด_overlap_ยอมรับกล่องที่แค่แตะขอบเขต(spray, classes):
    roi = RoiCfg(enabled=True, x1=0.15, y1=0.45, x2=0.85, y2=1.0, require="overlap")
    controller = SprayController(spray, roi, classes, base_conf=0.45, clock=FakeClock())
    # จุดกึ่งกลางอยู่ที่ y=180 (นอก ROI) แต่กล่องล้ำเข้ามาถึง y=250
    straddling = make_detection("weed", box=(280, 110, 360, 250))
    for _ in range(3):
        decision = controller.update([straddling], FRAME_SHAPE)
    assert decision.spray_on is True


# ---------------------------------------------------------------------------
# โหมดผกผัน (Inverse / Crop-vs-Non-crop) — ใช้กับโมเดลที่ไม่มีคลาสวัชพืชแยก
# เช่น โมเดลจำแนกความสุก strawberry-ripe/strawberry-unripe
# ---------------------------------------------------------------------------


@pytest.fixture
def inverse_classes() -> ClassCfg:
    return ClassCfg(
        names=["strawberry-ripe", "strawberry-unripe"],
        target=["strawberry-ripe", "strawberry-unripe"],
        weed=[],
        inverse_mode=True,
        per_class_conf={"strawberry-ripe": 0.35, "strawberry-unripe": 0.35},
    )


@pytest.fixture
def inverse_controller(spray, roi, inverse_classes):
    clock = FakeClock()
    ctrl = SprayController(spray, roi, inverse_classes, base_conf=0.35, clock=clock)
    ctrl.test_clock = clock
    return ctrl


def test_ไม่พบพืชเป้าหมายเลยทำให้พ่นทั้งเขต(inverse_controller):
    """กรณีหลักของโหมดผกผัน: ไม่เจอสตรอว์เบอร์รีในเฟรมเลย -> ถือว่าต้องพ่น"""
    for _ in range(3):
        decision = inverse_controller.update([], FRAME_SHAPE)
    assert decision.spray_on is True
    assert len(decision.analysis.actionable_weeds) == 1
    assert decision.analysis.actionable_weeds[0].class_name == "non-crop-area"


def test_พบพืชเป้าหมายในเขตพ่นทำให้งดพ่นทั้งเขต(inverse_controller):
    """แม้เจอแค่ต้นเดียวในเขตพ่น ก็ต้องงดพ่นทั้งเขตเพื่อความปลอดภัยของพืชหลัก"""
    ripe = make_detection("strawberry-ripe", conf=0.9)
    for _ in range(10):
        decision = inverse_controller.update([ripe], FRAME_SHAPE)
    assert decision.spray_on is False
    assert len(decision.analysis.suppressed_weeds) == 1
    assert "โหมดผกผัน" in decision.analysis.suppressed_weeds[0][1]
    assert len(decision.analysis.targets) == 1  # ยังเก็บไว้วาดกรอบตามปกติ


def test_พืชเป้าหมายอยู่นอกเขตพ่นไม่นับว่าพบ_จึงยังพ่น(inverse_controller):
    """สตรอว์เบอร์รีที่อยู่นอก ROI (เช่น อยู่แถวข้างๆ) ไม่ควรมีผลต่อการตัดสินใจ"""
    outside = make_detection("strawberry-unripe", conf=0.9, box=(280, 20, 360, 100))  # เหนือ ROI
    for _ in range(3):
        decision = inverse_controller.update([outside], FRAME_SHAPE)
    assert decision.spray_on is True


def test_ความเชื่อมั่นต่ำกว่าเกณฑ์ไม่นับว่าเป็นพืชเป้าหมาย(inverse_controller):
    weak = make_detection("strawberry-ripe", conf=0.10)  # ต่ำกว่า 0.35
    for _ in range(3):
        decision = inverse_controller.update([weak], FRAME_SHAPE)
    assert decision.spray_on is True, "ความเชื่อมั่นต่ำเกินไป ไม่ควรนับเป็นพืชเป้าหมาย"


def test_ทั้งสองคลาสความสุกถือเป็นพืชเป้าหมายเหมือนกัน(inverse_controller):
    unripe = make_detection("strawberry-unripe", conf=0.9)
    for _ in range(10):
        decision = inverse_controller.update([unripe], FRAME_SHAPE)
    assert decision.spray_on is False


def test_ยืนยันหลายเฟรมยังทำงานปกติในโหมดผกผัน(inverse_controller):
    """กลไกกันภาพเบลอ (confirm_frames) ต้องยังใช้ได้ ไม่ถูกโหมดผกผันข้าม"""
    decision = inverse_controller.update([], FRAME_SHAPE)
    assert decision.spray_on is False and decision.hit_streak == 1

    decision = inverse_controller.update([], FRAME_SHAPE)
    assert decision.spray_on is False and decision.hit_streak == 2

    decision = inverse_controller.update([], FRAME_SHAPE)  # ครบ confirm_frames=3
    assert decision.spray_on is True


def test_เจอพืชเป้าหมายกลางคันทำให้หยุดพ่นตามกลไกปกติ(inverse_controller):
    clock = inverse_controller.test_clock
    for _ in range(3):
        inverse_controller.update([], FRAME_SHAPE)
    assert inverse_controller.state == STATE_SPRAYING

    ripe = make_detection("strawberry-ripe", conf=0.9)
    clock.advance(0.7)  # เกิน min_on_seconds ของ fixture spray (0.5)
    inverse_controller.update([ripe], FRAME_SHAPE)
    decision = inverse_controller.update([ripe], FRAME_SHAPE)  # ครบ release_frames=2
    assert decision.spray_on is False
    assert decision.state == STATE_COOLDOWN


def test_ปิด_roi_ในโหมดผกผันใช้ทั้งเฟรมเป็นเขตพ่น(spray, inverse_classes):
    roi = RoiCfg(enabled=False)
    controller = SprayController(spray, roi, inverse_classes, base_conf=0.35, clock=FakeClock())
    # พืชเป้าหมายอยู่มุมบนซ้าย ซึ่งจะอยู่นอก ROI เดิม แต่ตอนนี้ ROI ปิดแล้วครอบทั้งเฟรม
    ripe_top_left = make_detection("strawberry-ripe", conf=0.9, box=(0, 0, 50, 50))
    for _ in range(10):
        decision = controller.update([ripe_top_left], FRAME_SHAPE)
    assert decision.spray_on is False, "ปิด ROI แล้วทั้งเฟรมคือเขตพ่น ต้องนับพืชเป้าหมายมุมไหนก็ได้"


def test_สรุปสถิติทำงานถูกต้องในโหมดผกผัน(inverse_controller):
    for _ in range(3):
        inverse_controller.update([], FRAME_SHAPE)
    summary = inverse_controller.summary()
    assert summary["spray_events"] == 1
    assert summary["frames_with_weed"] == 3
