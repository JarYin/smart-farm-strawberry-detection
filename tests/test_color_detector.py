"""
ทดสอบตัวตรวจจับด้วยสี (ColorDetector) — "ตา" ของ prototype-v1

สร้างภาพสังเคราะห์ที่รู้คำตอบล่วงหน้า (แผ่นสีทึบตำแหน่งที่กำหนดเอง) แล้วตรวจว่า
ตัวตรวจจับคืนกรอบตรงตำแหน่งนั้นจริง ทดสอบได้ทั้งหมดโดยไม่ต้องมีกล้อง

รันด้วย:  pytest tests/test_color_detector.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2", reason="ต้องติดตั้ง opencv ก่อนจึงจะรันเทสต์นี้ได้")

from src.config import ColorCfg, ConfigError
from src.detector import ColorDetector

FRAME_H, FRAME_W = 480, 640

# สีในระบบ BGR ของ OpenCV (ไม่ใช่ RGB!)
BGR_RED = (40, 40, 220)
BGR_GREEN = (60, 200, 60)
BGR_BACKGROUND = (30, 30, 30)


def blank_frame() -> np.ndarray:
    """ฉากพื้นหลังมืดสม่ำเสมอ — ไม่มีสีอะไรให้จับ"""
    return np.full((FRAME_H, FRAME_W, 3), BGR_BACKGROUND, dtype=np.uint8)


def paint(frame: np.ndarray, box, color) -> np.ndarray:
    """ระบายสี่เหลี่ยมทึบลงบนภาพ (box = x1, y1, x2, y2)"""
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = color
    return frame


def bgr_of_hsv(hue: int, sat: int = 255, val: int = 255) -> tuple:
    """แปลงค่า HSV ที่กำหนดเองเป็น BGR — ใช้สร้างเฉดสีที่ต้องการเป๊ะๆ"""
    pixel = np.array([[[hue, sat, val]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0][0]
    return (int(b), int(g), int(r))


@pytest.fixture
def detector() -> ColorDetector:
    return ColorDetector(ColorCfg(), conf=0.45, max_det=20)


# ---------------------------------------------------------------------------
# การตรวจจับพื้นฐาน
# ---------------------------------------------------------------------------


def test_ภาพว่างเปล่าไม่พบอะไรเลย(detector):
    """กรณีสำคัญที่สุดของ prototype: ฉากว่าง = ไม่มี detection (แล้วตรรกะจะสั่งยิง)"""
    assert detector.infer(blank_frame()) == []


def test_แผ่นสีแดงถูกตรวจพบพร้อมกรอบตรงตำแหน่ง(detector):
    frame = paint(blank_frame(), (200, 150, 340, 290), BGR_RED)
    detections = detector.infer(frame)

    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "red-object"
    assert det.class_id == 0
    # ยอมคลาดเคลื่อนได้ไม่กี่พิกเซล เพราะผ่าน GaussianBlur + morphology มาก่อน
    assert det.x1 == pytest.approx(200, abs=4)
    assert det.y1 == pytest.approx(150, abs=4)
    assert det.x2 == pytest.approx(340, abs=4)
    assert det.y2 == pytest.approx(290, abs=4)
    assert det.confidence > 0.9, "แผ่นสี่เหลี่ยมทึบต้องได้ fill ratio เกือบเต็ม"


def test_แผ่นสีเขียวถูกตรวจพบเป็นคนละคลาสกับสีแดง(detector):
    frame = paint(blank_frame(), (200, 150, 340, 290), BGR_GREEN)
    detections = detector.infer(frame)

    assert len(detections) == 1
    assert detections[0].class_name == "green-object"
    assert detections[0].class_id == 1


def test_เจอทั้งสองสีในเฟรมเดียวกันแยกกันได้(detector):
    frame = blank_frame()
    paint(frame, (60, 150, 200, 290), BGR_RED)
    paint(frame, (400, 150, 540, 290), BGR_GREEN)

    names = sorted(det.class_name for det in detector.infer(frame))
    assert names == ["green-object", "red-object"]


def test_สีแดงเฉดที่คร่อมปลายวงล้อสีก็ยังจับได้(detector):
    """กับดักคลาสสิก: สีแดงอยู่ทั้งที่ hue ใกล้ 0 และใกล้ 179

    ถ้าโค้ดใส่ช่วง hue ไว้ช่วงเดียว เฉดแดงอมชมพู (hue ~175) จะหลุดหายไปเงียบๆ
    """
    frame = paint(blank_frame(), (200, 150, 340, 290), bgr_of_hsv(175))
    detections = detector.infer(frame)

    assert len(detections) == 1
    assert detections[0].class_name == "red-object"


def test_สีเทาและสีขาวไม่ถูกนับว่าเป็นสีใดๆ(detector):
    """S ขั้นต่ำมีไว้กันสีจืด — กระดาษขาว/ผนังเทาต้องไม่กลายเป็น detection"""
    frame = blank_frame()
    paint(frame, (60, 150, 200, 290), (240, 240, 240))   # ขาว
    paint(frame, (400, 150, 540, 290), (128, 128, 128))  # เทากลาง

    assert detector.infer(frame) == []


def test_สีแดงในที่มืดสนิทไม่ถูกนับ(detector):
    """V ขั้นต่ำมีไว้กันบริเวณมืดที่แยกเนื้อสีไม่ออก"""
    frame = paint(blank_frame(), (200, 150, 340, 290), bgr_of_hsv(0, sat=255, val=30))
    assert detector.infer(frame) == []


# ---------------------------------------------------------------------------
# ตัวกรองต่างๆ
# ---------------------------------------------------------------------------


def test_จุดสีเล็กจิ๋วถูกตัดทิ้งด้วยเกณฑ์ขนาด(detector):
    """min_area_frac = 0.004 -> บนเฟรม 640x480 ต้องใหญ่กว่าราว 35x35 พิกเซล"""
    frame = paint(blank_frame(), (300, 200, 315, 215), BGR_RED)  # 15x15 พิกเซล
    assert detector.infer(frame) == []


def test_สีที่ท่วมทั้งเฟรมถูกตัดทิ้ง():
    """max_area_frac กันกรณีแสงไฟทั้งฉากเป็นสีเดียว จนได้กล่องเต็มจอที่ไร้ความหมาย"""
    detector = ColorDetector(ColorCfg(max_area_frac=0.5), conf=0.45, max_det=20)
    frame = np.full((FRAME_H, FRAME_W, 3), BGR_RED, dtype=np.uint8)
    assert detector.infer(frame) == []


def test_ริ้วสีบางๆถูกตัดด้วยเกณฑ์_fill_ratio(detector):
    """เส้นทแยงบางๆ มีกรอบใหญ่แต่เนื้อสีน้อย -> fill ratio ต่ำ ต้องไม่ผ่าน

    เป็นกลไกกันเงาสะท้อน/ขอบวัตถุที่บังเอิญเป็นสีเดียวกันแต่ไม่ใช่ "แผ่นสี" จริง
    """
    frame = blank_frame()
    cv2.line(frame, (220, 140), (420, 340), BGR_RED, thickness=9)

    detections = detector.infer(frame)
    assert detections == [], "เส้นบางต้องถูกตัดด้วย fill ratio"


def test_ลดเกณฑ์_fill_ratio_แล้วริ้วสีเดิมผ่านได้():
    """ยืนยันว่าที่ตัดทิ้งไปในเทสต์ก่อนหน้าคือ fill ratio จริงๆ ไม่ใช่ตัวกรองอื่น"""
    detector = ColorDetector(ColorCfg(), conf=0.02, max_det=20)
    frame = blank_frame()
    cv2.line(frame, (220, 140), (420, 340), BGR_RED, thickness=9)

    detections = detector.infer(frame)
    assert len(detections) == 1
    assert detections[0].confidence < 0.45


def test_จำกัดจำนวนวัตถุตาม_max_det_โดยเก็บก้อนใหญ่สุดไว้ก่อน():
    detector = ColorDetector(ColorCfg(), conf=0.45, max_det=2)
    frame = blank_frame()
    # ก้อนใหญ่ 2 ก้อน และก้อนเล็กอีก 3 ก้อน (ทุกก้อนใหญ่พอผ่านเกณฑ์ขนาด)
    paint(frame, (20, 20, 140, 140), BGR_RED)
    paint(frame, (200, 20, 320, 140), BGR_RED)
    for index in range(3):
        left = 20 + index * 80
        paint(frame, (left, 300, left + 45, 345), BGR_GREEN)

    detections = detector.infer(frame)
    assert len(detections) == 2
    assert all(det.class_name == "red-object" for det in detections), "ต้องเก็บก้อนใหญ่สุดไว้"


def test_ปิดการเบลอและ_morphology_แล้วยังทำงานได้():
    """ksize = 0 หมายถึงข้ามขั้นตอนนั้นไป (ใช้เมื่ออยากได้ FPS สูงสุดบน Pi)"""
    detector = ColorDetector(ColorCfg(blur_ksize=0, morph_ksize=0), conf=0.45, max_det=20)
    frame = paint(blank_frame(), (200, 150, 340, 290), BGR_RED)

    detections = detector.infer(frame)
    assert len(detections) == 1
    assert detections[0].class_name == "red-object"


def test_เปลี่ยนชื่อคลาสในไฟล์ตั้งค่าแล้วผลลัพธ์เปลี่ยนตาม():
    """ชื่อคลาสมาจาก config เสมอ เพราะ logic จับคู่บทบาทด้วย "ชื่อ" ไม่ใช่ class id"""
    detector = ColorDetector(
        ColorCfg(red_name="strawberry", green_name="weed"), conf=0.45, max_det=20
    )
    frame = paint(blank_frame(), (200, 150, 340, 290), BGR_GREEN)

    assert detector.infer(frame)[0].class_name == "weed"
    assert detector.class_names == ["strawberry", "weed"]


def test_บันทึกเวลาที่ใช้ในแต่ละขั้นตอน(detector):
    detector.infer(paint(blank_frame(), (200, 150, 340, 290), BGR_RED))
    timing = detector.last_timing_ms
    assert set(timing) == {"preprocess", "inference", "postprocess"}
    assert all(value >= 0.0 for value in timing.values())


# ---------------------------------------------------------------------------
# การตรวจสอบไฟล์ตั้งค่าของหมวด color
# ---------------------------------------------------------------------------


def test_ค่าเริ่มต้นของหมวด_color_ผ่านการตรวจสอบ():
    ColorCfg().validate()  # ต้องไม่ raise


def test_ค่าเริ่มต้นให้สีแดงสองช่วงและสีเขียวหนึ่งช่วง():
    bands = ColorCfg().bands()
    assert [band.class_id for band in bands] == [0, 1]
    assert len(bands[0].hue_ranges) == 2, "สีแดงต้องมีสองช่วงเสมอ (คร่อม hue 0)"
    assert len(bands[1].hue_ranges) == 1


def test_ค่า_hue_เกินขอบเขตของ_opencv_ถูกปฏิเสธ():
    """ดักคนที่เผลอใส่ค่าจากตำรา (0-359) แทนค่าของ OpenCV (0-179)"""
    with pytest.raises(ConfigError, match="0-179"):
        ColorCfg(green_hue_ranges=[[70, 200]]).validate()


def test_ช่วง_hue_ที่กลับด้านถูกปฏิเสธ():
    with pytest.raises(ConfigError):
        ColorCfg(green_hue_ranges=[[85, 35]]).validate()


def test_รูปแบบช่วงสีที่ผิดถูกปฏิเสธ():
    with pytest.raises(ConfigError, match="hue_min, hue_max"):
        ColorCfg(green_hue_ranges=[[35, 85, 100]]).validate()


def test_ช่วงสีว่างเปล่าถูกปฏิเสธ():
    with pytest.raises(ConfigError):
        ColorCfg(red_hue_ranges=[]).validate()


def test_ขนาด_kernel_เลขคู่ถูกปฏิเสธ():
    """OpenCV บังคับว่า kernel ของ GaussianBlur ต้องเป็นเลขคี่"""
    with pytest.raises(ConfigError, match="เลขคี่"):
        ColorCfg(blur_ksize=4).validate()


def test_ปิด_kernel_ด้วยเลขศูนย์ได้():
    ColorCfg(blur_ksize=0, morph_ksize=0).validate()  # ต้องไม่ raise


def test_เกณฑ์ขนาดที่กลับด้านถูกปฏิเสธ():
    with pytest.raises(ConfigError):
        ColorCfg(min_area_frac=0.5, max_area_frac=0.1).validate()


def test_ชื่อสีซ้ำกันถูกปฏิเสธ():
    with pytest.raises(ConfigError, match="ไม่ซ้ำกัน"):
        ColorCfg(red_name="thing", green_name="thing").validate()
