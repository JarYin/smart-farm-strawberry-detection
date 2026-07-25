"""
ทดสอบฟังก์ชันประมวลผลกล่อง — ส่วนที่ TFLite ต้องใช้แปลงพิกัดเอง

ถ้าฟังก์ชันกลุ่มนี้ผิด กรอบที่วาดจะเพี้ยนไปจากตำแหน่งจริง และการเช็ค ROI
จะผิดตามไปด้วย ทำให้พ่นผิดจุดโดยที่ดูเผินๆ เหมือนระบบทำงานปกติ
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision_utils import (
    box_iou,
    boxes_overlap,
    class_aware_nms,
    expand_box,
    nms,
    point_in_box,
    scale_boxes_back,
    xywh2xyxy,
)


def test_แปลงกล่องจากจุดกึ่งกลางเป็นมุม():
    boxes = np.array([[100.0, 200.0, 40.0, 60.0]], dtype=np.float32)
    result = xywh2xyxy(boxes)
    assert result.tolist() == [[80.0, 170.0, 120.0, 230.0]]


def test_แปลงกล่องว่างไม่พัง():
    assert xywh2xyxy(np.zeros((0, 4), dtype=np.float32)).shape == (0, 4)
    assert scale_boxes_back(np.zeros((0, 4), dtype=np.float32), 1.0, (0, 0), (480, 640)).shape == (0, 4)


def test_ค่า_iou_ของกล่องเดียวกันเท่ากับหนึ่ง():
    box = np.array([10.0, 10.0, 50.0, 50.0])
    assert box_iou(box, box.reshape(1, 4))[0] == pytest.approx(1.0)


def test_ค่า_iou_ของกล่องที่ไม่ทับกันเท่ากับศูนย์():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert box_iou(a, b)[0] == pytest.approx(0.0)


def test_ค่า_iou_ของกล่องที่ทับกันครึ่งหนึ่ง():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    # พื้นที่ทับ 50, พื้นที่รวม 150 -> 1/3
    assert box_iou(a, b)[0] == pytest.approx(1 / 3, abs=1e-4)


def test_nms_ตัดกล่องซ้ำที่ทับกันมาก():
    boxes = np.array(
        [
            [10.0, 10.0, 100.0, 100.0],   # คะแนนสูงสุด
            [12.0, 12.0, 102.0, 102.0],   # ทับกันเกือบสนิท ต้องถูกตัด
            [300.0, 300.0, 400.0, 400.0],  # อยู่คนละที่ ต้องเก็บไว้
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.45)
    assert keep == [0, 2]


def test_nms_เก็บทุกกล่องเมื่อไม่มีอันไหนทับกัน():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]], dtype=np.float32)
    scores = np.array([0.5, 0.9], dtype=np.float32)
    assert sorted(nms(boxes, scores, 0.45)) == [0, 1]


def test_nms_แยกคลาสไม่ตัดกล่องต่างคลาสที่ทับกัน():
    """สตรอว์เบอร์รีกับวัชพืชอาจอยู่ตำแหน่งใกล้กันมาก ต้องไม่ตัดทิ้งกันเอง"""
    boxes = np.array([[10.0, 10.0, 100.0, 100.0], [11.0, 11.0, 101.0, 101.0]], dtype=np.float32)
    scores = np.array([0.9, 0.85], dtype=np.float32)

    same_class = class_aware_nms(boxes, scores, np.array([0, 0]), 0.45)
    assert len(same_class) == 1, "คลาสเดียวกันและทับกัน ต้องเหลือกล่องเดียว"

    different_class = class_aware_nms(boxes, scores, np.array([0, 1]), 0.45)
    assert len(different_class) == 2, "คนละคลาส ต้องเก็บไว้ทั้งคู่"


def test_nms_จำกัดจำนวนกล่องสูงสุด():
    boxes = np.array([[i * 100.0, 0.0, i * 100.0 + 50, 50.0] for i in range(10)], dtype=np.float32)
    scores = np.linspace(0.5, 0.95, 10).astype(np.float32)
    keep = class_aware_nms(boxes, scores, np.zeros(10, dtype=int), 0.45, max_det=3)
    assert len(keep) == 3


def test_แปลงพิกัดกลับจากภาพ_letterbox():
    """ภาพ 640x480 ย่อเป็น 320x320 -> ratio 0.5, ขอบบน-ล่างข้างละ 40 พิกเซล"""
    ratio = 320 / 640          # = 0.5
    pad = (0, 40)              # 480*0.5 = 240 -> เหลือที่ว่าง 80 แบ่งบนล่างข้างละ 40
    boxes = np.array([[50.0, 60.0, 150.0, 160.0]], dtype=np.float32)

    result = scale_boxes_back(boxes, ratio, pad, (480, 640))

    # (50-0)/0.5 = 100, (60-40)/0.5 = 40, (150-0)/0.5 = 300, (160-40)/0.5 = 240
    assert result[0].tolist() == pytest.approx([100.0, 40.0, 300.0, 240.0])


def test_แปลงพิกัดกลับแล้วไม่หลุดขอบภาพ():
    boxes = np.array([[-50.0, -50.0, 5000.0, 5000.0]], dtype=np.float32)
    result = scale_boxes_back(boxes, 0.5, (0, 0), (480, 640))
    x1, y1, x2, y2 = result[0]
    assert 0 <= x1 <= 640 and 0 <= x2 <= 640
    assert 0 <= y1 <= 480 and 0 <= y2 <= 480


def test_ขยายกล่องด้วยระยะกันชน():
    result = expand_box((100, 100, 200, 200), 20)
    assert result.tolist() == [80.0, 80.0, 220.0, 220.0]


def test_การซ้อนทับของกล่อง():
    assert boxes_overlap((0, 0, 10, 10), (5, 5, 15, 15)) is True
    assert boxes_overlap((0, 0, 10, 10), (20, 20, 30, 30)) is False
    # แตะขอบพอดีไม่นับว่าซ้อนทับ
    assert boxes_overlap((0, 0, 10, 10), (10, 0, 20, 10)) is False


def test_จุดอยู่ในกล่อง():
    assert point_in_box(5, 5, (0, 0, 10, 10)) is True
    assert point_in_box(15, 5, (0, 0, 10, 10)) is False
    assert point_in_box(0, 0, (0, 0, 10, 10)) is True  # อยู่บนขอบพอดี นับว่าอยู่ใน
