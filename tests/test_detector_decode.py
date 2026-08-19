"""
ทดสอบการถอดผลลัพธ์ดิบจากโมเดล TFLite

นี่คือส่วนที่พังง่ายที่สุดของโค้ดฝั่ง Raspberry Pi เพราะ ultralytics
ไม่ได้จัดการให้ เราต้องแกะเอง และผลลัพธ์แต่ละเวอร์ชันสลับแกน/สลับหน่วยกันไปมา

ทดสอบด้วยตัวรันจำลอง (ไม่ต้องมีไฟล์ .tflite และไม่ต้องติดตั้ง tflite-runtime)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector import DetectorError, TFLiteDetector

CLASS_NAMES = ["strawberry", "weed"]
IMGSZ = 320


class FakeInterpreter:
    """ตัวรันจำลองที่เลียนแบบหน้าตา API ของ tflite Interpreter"""

    def __init__(self, output: np.ndarray, input_dtype=np.float32, quantization=(0.0, 0)):
        self._output = output
        self._input_dtype = input_dtype
        self._quantization = quantization
        self.received = None

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [
            {
                "index": 0,
                "shape": np.array([1, IMGSZ, IMGSZ, 3]),
                "dtype": self._input_dtype,
                "quantization": self._quantization,
            }
        ]

    def get_output_details(self):
        return [
            {
                "index": 1,
                "shape": np.array(self._output.shape),
                "dtype": self._output.dtype.type,
                "quantization": (0.0, 0),
            }
        ]

    def set_tensor(self, index, tensor):
        self.received = tensor

    def invoke(self):
        pass

    def get_tensor(self, index):
        return self._output


def build_detector(output: np.ndarray, **kwargs) -> TFLiteDetector:
    """สร้าง TFLiteDetector โดยข้ามการโหลดไฟล์จริง"""
    detector = object.__new__(TFLiteDetector)
    interpreter = FakeInterpreter(output, **kwargs)

    detector.class_names = list(CLASS_NAMES)
    detector.conf = 0.45
    detector.iou = 0.45
    detector.max_det = 20
    detector.last_timing_ms = {}
    detector.interpreter = interpreter
    detector.input_detail = interpreter.get_input_details()[0]
    detector.output_detail = interpreter.get_output_details()[0]
    detector.imgsz = (IMGSZ, IMGSZ)
    detector.input_dtype = detector.input_detail["dtype"]
    detector.input_quant = detector.input_detail["quantization"]
    detector.output_dtype = detector.output_detail["dtype"]
    detector.output_quant = detector.output_detail["quantization"]
    detector.is_quantized_input = detector.input_dtype in (np.int8, np.uint8)
    detector.input_layout = "nhwc"
    return detector


def make_raw_output(entries, normalized=True, transposed=True) -> np.ndarray:
    """สร้างผลลัพธ์ดิบแบบที่ YOLOv8 TFLite ให้มา

    entries : รายการของ (cx, cy, w, h, [คะแนนแต่ละคลาส]) หน่วยเป็นพิกเซลของภาพ 320x320
    transposed : True = รูปแบบ (1, 4+nc, anchors) ตามที่ ultralytics ให้มาจริง
    """
    num_classes = len(CLASS_NAMES)
    anchors = max(len(entries), 1)
    data = np.zeros((4 + num_classes, anchors), dtype=np.float32)

    for index, (cx, cy, w, h, scores) in enumerate(entries):
        divisor = IMGSZ if normalized else 1.0
        data[0, index] = cx / divisor
        data[1, index] = cy / divisor
        data[2, index] = w / divisor
        data[3, index] = h / divisor
        for class_index, score in enumerate(scores):
            data[4 + class_index, index] = score

    if not transposed:
        data = data.T
    return data[np.newaxis, ...]


def test_ถอดผลลัพธ์พิกัดแบบ_normalize():
    raw = make_raw_output([(160, 160, 80, 80, [0.05, 0.92])])
    detector = build_detector(raw)

    detections = detector._decode(raw, ratio=1.0, pad=(0, 0), original_shape=(320, 320))

    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "weed"
    assert det.confidence == pytest.approx(0.92, abs=1e-4)
    assert det.box == pytest.approx((120.0, 120.0, 200.0, 200.0))


def test_ถอดผลลัพธ์พิกัดแบบพิกเซล():
    raw = make_raw_output([(160, 160, 80, 80, [0.88, 0.02])], normalized=False)
    detector = build_detector(raw)

    detections = detector._decode(raw, ratio=1.0, pad=(0, 0), original_shape=(320, 320))

    assert len(detections) == 1
    assert detections[0].class_name == "strawberry"
    assert detections[0].box == pytest.approx((120.0, 120.0, 200.0, 200.0))


def test_รองรับผลลัพธ์ที่สลับแกน():
    raw = make_raw_output([(160, 160, 80, 80, [0.05, 0.92])], transposed=False)
    detector = build_detector(raw)

    detections = detector._decode(raw, ratio=1.0, pad=(0, 0), original_shape=(320, 320))
    assert len(detections) == 1
    assert detections[0].class_name == "weed"


def test_ตัดผลลัพธ์ที่ความเชื่อมั่นต่ำกว่าเกณฑ์():
    raw = make_raw_output(
        [
            (100, 100, 40, 40, [0.10, 0.20]),  # ต่ำกว่า 0.45 -> ตัดทิ้ง
            (200, 200, 40, 40, [0.02, 0.80]),  # ผ่าน
        ]
    )
    detector = build_detector(raw)

    detections = detector._decode(raw, ratio=1.0, pad=(0, 0), original_shape=(320, 320))
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.80, abs=1e-4)


def test_ไม่พบวัตถุเลยคืนรายการว่าง():
    raw = make_raw_output([(100, 100, 40, 40, [0.01, 0.02])])
    detector = build_detector(raw)
    assert detector._decode(raw, ratio=1.0, pad=(0, 0), original_shape=(320, 320)) == []


def test_แปลงพิกัดกลับไปยังภาพต้นฉบับที่ไม่ใช่สี่เหลี่ยมจัตุรัส():
    """ภาพจริง 640x480 -> letterbox เป็น 320x320 ด้วย ratio 0.5 และขอบบนล่าง 40 พิกเซล"""
    raw = make_raw_output([(160, 160, 100, 100, [0.02, 0.90])])
    detector = build_detector(raw)

    detections = detector._decode(raw, ratio=0.5, pad=(0, 40), original_shape=(480, 640))

    det = detections[0]
    # กล่องในภาพ letterbox = (110,110)-(210,210)
    # แปลงกลับ: x -> /0.5 = 220,420 ; y -> (110-40)/0.5 = 140, (210-40)/0.5 = 340
    assert det.box == pytest.approx((220.0, 140.0, 420.0, 340.0))


def test_ตัดกล่องซ้ำด้วย_nms():
    raw = make_raw_output(
        [
            (160, 160, 80, 80, [0.02, 0.90]),
            (162, 162, 80, 80, [0.02, 0.85]),  # ทับกันเกือบสนิท คลาสเดียวกัน
        ]
    )
    detector = build_detector(raw)

    detections = detector._decode(raw, ratio=1.0, pad=(0, 0), original_shape=(320, 320))
    assert len(detections) == 1


def test_จำนวนคลาสไม่ตรงกับโมเดลต้องฟ้อง():
    raw = make_raw_output([(160, 160, 80, 80, [0.02, 0.90])])
    detector = build_detector(raw)
    detector.class_names = ["a", "b", "c", "d"]  # จงใจตั้งผิด

    with pytest.raises(DetectorError, match="classes.names"):
        detector._decode(raw, ratio=1.0, pad=(0, 0), original_shape=(320, 320))


def test_ถอดผลลัพธ์แบบ_quantized_int8():
    """โมเดล int8 ให้ค่าเป็นจำนวนเต็ม ต้อง dequantize ก่อนใช้"""
    scale, zero_point = 0.004, -128
    float_output = make_raw_output([(160, 160, 80, 80, [0.05, 0.92])])
    quantized = np.clip(np.round(float_output / scale + zero_point), -128, 127).astype(np.int8)

    detector = build_detector(quantized)
    detector.output_quant = (scale, zero_point)

    detections = detector._decode(quantized, ratio=1.0, pad=(0, 0), original_shape=(320, 320))

    assert len(detections) == 1
    assert detections[0].class_name == "weed"
    assert detections[0].confidence == pytest.approx(0.92, abs=0.01)
    assert detections[0].box == pytest.approx((120.0, 120.0, 200.0, 200.0), abs=2.0)


# ---------------------------------------------------------------------------
# ทดสอบ _preprocess() รองรับทั้ง layout NHWC และ NCHW ของ input tensor
#
# พบบั๊กจริงตอน deploy บน Raspberry Pi 2026-08-08: best_float32.tflite ที่ export
# มา มี input shape (1, 3, 320, 320) = NCHW แต่ _preprocess() เดิมสร้าง tensor
# แบบ NHWC เสมอ ทำให้ set_tensor() พังด้วย "Dimension mismatch" ทุกเฟรม
# ---------------------------------------------------------------------------


def test_เตรียมภาพแบบ_nhwc_ได้รูปทรงถูกต้อง():
    raw = make_raw_output([(160, 160, 80, 80, [0.05, 0.92])])
    detector = build_detector(raw)
    detector.input_layout = "nhwc"

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, ratio, pad = detector._preprocess(frame)

    assert tensor.shape == (1, IMGSZ, IMGSZ, 3)


def test_เตรียมภาพแบบ_nchw_สลับแกนให้ถูกต้อง():
    raw = make_raw_output([(160, 160, 80, 80, [0.05, 0.92])])
    detector = build_detector(raw)
    detector.input_layout = "nchw"

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, ratio, pad = detector._preprocess(frame)

    assert tensor.shape == (1, 3, IMGSZ, IMGSZ)
