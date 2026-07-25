"""
ทดสอบ RoboflowDetector — เอนจิ้นคลาวด์ที่เรียกโมเดลผ่าน API ของ Roboflow

ทดสอบด้วยข้อมูล JSON จำลอง (ไม่เรียก API จริง ไม่ต้องมี inference-sdk ติดตั้ง
ไม่ต้องมี API key และไม่เสียโควตาการใช้งาน) โฟกัสที่การถอดผลลัพธ์ (_decode)
ซึ่งเป็นจุดที่พังง่ายที่สุด: รูปแบบ JSON ของ Roboflow อาจมีคีย์ไม่ครบ หรือ
มีค่าที่แปลงชนิดไม่ได้ปนมา
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ClassCfg, Config, ConfigError, ModelCfg, RoboflowCfg
from src.detector import DetectorError, RoboflowDetector, build_detector


def make_bare_detector(conf: float = 0.45, class_names=()) -> RoboflowDetector:
    """สร้าง RoboflowDetector โดยข้าม __init__ (ไม่ต้องมี inference-sdk หรือ API key จริง)

    ตั้งชื่อไม่ให้ชนกับ build_detector() ที่ import มาจาก src.detector — เคยพลาด
    ตั้งชื่อซ้ำมาก่อนแล้วทำให้เทสต์ที่เรียก build_detector(cfg) จริงๆ กลับไปเรียก
    ฟังก์ชันนี้แทนโดยไม่รู้ตัว (ตัวแปรระดับโมดูลถูกฟังก์ชันนี้บังไว้)
    """
    detector = object.__new__(RoboflowDetector)
    detector.class_names = list(class_names)
    detector.conf = conf
    detector.iou = 0.45
    detector.max_det = 20
    detector.last_timing_ms = {}
    detector.model_id = "strawberry-detection-msf0m/3"
    detector.timeout = 10.0
    detector._dynamic_names = len(class_names) == 0
    return detector


def test_ถอดผลลัพธ์รูปแบบมาตรฐานของ_roboflow():
    detector = make_bare_detector()
    predictions = [
        {"class": "strawberry", "confidence": 0.91, "x": 246.5, "y": 300.5, "width": 60.0, "height": 82.0}
    ]

    detections = detector._decode(predictions)

    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "strawberry"
    assert det.confidence == pytest.approx(0.91)
    # x,y คือจุดกึ่งกลาง width/height คือขนาดเต็ม -> x1 = 246.5 - 30 = 216.5
    assert det.box == pytest.approx((216.5, 259.5, 276.5, 341.5))


def test_ตัดผลลัพธ์ที่ความเชื่อมั่นต่ำกว่าเกณฑ์():
    detector = make_bare_detector(conf=0.5)
    predictions = [
        {"class": "weed", "confidence": 0.30, "x": 10, "y": 10, "width": 5, "height": 5},
        {"class": "weed", "confidence": 0.80, "x": 50, "y": 50, "width": 5, "height": 5},
    ]

    detections = detector._decode(predictions)

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.80)


def test_ไม่มีวัตถุเลยคืนรายการว่าง():
    detector = make_bare_detector()
    assert detector._decode([]) == []


def test_รายการที่ขาดคีย์ที่จำเป็นถูกข้ามไปไม่ทำให้ทั้งเฟรมพัง():
    """API อาจส่งข้อมูลผิดรูปแบบมาบางรายการ (บั๊กฝั่งเซิร์ฟเวอร์/เวอร์ชันเปลี่ยน)
    ต้องข้ามแค่รายการนั้น ไม่ใช่ทำให้ทั้งเฟรมล้มเหลว"""
    detector = make_bare_detector()
    predictions = [
        {"class": "weed", "confidence": 0.80},  # ขาด x, y, width, height
        {"confidence": "not-a-number", "x": 1, "y": 1, "width": 1, "height": 1},  # แปลงชนิดไม่ได้
        {"class": "strawberry", "confidence": 0.95, "x": 100, "y": 100, "width": 20, "height": 20},  # ปกติ
    ]

    detections = detector._decode(predictions)

    assert len(detections) == 1
    assert detections[0].class_name == "strawberry"


def test_ใช้_class_name_สำรองเมื่อไม่มีคีย์_class():
    detector = make_bare_detector()
    predictions = [{"class_name": "grass", "confidence": 0.6, "x": 5, "y": 5, "width": 2, "height": 2}]

    detections = detector._decode(predictions)

    assert detections[0].class_name == "grass"


def test_ไม่ระบุ_api_key_ถูกปฏิเสธก่อนเรียก_api():
    """ต้องฟ้อง error ที่ชัดเจนตั้งแต่ต้นทาง ไม่ใช่ปล่อยให้เรียก API แล้วได้ 401 มาแบบงงๆ"""
    with pytest.raises(DetectorError, match="API key"):
        RoboflowDetector(model_id="x/1", api_url="https://serverless.roboflow.com", api_key="")


def test_ข้อความ_error_ไม่รั่วค่า_api_key():
    """ทดสอบว่าข้อความ error ตอนไม่มีคีย์ไม่ได้พิมพ์ค่าคีย์ใดๆ ออกมา (เพราะไม่มีคีย์ให้พิมพ์)"""
    with pytest.raises(DetectorError) as exc_info:
        RoboflowDetector(model_id="x/1", api_url="https://serverless.roboflow.com", api_key="")
    assert "sk-" not in str(exc_info.value)  # กันเผื่อในอนาคตมีคนพลาดใส่ค่าคีย์ลงข้อความ


# ---------------------------------------------------------------------------
# ตั้งค่า (RoboflowCfg)
# ---------------------------------------------------------------------------


def test_roboflow_cfg_ค่าเริ่มต้นผ่านการตรวจสอบ():
    RoboflowCfg().validate()


def test_roboflow_cfg_model_id_ต้องมีเครื่องหมายทับ():
    with pytest.raises(ConfigError, match="model_id"):
        RoboflowCfg(model_id="no-slash-here").validate()


def test_roboflow_cfg_api_url_ห้ามว่าง():
    with pytest.raises(ConfigError, match="api_url"):
        RoboflowCfg(api_url="").validate()


def test_roboflow_cfg_timeout_ต้องมากกว่าศูนย์():
    with pytest.raises(ConfigError, match="timeout"):
        RoboflowCfg(timeout_seconds=0).validate()


def test_model_backend_roboflow_เป็นค่าที่ยอมรับได้():
    cfg = ModelCfg(backend="roboflow")
    cfg.validate()
    assert cfg.resolved_backend() == "roboflow"


# ---------------------------------------------------------------------------
# build_detector() wiring — ปลอม inference_sdk เพื่อไม่ต้องเรียก API จริง
# ---------------------------------------------------------------------------


class FakeInferenceHTTPClient:
    """ตัวปลอมแทน inference_sdk.InferenceHTTPClient จำไว้ว่าถูกสร้างด้วยอะไรบ้าง"""

    last_instance = None

    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
        FakeInferenceHTTPClient.last_instance = self

    def infer(self, image, model_id):
        return {"predictions": []}


@pytest.fixture
def fake_inference_sdk(monkeypatch):
    """ปลอมโมดูล inference_sdk ทั้งโมดูล เพื่อทดสอบ build_detector โดยไม่ติดตั้งไลบรารีจริง"""
    module = types.ModuleType("inference_sdk")
    module.InferenceHTTPClient = FakeInferenceHTTPClient
    monkeypatch.setitem(sys.modules, "inference_sdk", module)
    yield module


def test_build_detector_อ่าน_api_key_จากตัวแปรสภาพแวดล้อมที่กำหนดไว้(fake_inference_sdk, monkeypatch):
    monkeypatch.setenv("MY_CUSTOM_ROBOFLOW_KEY", "ทดสอบคีย์ลับ")

    cfg = Config(
        model=ModelCfg(backend="roboflow"),
        roboflow=RoboflowCfg(model_id="strawberry-detection-msf0m/3", api_key_env="MY_CUSTOM_ROBOFLOW_KEY"),
        classes=ClassCfg(names=["strawberry", "weed"]),
    )
    cfg.validate()

    detector = build_detector(cfg)

    assert isinstance(detector, RoboflowDetector)
    assert FakeInferenceHTTPClient.last_instance.api_key == "ทดสอบคีย์ลับ"
    assert FakeInferenceHTTPClient.last_instance.api_url == cfg.roboflow.api_url
    assert detector.model_id == "strawberry-detection-msf0m/3"

    monkeypatch.delenv("MY_CUSTOM_ROBOFLOW_KEY", raising=False)


def test_build_detector_ไม่มีตัวแปรสภาพแวดล้อมฟ้อง_error_ชัดเจน(fake_inference_sdk, monkeypatch):
    monkeypatch.delenv("MY_MISSING_KEY", raising=False)

    cfg = Config(
        model=ModelCfg(backend="roboflow"),
        roboflow=RoboflowCfg(api_key_env="MY_MISSING_KEY"),
        classes=ClassCfg(names=["strawberry", "weed"]),
    )
    cfg.validate()

    with pytest.raises(DetectorError, match="MY_MISSING_KEY"):
        build_detector(cfg)


def test_infer_แปลง_bgr_เป็น_rgb_ก่อนส่งและถอดผลลัพธ์ถูกต้อง(fake_inference_sdk):
    import numpy as np

    detector = RoboflowDetector(
        model_id="strawberry-detection-msf0m/3",
        api_url="https://serverless.roboflow.com",
        api_key="ทดสอบ",
        class_names=["strawberry", "weed"],
        conf=0.4,
    )

    captured = {}

    def fake_infer(image, model_id):
        captured["image_mode"] = image.mode
        captured["model_id"] = model_id
        return {
            "predictions": [
                {"class": "strawberry", "confidence": 0.88, "x": 50, "y": 60, "width": 20, "height": 30}
            ]
        }

    detector.client.infer = fake_infer

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.infer(frame)

    assert captured["model_id"] == "strawberry-detection-msf0m/3"
    assert captured["image_mode"] == "RGB"
    assert len(detections) == 1
    assert detections[0].class_name == "strawberry"
    assert "inference" in detector.last_timing_ms


def test_infer_เมื่อ_api_ล้มเหลวได้_error_ที่อ่านง่าย(fake_inference_sdk):
    detector = RoboflowDetector(
        model_id="strawberry-detection-msf0m/3",
        api_url="https://serverless.roboflow.com",
        api_key="ทดสอบ",
    )

    def broken_infer(image, model_id):
        raise ConnectionError("จำลองเน็ตหลุด")

    detector.client.infer = broken_infer

    import numpy as np

    with pytest.raises(DetectorError, match="Roboflow API"):
        detector.infer(np.zeros((10, 10, 3), dtype=np.uint8))
