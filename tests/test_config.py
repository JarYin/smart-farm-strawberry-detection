"""
ทดสอบการโหลดและตรวจสอบไฟล์ตั้งค่า

ค่าที่ตั้งผิดต้องถูกจับได้ตั้งแต่ตอนโหลด ไม่ใช่ไปพังตอนรถกำลังวิ่งอยู่กลางแปลง
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ClassCfg,
    ConfigError,
    ModelCfg,
    RoiCfg,
    SprayCfg,
    _load_dotenv_if_present,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_ไฟล์ตั้งค่าจริงของโปรเจกต์โหลดผ่าน():
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    assert cfg.model.imgsz == 320, "เอกสารโครงงานกำหนดให้ใช้ภาพขนาด 320x320"
    assert cfg.classes.names, "ต้องระบุชื่อคลาส"
    assert cfg.spray.max_on_seconds > 0, "ต้องมีระบบกันปั๊มเปิดค้าง"


def test_เลือกเอนจิ้นอัตโนมัติจากนามสกุลไฟล์():
    assert ModelCfg(weights="models/best.pt").resolved_backend() == "ultralytics"
    assert ModelCfg(weights="models/best_float32.tflite").resolved_backend() == "tflite"
    assert ModelCfg(backend="tflite", weights="models/best.pt").resolved_backend() == "tflite"


def test_ขนาดภาพต้องหารด้วย_32_ลงตัว():
    with pytest.raises(ConfigError, match="32"):
        ModelCfg(imgsz=333).validate()


def test_ค่า_confidence_ต้องอยู่ระหว่างศูนย์ถึงหนึ่ง():
    with pytest.raises(ConfigError):
        ModelCfg(conf=1.5).validate()


def test_คลาสเดียวกันเป็นทั้งพืชเป้าหมายและวัชพืชไม่ได้():
    with pytest.raises(ConfigError, match="วัชพืช"):
        ClassCfg(names=["weed"], target=["weed"], weed=["weed"]).validate()


def test_บทบาทของแต่ละคลาส():
    classes = ClassCfg(names=["strawberry", "weed", "rock"], target=["strawberry"], weed=["weed"])
    assert classes.role_of("strawberry") == "target"
    assert classes.role_of("weed") == "weed"
    assert classes.role_of("rock") == "unknown"

    classes.unknown_policy = "weed"
    assert classes.role_of("rock") == "weed"


def test_เกณฑ์_confidence_รายคลาส():
    classes = ClassCfg(names=["strawberry", "weed"], per_class_conf={"weed": 0.6})
    assert classes.conf_for("weed", 0.45) == pytest.approx(0.6)
    assert classes.conf_for("strawberry", 0.45) == pytest.approx(0.45)


def test_target_ว่างเปล่าถูกปฏิเสธ():
    with pytest.raises(ConfigError, match="target"):
        ClassCfg(names=["a", "b"], target=[]).validate()


def test_inverse_mode_ค่าเริ่มต้นคือปิด():
    assert ClassCfg().inverse_mode is False


def test_inverse_mode_เปิดพร้อม_weed_ว่างผ่านการตรวจสอบ():
    classes = ClassCfg(
        names=["strawberry-ripe", "strawberry-unripe"],
        target=["strawberry-ripe", "strawberry-unripe"],
        weed=[],
        inverse_mode=True,
    )
    classes.validate()  # ต้องไม่ raise


def test_แปลง_roi_เป็นพิกัดพิกเซล():
    roi = RoiCfg(x1=0.0, y1=0.5, x2=1.0, y2=1.0)
    assert roi.pixel_box(640, 480) == (0, 240, 640, 480)


def test_roi_ที่กลับด้านถูกปฏิเสธ():
    with pytest.raises(ConfigError):
        RoiCfg(x1=0.8, y1=0.1, x2=0.2, y2=0.9).validate()


def test_เวลาพ่นสูงสุดต้องไม่น้อยกว่าเวลาพ่นขั้นต่ำ():
    with pytest.raises(ConfigError):
        SprayCfg(min_on_seconds=5.0, max_on_seconds=1.0).validate()


def test_คีย์ที่พิมพ์ผิดถูกจับได้(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  imgz: 320\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="imgz"):
        load_config(bad)


def test_หมวดที่ไม่รู้จักถูกจับได้(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("camara:\n  index: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="camara"):
        load_config(bad)


def test_แหล่งภาพแบบวิดีโอต้องระบุพาธ(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("camera:\n  source: video\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="camera.path"):
        load_config(bad)


def test_พาธแบบสัมพัทธ์อ้างอิงจากรากโปรเจกต์เสมอ():
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    resolved = cfg.resolve_path("models/best.pt")
    assert resolved.is_absolute()
    assert resolved.parent.name == "models"


def test_ไฟล์ตั้งค่าจริงมีหมวด_roboflow_พร้อมค่าเริ่มต้นที่สมเหตุสมผล():
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    assert "/" in cfg.roboflow.model_id
    assert cfg.roboflow.api_key_env, "ต้องระบุชื่อตัวแปรสภาพแวดล้อมที่จะอ่านคีย์"
    assert cfg.roboflow.timeout_seconds > 0


def test_dotenv_เติมตัวแปรที่ยังไม่มีในสภาพแวดล้อม(tmp_path, monkeypatch):
    monkeypatch.delenv("SMARTFARM_TEST_VAR", raising=False)
    (tmp_path / ".env").write_text('SMARTFARM_TEST_VAR="จากไฟล์ .env"\n# คอมเมนต์\n\n', encoding="utf-8")

    _load_dotenv_if_present(tmp_path)

    assert os.environ.get("SMARTFARM_TEST_VAR") == "จากไฟล์ .env"
    monkeypatch.delenv("SMARTFARM_TEST_VAR", raising=False)


def test_dotenv_ไม่ทับตัวแปรที่ตั้งไว้แล้วในเชลล์(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTFARM_TEST_VAR", "จากเชลล์")
    (tmp_path / ".env").write_text("SMARTFARM_TEST_VAR=จากไฟล์\n", encoding="utf-8")

    _load_dotenv_if_present(tmp_path)

    assert os.environ.get("SMARTFARM_TEST_VAR") == "จากเชลล์"
    monkeypatch.delenv("SMARTFARM_TEST_VAR", raising=False)


def test_dotenv_ไม่มีไฟล์ก็ไม่พัง(tmp_path):
    _load_dotenv_if_present(tmp_path)  # ต้องไม่ throw
