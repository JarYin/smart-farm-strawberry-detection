"""
ทดสอบการขับเลเซอร์ผ่านขา GPIO โดยตรง (backend: gpio)

ทดสอบได้ครบบน PC โดยไม่ต้องมี Raspberry Pi เพราะ build_relay() จะถอยไปใช้
ตัวจำลองเองเมื่อไม่ได้อยู่บน Pi — สิ่งที่เทสต์ชุดนี้เฝ้าคือ "การตั้งค่าที่ทำให้
เลเซอร์ติดค้าง" ซึ่งเป็นความผิดพลาดที่อันตรายและมองไม่เห็นจนกว่าจะต่อของจริง

รันด้วย:  pytest tests/test_laser_output.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import relay as relay_module
from src.config import ConfigError, RelayCfg, load_config
from src.relay import MockRelay, build_relay

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# การตรวจสอบไฟล์ตั้งค่า
# ---------------------------------------------------------------------------


def test_backend_gpio_ผ่านการตรวจสอบ():
    RelayCfg(backend="gpio", active_high=True, actuator="laser").validate()


def test_backend_ที่ไม่รู้จักถูกปฏิเสธ():
    with pytest.raises(ConfigError, match="relay.backend"):
        RelayCfg(backend="transistor").validate()


def test_ค่ากำลังเริ่มต้นคือเต็มร้อย():
    assert RelayCfg().power == 1.0


@pytest.mark.parametrize("power", [0.0, -0.2, 1.5])
def test_กำลังนอกช่วงถูกปฏิเสธ(power):
    with pytest.raises(ConfigError, match="relay.power"):
        RelayCfg(backend="gpio", power=power).validate()


def test_ลดกำลังกับรีเลย์จริงถูกปฏิเสธ():
    """PWM ทำให้หน้าสัมผัสของรีเลย์กลไกพัง — ต้องกันไว้ตั้งแต่ตอนโหลด config"""
    with pytest.raises(ConfigError, match="PWM"):
        RelayCfg(backend="rpigpio", power=0.5).validate()


def test_ลดกำลังกับ_backend_gpio_ทำได้():
    RelayCfg(backend="gpio", active_high=True, power=0.3).validate()


def test_actuator_ที่ไม่รู้จักถูกปฏิเสธ():
    with pytest.raises(ConfigError, match="relay.actuator"):
        RelayCfg(actuator="servo").validate()


# ---------------------------------------------------------------------------
# การเลือกตัวขับ
# ---------------------------------------------------------------------------


def test_backend_gpio_บนเครื่องที่ไม่ใช่_pi_ถอยไปใช้ตัวจำลอง(monkeypatch):
    """ต้องพัฒนาและสาธิตบน PC ได้โดยไม่ต้องแก้ config กลับไปกลับมา"""
    monkeypatch.setattr(relay_module, "_looks_like_raspberry_pi", lambda: False)
    device = build_relay(RelayCfg(backend="gpio", active_high=True, actuator="laser"))

    assert isinstance(device, MockRelay)
    assert device.is_on is False, "ต้องเริ่มที่สถานะดับเสมอ"
    device.close()


def test_backend_gpio_บน_pi_ใช้ตัวขับขาตรง(monkeypatch):
    """บน Pi จริงต้องไม่หลุดไปใช้ตัวจำลองเงียบๆ จนคิดว่าระบบทำงานทั้งที่เลเซอร์ไม่ติด"""
    monkeypatch.setattr(relay_module, "_looks_like_raspberry_pi", lambda: True)
    built = {}

    class FakeDirect:
        def __init__(self, pin, active_high, power, actuator):
            built.update(pin=pin, active_high=active_high, power=power, actuator=actuator)

    monkeypatch.setattr(relay_module, "DirectGpioOutput", FakeDirect)
    build_relay(RelayCfg(backend="gpio", pin=17, active_high=True, actuator="laser", power=0.4))

    assert built == {"pin": 17, "active_high": True, "power": 0.4, "actuator": "laser"}


def test_บังคับตัวจำลองได้เสมอแม้ตั้ง_backend_gpio(monkeypatch):
    """ธง --mock-relay ต้องชนะทุกกรณี ไม่งั้นทดสอบบนโต๊ะแล้วเลเซอร์ยิงจริง"""
    monkeypatch.setattr(relay_module, "_looks_like_raspberry_pi", lambda: True)
    device = build_relay(RelayCfg(backend="gpio", active_high=True), force_mock=True)

    assert isinstance(device, MockRelay)
    device.close()


def test_ตัวจำลองใช้คำเรียกตามอุปกรณ์ที่ต่ออยู่():
    laser = MockRelay(17, True, verbose=False, actuator="laser")
    pump = MockRelay(17, False, verbose=False, actuator="pump")

    assert (laser.device_th, laser.device_en) == ("เลเซอร์", "LASER")
    assert (pump.device_th, pump.device_en) == ("ปั๊ม", "SPRAY")


# ---------------------------------------------------------------------------
# พฤติกรรมพื้นฐานที่ห้ามพลาด
# ---------------------------------------------------------------------------


def test_เลเซอร์ดับเสมอเมื่อออกจาก_context_manager():
    laser = MockRelay(17, True, verbose=False, actuator="laser")

    with pytest.raises(RuntimeError):
        with laser:
            laser.on()
            assert laser.is_on is True
            raise RuntimeError("จำลองโปรแกรมพังระหว่างยิง")

    assert laser.is_on is False, "เกิด error แล้วเลเซอร์ต้องดับอัตโนมัติ"


def test_สั่งซ้ำไม่นับเป็นการเปิดใหม่():
    laser = MockRelay(17, True, verbose=False, actuator="laser")
    for _ in range(5):
        laser.set(True)
    laser.set(False)

    assert laser.activation_count == 1
    laser.close()


# ---------------------------------------------------------------------------
# ไฟล์ตั้งค่าจริงของ branch นี้
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config(PROJECT_ROOT / "config.yaml")


def test_ไฟล์ตั้งค่าจริงขับเลเซอร์ผ่านขา_gpio_ตรง(cfg):
    assert cfg.relay.backend == "gpio"
    assert cfg.relay.actuator == "laser"


def test_ไฟล์ตั้งค่าจริงตั้ง_active_high_ถูกด้าน(cfg):
    """วงจรทรานซิสเตอร์ NPN แบบ low-side: HIGH = ติด

    ถ้าใครเผลอแก้กลับเป็น false เลเซอร์จะติดค้างตั้งแต่โปรแกรมเริ่ม
    และไม่มีทางรู้จนกว่าจะต่อของจริงแล้วมองเห็นจุดแดง — จึงล็อกไว้ที่นี่
    """
    assert cfg.relay.active_high is True, (
        "backend gpio + active_high false = เลเซอร์ติดค้างตลอดเวลา"
    )


def test_เวลาทำงานสูงสุดยังถูกจำกัดไว้(cfg):
    """กลไกกันค้างต้องอยู่เสมอ ไม่ว่าอุปกรณ์ปลายทางจะเป็นอะไร"""
    assert 0 < cfg.spray.max_on_seconds <= 60.0
