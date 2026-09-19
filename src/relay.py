"""
ส่วนสั่งการ (The Action) — ควบคุมโมดูลรีเลย์ 1 ช่องเพื่อเปิด/ปิดอุปกรณ์ปลายทาง

อุปกรณ์ปลายทางคือปั๊มพ่นยา (ระบบจริง) หรือเลเซอร์ (prototype-v1 ที่ใช้นำเสนอผลงาน)
ตั้งที่ relay.actuator ใน config.yaml — โค้ดในไฟล์นี้ทำงานเหมือนกันทุกบรรทัด
ต่างแค่คำที่พิมพ์ออกคอนโซลของรีเลย์จำลอง

ข้อควรระวังเรื่องความปลอดภัย 3 ข้อ:
  1. ตั้ง relay.active_high ให้ตรงกับวงจร ไม่งั้นอุปกรณ์จะทำงานทันทีที่เปิดโปรแกรม
       โมดูลรีเลย์ราคาถูกส่วนใหญ่เป็น Active LOW (ส่งสัญญาณ 0 = รีเลย์ทำงาน) -> false
       เลเซอร์ผ่านทรานซิสเตอร์ NPN แบบ low-side (ส่งสัญญาณ 1 = ติด)        -> true
     ทดสอบก่อนต่ออุปกรณ์จริงเสมอ:
       ปั๊ม/รีเลย์ :  python tools/relay_test.py
       เลเซอร์     :  python tools/laser_test.py
  2. ทุกทางออกของโปรแกรม (จบปกติ, Ctrl+C, error) ต้องปิดรีเลย์ให้ได้
     จึงบังคับใช้ผ่าน try/finally และ context manager
  3. ถ้าไม่ได้อยู่บน Raspberry Pi จะใช้ MockRelay ที่แค่พิมพ์ข้อความ
     ทำให้ทดสอบตรรกะทั้งหมดบน PC ได้โดยไม่ต้องมีฮาร์ดแวร์
"""

from __future__ import annotations

import platform
import time
from abc import ABC, abstractmethod

from .config import RelayCfg


class RelayError(Exception):
    """ข้อผิดพลาดที่เกิดจากการควบคุมรีเลย์"""


class BaseRelay(ABC):
    """รีเลย์ 1 ช่อง"""

    def __init__(self, pin: int, active_high: bool) -> None:
        self.pin = int(pin)
        self.active_high = bool(active_high)
        self._is_on = False
        self.total_on_seconds = 0.0
        self.activation_count = 0
        self._turned_on_at: float | None = None

    @property
    def is_on(self) -> bool:
        return self._is_on

    @abstractmethod
    def _write(self, energized: bool) -> None:
        """เขียนสัญญาณลงขา GPIO จริง (คลาสลูกไปทำเอง)"""

    def on(self) -> None:
        """สั่งอุปกรณ์ปลายทางทำงาน (ไม่ทำอะไรถ้าทำงานอยู่แล้ว)"""
        if self._is_on:
            return
        self._write(True)
        self._is_on = True
        self._turned_on_at = time.monotonic()
        self.activation_count += 1

    def off(self) -> None:
        """สั่งอุปกรณ์ปลายทางหยุด (ไม่ทำอะไรถ้าหยุดอยู่แล้ว)"""
        if not self._is_on:
            return
        self._write(False)
        self._is_on = False
        if self._turned_on_at is not None:
            self.total_on_seconds += time.monotonic() - self._turned_on_at
            self._turned_on_at = None

    def set(self, should_be_on: bool) -> None:
        self.on() if should_be_on else self.off()

    def pulse(self, seconds: float) -> None:
        """เปิดค้างไว้ตามเวลาที่กำหนดแล้วปิด (ใช้ทดสอบเป็นหลัก)"""
        self.on()
        try:
            time.sleep(max(0.0, seconds))
        finally:
            self.off()

    def close(self) -> None:
        """ปิดรีเลย์และคืนทรัพยากร GPIO — ต้องเรียกเสมอไม่ว่าจะจบยังไง"""
        try:
            self.off()
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """คืนทรัพยากรเฉพาะของแต่ละ backend"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# คำที่รีเลย์จำลองใช้พิมพ์ออกคอนโซล ตาม relay.actuator — (คำไทย, คำอังกฤษ)
MOCK_WORDS = {
    "pump": ("ปั๊ม", "SPRAY"),
    "laser": ("เลเซอร์", "LASER"),
    "led": ("ไฟ LED", "LED"),
}


class MockRelay(BaseRelay):
    """รีเลย์จำลอง — ใช้บน PC ตอนพัฒนา แค่พิมพ์ข้อความแทนการสั่ง GPIO จริง"""

    def __init__(
        self,
        pin: int,
        active_high: bool,
        verbose: bool = True,
        actuator: str = "pump",
    ) -> None:
        super().__init__(pin, active_high)
        self.verbose = verbose
        self.history: list[tuple[float, bool]] = []
        self.device_th, self.device_en = MOCK_WORDS.get(actuator, MOCK_WORDS["pump"])
        if verbose:
            print(
                f"[relay] โหมดจำลอง (MOCK) — ไม่มีการสั่ง GPIO จริง | ขา {pin} "
                f"| อุปกรณ์ {self.device_th}"
            )

    def _write(self, energized: bool) -> None:
        self.history.append((time.monotonic(), energized))
        if self.verbose:
            if energized:
                state = f"เปิด{self.device_th}  >>> {self.device_en} ON"
            else:
                state = f"ปิด{self.device_th}   <<< {self.device_en} OFF"
            print(f"[relay] {state}")


class DirectGpioOutput(BaseRelay):
    """ขับขา GPIO ตรงๆ ไม่มีรีเลย์คั่น — สำหรับเลเซอร์ผ่านทรานซิสเตอร์ (backend: gpio)

    ต่างจาก GpiozeroRelay ตรง "สิ่งที่อยู่ปลายสาย" ไม่ใช่ตรงโค้ด:

      รีเลย์  : Pi -> ขา IN ของโมดูลรีเลย์ -> หน้าสัมผัสกล -> ปั๊ม (แยกไฟกันคนละวงจร)
      ตรงนี้  : Pi -> ตัวต้านทาน -> ขา B ของทรานซิสเตอร์ -> ทรานซิสเตอร์ต่อ/ตัดวงจรเลเซอร์

    ทำไมต้องมีทรานซิสเตอร์ ต่อเลเซอร์เข้าขา GPIO ตรงๆ ไม่ได้:
      ขา GPIO ของ Raspberry Pi จ่ายได้ปลอดภัยแค่ ~16 mA ต่อขา (รวมทุกขาไม่เกิน 50 mA)
      แต่โมดูลเลเซอร์ KY-008/HW-483 กินราว 30-40 mA — เกินพิกัด เสี่ยงทำให้ชิปพัง
      และแรงดัน GPIO มีแค่ 3.3V ขณะที่โมดูลออกแบบมาสำหรับ 5V (ต่อตรงจะหรี่หรือไม่ติด)
      ทรานซิสเตอร์แก้ทั้งสองข้อ: Pi จ่ายกระแสเข้าขา B แค่ ~3 mA ส่วนเลเซอร์ดูดไฟ 5V
      จากขาจ่ายไฟของบอร์ดโดยตรง

    ความปลอดภัย: ขาต้องอยู่สถานะ "ดับ" ตั้งแต่วินาทีแรกที่ตั้งเป็น OUTPUT เสมอ
    ซึ่ง gpiozero จัดการให้ด้วย initial_value=False — แต่ถ้าตั้ง active_high ผิดด้าน
    เลเซอร์จะติดค้างทันทีที่โปรแกรมเริ่ม จึงเตือนดังๆ ไว้ในคอนสตรัคเตอร์
    """

    def __init__(self, pin: int, active_high: bool, power: float = 1.0, actuator: str = "laser") -> None:
        super().__init__(pin, active_high)
        try:
            from gpiozero import OutputDevice, PWMOutputDevice  # type: ignore
        except ImportError as exc:
            raise RelayError(
                "ไม่พบไลบรารี gpiozero — ติดตั้งด้วย: sudo apt install python3-gpiozero"
            ) from exc

        self.power = float(power)
        device_words = MOCK_WORDS.get(actuator, MOCK_WORDS["pump"])
        self.device_th = device_words[0]

        if self.power < 1.0:
            # PWM ที่ 1 kHz เร็วเกินกว่าตาจะเห็นการกะพริบ ได้ผลเป็นความสว่างที่ลดลง
            self.device = PWMOutputDevice(
                self.pin, active_high=self.active_high, initial_value=0.0, frequency=1000
            )
        else:
            self.device = OutputDevice(self.pin, active_high=self.active_high, initial_value=False)

        print(
            f"[relay] ขับขา GPIO ตรง (ไม่มีรีเลย์) | ขา GPIO{self.pin} | "
            f"{'Active HIGH' if self.active_high else 'Active LOW'} | "
            f"อุปกรณ์ {self.device_th} | กำลัง {self.power * 100:.0f}%"
        )
        if not self.active_high:
            print(
                "[relay] ⚠️  เตือน: ขับ GPIO ตรงแต่ตั้ง active_high: false\n"
                f"        ขาจะถูกดึงเป็น HIGH ตอนพัก -> ถ้าใช้ทรานซิสเตอร์ NPN แบบ low-side\n"
                f"        {self.device_th}จะติดค้างตลอดเวลา ให้แก้ config.yaml เป็น active_high: true"
            )

    def _write(self, energized: bool) -> None:
        if self.power < 1.0:
            self.device.value = self.power if energized else 0.0
        else:
            self.device.on() if energized else self.device.off()

    def _cleanup(self) -> None:
        device = getattr(self, "device", None)
        if device is not None:
            try:
                device.close()
            except Exception:  # pragma: no cover
                pass


class GpiozeroRelay(BaseRelay):
    """ใช้ไลบรารี gpiozero (มาพร้อม Raspberry Pi OS อยู่แล้ว และรองรับ Pi 5)"""

    def __init__(self, pin: int, active_high: bool) -> None:
        super().__init__(pin, active_high)
        try:
            from gpiozero import OutputDevice  # type: ignore
        except ImportError as exc:
            raise RelayError("ไม่พบไลบรารี gpiozero — ติดตั้งด้วย: sudo apt install python3-gpiozero") from exc

        # initial_value=False = เริ่มต้นในสถานะ "ปั๊มปิด" เสมอ
        # gpiozero จัดการเรื่อง active_high ให้เอง ระดับไฟจึงถูกต้องตั้งแต่บูต
        self.device = OutputDevice(self.pin, active_high=self.active_high, initial_value=False)
        print(
            f"[relay] ใช้ gpiozero | ขา GPIO{self.pin} | "
            f"{'Active HIGH' if self.active_high else 'Active LOW'}"
        )

    def _write(self, energized: bool) -> None:
        self.device.on() if energized else self.device.off()

    def _cleanup(self) -> None:
        device = getattr(self, "device", None)
        if device is not None:
            try:
                device.close()
            except Exception:  # pragma: no cover
                pass


class RpiGpioRelay(BaseRelay):
    """ใช้ไลบรารี RPi.GPIO (ทางเลือกสำรอง ใช้ได้บน Pi 4 และรุ่นเก่ากว่า)"""

    def __init__(self, pin: int, active_high: bool) -> None:
        super().__init__(pin, active_high)
        try:
            import RPi.GPIO as GPIO  # type: ignore
        except ImportError as exc:
            raise RelayError("ไม่พบไลบรารี RPi.GPIO — ติดตั้งด้วย: pip install RPi.GPIO") from exc

        self.GPIO = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        # ตั้งค่าเริ่มต้นเป็นระดับ "ปั๊มปิด" ตั้งแต่บรรทัดแรกที่ตั้งขาเป็น OUTPUT
        idle_level = GPIO.LOW if self.active_high else GPIO.HIGH
        GPIO.setup(self.pin, GPIO.OUT, initial=idle_level)
        print(
            f"[relay] ใช้ RPi.GPIO | ขา GPIO{self.pin} | "
            f"{'Active HIGH' if self.active_high else 'Active LOW'}"
        )

    def _write(self, energized: bool) -> None:
        if self.active_high:
            level = self.GPIO.HIGH if energized else self.GPIO.LOW
        else:
            level = self.GPIO.LOW if energized else self.GPIO.HIGH
        self.GPIO.output(self.pin, level)

    def _cleanup(self) -> None:
        gpio = getattr(self, "GPIO", None)
        if gpio is not None:
            try:
                gpio.cleanup(self.pin)
            except Exception:  # pragma: no cover
                pass


def _looks_like_raspberry_pi() -> bool:
    """เดาว่ากำลังรันอยู่บน Raspberry Pi หรือไม่"""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/device-tree/model", "r", encoding="utf-8", errors="ignore") as fh:
            return "raspberry pi" in fh.read().lower()
    except OSError:
        # เครื่อง Linux ทั่วไปที่ไม่มีไฟล์นี้ = ไม่ใช่ Pi
        return False


def build_relay(cfg: RelayCfg, force_mock: bool = False) -> BaseRelay:
    """สร้างตัวควบคุมรีเลย์ตาม config (auto = ใช้ของจริงถ้าอยู่บน Pi)"""
    if force_mock or cfg.backend == "mock":
        return MockRelay(cfg.pin, cfg.active_high, actuator=cfg.actuator)

    if cfg.backend == "gpio":
        # ขับขาตรงๆ (เลเซอร์ผ่านทรานซิสเตอร์) — บนเครื่องที่ไม่ใช่ Pi ถอยไป mock เหมือน auto
        # เพื่อให้พัฒนาและสาธิตบน PC ได้โดยไม่ต้องแก้ config
        if not _looks_like_raspberry_pi():
            print("[relay] ไม่ได้รันอยู่บน Raspberry Pi -> ใช้รีเลย์จำลองแทนโดยอัตโนมัติ")
            return MockRelay(cfg.pin, cfg.active_high, actuator=cfg.actuator)
        return DirectGpioOutput(cfg.pin, cfg.active_high, cfg.power, cfg.actuator)

    if cfg.backend == "gpiozero":
        return GpiozeroRelay(cfg.pin, cfg.active_high)
    if cfg.backend == "rpigpio":
        return RpiGpioRelay(cfg.pin, cfg.active_high)

    # backend == "auto"
    if not _looks_like_raspberry_pi():
        print("[relay] ไม่ได้รันอยู่บน Raspberry Pi -> ใช้รีเลย์จำลองแทนโดยอัตโนมัติ")
        return MockRelay(cfg.pin, cfg.active_high, actuator=cfg.actuator)

    for builder in (GpiozeroRelay, RpiGpioRelay):
        try:
            return builder(cfg.pin, cfg.active_high)
        except RelayError as exc:
            print(f"[relay] {exc}")

    words = MOCK_WORDS.get(cfg.actuator, MOCK_WORDS["pump"])
    print(f"[relay] เตือน: ควบคุม GPIO ไม่ได้ -> ถอยไปใช้รีเลย์จำลอง ({words[0]}จะไม่ทำงานจริง)")
    return MockRelay(cfg.pin, cfg.active_high, actuator=cfg.actuator)
