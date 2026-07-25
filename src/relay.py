"""
ส่วนสั่งการ (The Action) — ควบคุมโมดูลรีเลย์ 1 ช่องเพื่อเปิด/ปิดปั๊มพ่นยา

ข้อควรระวังเรื่องความปลอดภัย 3 ข้อ:
  1. โมดูลรีเลย์ราคาถูกส่วนใหญ่เป็น Active LOW (ส่งสัญญาณ 0 = รีเลย์ทำงาน)
     ถ้าตั้ง relay.active_high ผิด ปั๊มจะทำงานทันทีที่เปิดโปรแกรม
     ให้ทดสอบด้วย  python tools/relay_test.py  ก่อนต่อปั๊มจริงเสมอ
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
        """เปิดปั๊ม (ไม่ทำอะไรถ้าเปิดอยู่แล้ว)"""
        if self._is_on:
            return
        self._write(True)
        self._is_on = True
        self._turned_on_at = time.monotonic()
        self.activation_count += 1

    def off(self) -> None:
        """ปิดปั๊ม (ไม่ทำอะไรถ้าปิดอยู่แล้ว)"""
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


class MockRelay(BaseRelay):
    """รีเลย์จำลอง — ใช้บน PC ตอนพัฒนา แค่พิมพ์ข้อความแทนการสั่ง GPIO จริง"""

    def __init__(self, pin: int, active_high: bool, verbose: bool = True) -> None:
        super().__init__(pin, active_high)
        self.verbose = verbose
        self.history: list[tuple[float, bool]] = []
        if verbose:
            print(f"[relay] โหมดจำลอง (MOCK) — ไม่มีการสั่ง GPIO จริง | ขา {pin}")

    def _write(self, energized: bool) -> None:
        self.history.append((time.monotonic(), energized))
        if self.verbose:
            state = "เปิดปั๊ม  >>> SPRAY ON" if energized else "ปิดปั๊ม   <<< SPRAY OFF"
            print(f"[relay] {state}")


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
        return MockRelay(cfg.pin, cfg.active_high)

    if cfg.backend == "gpiozero":
        return GpiozeroRelay(cfg.pin, cfg.active_high)
    if cfg.backend == "rpigpio":
        return RpiGpioRelay(cfg.pin, cfg.active_high)

    # backend == "auto"
    if not _looks_like_raspberry_pi():
        print("[relay] ไม่ได้รันอยู่บน Raspberry Pi -> ใช้รีเลย์จำลองแทนโดยอัตโนมัติ")
        return MockRelay(cfg.pin, cfg.active_high)

    for builder in (GpiozeroRelay, RpiGpioRelay):
        try:
            return builder(cfg.pin, cfg.active_high)
        except RelayError as exc:
            print(f"[relay] {exc}")

    print("[relay] เตือน: ควบคุม GPIO ไม่ได้ -> ถอยไปใช้รีเลย์จำลอง (ปั๊มจะไม่ทำงานจริง)")
    return MockRelay(cfg.pin, cfg.active_high)
