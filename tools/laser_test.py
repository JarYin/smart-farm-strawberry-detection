"""
ทดสอบเลเซอร์ KY-008 / HW-483 ที่ขับผ่านทรานซิสเตอร์ — รันให้ผ่านก่อนรันระบบเต็ม

ทำไมไม่ใช้ tools/relay_test.py: ตัวนั้นให้ "ฟังเสียงคลิก" ของรีเลย์เป็นหลักฐาน
แต่เลเซอร์ไม่มีเสียง หลักฐานคือ "แสง" ซึ่งเป็นสิ่งที่อันตรายต่อตาพอดี
สคริปต์นี้จึงบังคับลำดับที่ปลอดภัย: ตรวจสถานะพักก่อน -> เตือน -> ค่อยยิงสั้นๆ

⚠️ ความปลอดภัยของเลเซอร์ อ่านให้ครบก่อนรัน:
  1. หันหัวเลเซอร์ "ลงพื้นโต๊ะ" หรือใส่กล่อง/กระดาษขาวรองไว้เสมอ
     ห้ามหันเข้าหาคน สัตว์ กระจก จอมอนิเตอร์ หรือของมันวาวที่สะท้อนได้
  2. ห้ามมองเข้าไปในลำแสงหรือจุดสะท้อนโดยตรง แม้จะเป็นเลเซอร์กำลังต่ำ (~5 mW)
     ดูที่ "จุดแดงบนพื้น" จากมุมเฉียงแทน
  3. ทดสอบครั้งแรกให้ถอดสายเลเซอร์ออกก่อน แล้ววัดแรงดันที่ขาคอลเลกเตอร์ด้วยมัลติมิเตอร์
     (ใช้โหมด --dry-run ของสคริปต์นี้) เมื่อมั่นใจว่าลำดับถูกต้องแล้วค่อยต่อเลเซอร์
  4. อย่าให้เด็กหรือผู้ชมเข้ามาในแนวลำแสงระหว่างสาธิต

สิ่งที่สคริปต์นี้พิสูจน์:
  1. ตอนโปรแกรมยังไม่สั่งอะไร เลเซอร์ต้อง "ดับ" — ถ้าติดตั้งแต่แรกแปลว่าตั้ง
     relay.active_high ผิดด้าน (ดู README หัวข้อ 6)
  2. สั่ง on แล้วต้องติด สั่ง off แล้วต้องดับ
  3. กด Ctrl+C กลางคัน เลเซอร์ต้องดับเองทุกครั้ง
  4. ลดกำลังด้วย relay.power แล้วความสว่างลดลงจริง (ถ้าใช้ backend gpio/gpiozero)

วิธีใช้:
    python tools/laser_test.py                 # ลำดับทดสอบมาตรฐาน
    python tools/laser_test.py --dry-run       # ไม่แตะ GPIO จริง ดูลำดับคำสั่งอย่างเดียว
    python tools/laser_test.py --cycles 5 --on 0.3 --off 0.7
    python tools/laser_test.py --power 0.3     # ทดสอบที่กำลัง 30%
    python tools/laser_test.py --sweep         # ไล่กำลัง 20% -> 100% ดูว่า PWM ทำงานไหม
"""

import _bootstrap  # noqa: F401

import argparse
import time

from src.config import load_config
from src.relay import build_relay

SAFETY_BANNER = """
    ██  ความปลอดภัยของเลเซอร์  ██
    - หันหัวเลเซอร์ลงพื้นโต๊ะ / ใส่กล่องรองไว้
    - ห้ามมองลำแสงหรือจุดสะท้อนตรงๆ ดูจุดแดงบนพื้นจากมุมเฉียงแทน
    - ห้ามหันเข้าหาคน กระจก จอภาพ หรือของมันวาว
"""


def countdown(seconds: int, message: str) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"  {message} ใน {remaining} วินาที...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 70, end="\r")


def parse_args():
    parser = argparse.ArgumentParser(description="ทดสอบเลเซอร์ KY-008/HW-483 ที่ขับผ่านทรานซิสเตอร์")
    parser.add_argument("--config", default=None)
    parser.add_argument("--pin", type=int, default=None, help="ขา GPIO (แทนค่าใน config)")
    parser.add_argument("--power", type=float, default=None, help="กำลัง 0.0-1.0 (แทนค่าใน config)")
    parser.add_argument("--cycles", type=int, default=3, help="จำนวนรอบติด-ดับ")
    parser.add_argument("--on", type=float, default=0.5, help="ติดค้างกี่วินาทีต่อรอบ")
    parser.add_argument("--off", type=float, default=0.8, help="ดับพักกี่วินาทีต่อรอบ")
    parser.add_argument("--sweep", action="store_true", help="ไล่ระดับกำลังเพื่อตรวจว่า PWM ทำงาน")
    parser.add_argument("--dry-run", action="store_true", help="ไม่แตะ GPIO จริง (ใช้ตัวจำลอง)")
    parser.add_argument("--yes", action="store_true", help="ข้ามการถามยืนยันความปลอดภัย")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if args.pin is not None:
        cfg.relay.pin = args.pin
    if args.power is not None:
        cfg.relay.power = args.power
    if args.dry_run:
        cfg.relay.backend = "mock"
        cfg.relay.power = 1.0  # ตัวจำลองไม่รองรับ PWM
    cfg.validate()

    print("=" * 72)
    print(" ทดสอบเลเซอร์ (KY-008 / HW-483) ที่ขับผ่านทรานซิสเตอร์")
    print("=" * 72)
    print(f" ขา GPIO      : {cfg.relay.pin}  (BCM numbering)")
    print(f" ตรรกะสัญญาณ  : {'Active HIGH' if cfg.relay.active_high else 'Active LOW'}")
    print(f" backend      : {cfg.relay.backend}")
    print(f" กำลัง        : {cfg.relay.power * 100:.0f}%")
    print(SAFETY_BANNER)

    if cfg.relay.backend == "gpio" and not cfg.relay.active_high:
        print(" ⚠️  ตั้ง backend: gpio คู่กับ active_high: false — วงจรทรานซิสเตอร์ NPN แบบ")
        print("     low-side จะทำให้เลเซอร์ติดค้างตลอดเวลา ตรวจ config.yaml ก่อนรันต่อ")

    if not args.yes:
        answer = input(" หันหัวเลเซอร์ลงพื้นและไม่มีใครอยู่ในแนวลำแสงแล้วใช่ไหม? (พิมพ์ y เพื่อไปต่อ) ")
        if answer.strip().lower() not in {"y", "yes"}:
            print(" ยกเลิกการทดสอบ")
            return 1

    laser = build_relay(cfg.relay)

    try:
        print("-" * 72)
        print("[ขั้นที่ 1] ตรวจสถานะเริ่มต้น — เลเซอร์ต้อง 'ดับ' อยู่")
        print(f"  สถานะที่โปรแกรมเชื่อ : {'ติด (ผิดปกติ!)' if laser.is_on else 'ดับ'}")
        print("  ดูของจริง: ถ้าจุดแดงติดอยู่แล้วทั้งที่ยังไม่สั่งอะไร -> ตั้ง active_high ผิดด้าน")
        countdown(3, "ไปขั้นถัดไป")

        print(f"\n[ขั้นที่ 2] ยิงสั้นๆ {args.cycles} รอบ (ติด {args.on} วิ / ดับ {args.off} วิ)")
        for index in range(1, args.cycles + 1):
            print(f"  รอบ {index}/{args.cycles} : ยิง", end="", flush=True)
            laser.on()
            time.sleep(args.on)
            laser.off()
            print(" -> ดับ")
            time.sleep(args.off)

        if args.sweep:
            print("\n[ขั้นที่ 3] ไล่ระดับกำลัง — ความสว่างต้องเพิ่มขึ้นเป็นขั้นๆ")
            if not hasattr(laser, "power"):
                print("  ข้าม: backend ปัจจุบันไม่รองรับการลดกำลังด้วย PWM")
            else:
                original = laser.power
                for level in (0.2, 0.4, 0.6, 0.8, 1.0):
                    laser.power = level
                    print(f"  กำลัง {level * 100:>3.0f}%", end="", flush=True)
                    laser.off()
                    laser.on()
                    time.sleep(0.8)
                    print(" ok")
                laser.off()
                laser.power = original

        print("\n[ขั้นที่ 4] ทดสอบการปิดตอนโปรแกรมพัง — ยิงค้างแล้วโยน error")
        try:
            with laser:
                laser.on()
                raise RuntimeError("จำลองโปรแกรมพังระหว่างยิง")
        except RuntimeError:
            pass
        print(f"  สถานะหลังเกิด error : {'ติดค้าง (ผิดปกติ!)' if laser.is_on else 'ดับเรียบร้อย'}")

    except KeyboardInterrupt:
        print("\n ผู้ใช้กด Ctrl+C")
    finally:
        laser.close()

    print("-" * 72)
    print(f" เปิดใช้งานทั้งหมด : {laser.activation_count} ครั้ง")
    print(f" เวลาติดรวม        : {laser.total_on_seconds:.2f} วินาที")
    print(f" สถานะตอนจบ        : {'ติดค้าง (ผิดปกติ!)' if laser.is_on else 'ดับเรียบร้อย'}")
    print("=" * 72)
    print(" ถ้าทุกขั้นตรงตามที่อธิบายไว้ ต่อระบบเต็มได้เลย:  python -m src.main --headless")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
