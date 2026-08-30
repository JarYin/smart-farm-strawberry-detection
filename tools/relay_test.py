"""
ทดสอบรีเลย์และอุปกรณ์ปลายทาง — ต้องรันตัวนี้ให้ผ่านก่อนรันระบบเต็ม

อุปกรณ์ปลายทางคือปั๊มพ่นยาหรือเลเซอร์ ตามที่ตั้งไว้ที่ relay.actuator ใน config.yaml
ขั้นตอนการทดสอบเหมือนกันทั้งสองแบบ (สิ่งที่ทดสอบคือรีเลย์ ไม่ใช่ตัวอุปกรณ์)

สิ่งที่ต้องพิสูจน์ก่อนต่ออุปกรณ์จริง:
  1. ตอนโปรแกรมยังไม่สั่งอะไร รีเลย์ต้องอยู่สถานะ "ปิด"
     ถ้าได้ยินเสียงคลิกและรีเลย์ติดทันทีที่รันโปรแกรม แปลว่าตั้ง active_high ผิด
  2. สั่ง on แล้วรีเลย์ต้องทำงาน สั่ง off แล้วต้องหยุด
  3. กด Ctrl+C กลางคัน รีเลย์ต้องปิดเองทุกครั้ง

คำเตือน: ทดสอบครั้งแรกให้ถอดสายอุปกรณ์ออกก่อน ฟังแค่เสียงคลิกของรีเลย์
เมื่อมั่นใจว่าลำดับถูกต้องแล้วค่อยต่อของจริง (ปั๊ม: ใส่น้ำเปล่าทดสอบ |
เลเซอร์: หันลำแสงลงพื้นโต๊ะเสมอ ห้ามเข้าตาหรือสะท้อนพื้นผิวมันวาว)

วิธีใช้:
    python tools/relay_test.py                  # ทดสอบเปิด-ปิด 3 รอบ
    python tools/relay_test.py --cycles 5 --on 1.0 --off 1.0
    python tools/relay_test.py --interactive    # สั่งเองทีละคำสั่ง
"""

import _bootstrap  # noqa: F401

import argparse
import time

from src.config import load_config
from src.relay import build_relay


def countdown(seconds: int, message: str) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"  {message} ใน {remaining} วินาที...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 60, end="\r")


def main() -> int:
    parser = argparse.ArgumentParser(description="ทดสอบโมดูลรีเลย์ควบคุมปั๊ม/เลเซอร์")
    parser.add_argument("--config", default=None)
    parser.add_argument("--pin", type=int, default=None, help="ขา GPIO (แทนค่าใน config)")
    parser.add_argument("--cycles", type=int, default=3, help="จำนวนรอบเปิด-ปิด")
    parser.add_argument("--on", type=float, default=1.0, help="เปิดค้างกี่วินาทีต่อรอบ")
    parser.add_argument("--off", type=float, default=1.0, help="ปิดพักกี่วินาทีต่อรอบ")
    parser.add_argument("--interactive", action="store_true", help="โหมดสั่งเองทีละคำสั่ง")
    parser.add_argument("--mock", action="store_true", help="บังคับใช้รีเลย์จำลอง")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.pin is not None:
        cfg.relay.pin = args.pin
        cfg.validate()

    device = "เลเซอร์" if cfg.relay.actuator == "laser" else "ปั๊มพ่นยา"

    print("=" * 72)
    print(f" ทดสอบรีเลย์ควบคุม{device}")
    print("=" * 72)
    print(f" ขา GPIO      : {cfg.relay.pin}  (BCM numbering)")
    print(f" ตรรกะสัญญาณ  : {'Active HIGH' if cfg.relay.active_high else 'Active LOW'}")
    print(f" backend      : {cfg.relay.backend}")
    print(f" อุปกรณ์      : {device}  (relay.actuator = {cfg.relay.actuator})")
    print("-" * 72)
    print(f" คำเตือน: ครั้งแรกให้ถอดสาย{device}ออกก่อน ฟังแค่เสียงคลิกของรีเลย์")
    print("-" * 72)

    relay = build_relay(cfg.relay, force_mock=args.mock)

    try:
        print("\n[ขั้นที่ 1] ตรวจสถานะเริ่มต้น — รีเลย์ต้องยังไม่ทำงาน")
        print(f"  สถานะที่โปรแกรมเข้าใจ: {'เปิด' if relay.is_on else 'ปิด'}")
        print("  ถ้าได้ยินเสียงคลิกตั้งแต่ตอนนี้ = ตั้ง relay.active_high ผิด ให้สลับค่าใน config.yaml")
        countdown(3, "เริ่มทดสอบ")

        if args.interactive:
            print("\n[โหมดสั่งเอง] พิมพ์คำสั่ง: on / off / pulse / q")
            while True:
                try:
                    command = input("  คำสั่ง > ").strip().lower()
                except EOFError:
                    break
                if command in ("q", "quit", "exit"):
                    break
                if command == "on":
                    relay.on()
                elif command == "off":
                    relay.off()
                elif command.startswith("pulse"):
                    parts = command.split()
                    duration = float(parts[1]) if len(parts) > 1 else 1.0
                    relay.pulse(duration)
                else:
                    print("    คำสั่งที่ใช้ได้: on, off, pulse [วินาที], q")
                print(f"    สถานะปัจจุบัน: {'เปิด' if relay.is_on else 'ปิด'}")
        else:
            print(f"\n[ขั้นที่ 2] ทดสอบเปิด-ปิด {args.cycles} รอบ")
            for cycle in range(1, args.cycles + 1):
                print(f"  รอบที่ {cycle}/{args.cycles}: เปิด {args.on} วินาที")
                relay.on()
                time.sleep(args.on)
                print(f"  รอบที่ {cycle}/{args.cycles}: ปิด {args.off} วินาที")
                relay.off()
                time.sleep(args.off)

        print("\n[ขั้นที่ 3] ทดสอบระบบตัดฉุกเฉิน — เปิดค้างแล้วสั่งปิดด้วย close()")
        relay.on()
        time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n  ผู้ใช้กด Ctrl+C — ระบบต้องปิดรีเลย์ให้เองต่อจากนี้")
    finally:
        relay.close()

    print("\n" + "=" * 72)
    print(" ผลการทดสอบ")
    print("=" * 72)
    print(f" จำนวนครั้งที่สั่งเปิด : {relay.activation_count}")
    print(f" เวลาเปิดรวม         : {relay.total_on_seconds:.2f} วินาที")
    print(f" สถานะตอนจบ         : {'เปิดอยู่ (ผิดปกติ!)' if relay.is_on else 'ปิดเรียบร้อย'}")
    print("\n ถ้าทุกอย่างถูกต้อง ขั้นตอนถัดไป:  python -m src.main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
