"""
Smart Farm Autonomous Robot — ระบบตรวจจับวัชพืชด้วย Edge AI และสั่งพ่นยาเฉพาะจุด

โครงสร้างโมดูล (แต่ละไฟล์รับผิดชอบเรื่องเดียว แก้ที่เดียวไม่กระทบส่วนอื่น):

    config.py       โหลดและตรวจสอบ config.yaml
    camera.py       รับภาพ (เว็บแคม / Pi Camera / วิดีโอ / โฟลเดอร์ภาพ)
    detection.py    โครงสร้างข้อมูลของวัตถุที่ตรวจพบ
    detector.py     รันโมเดล AI (ultralytics บน PC, TFLite บน Pi)
    vision_utils.py ฟังก์ชันภาพและกล่องพื้นฐาน (letterbox, NMS)
    logic.py        ตรรกะตัดสินใจพ่น (ROI, ยืนยันหลายเฟรม, กันพ่นค้าง)
    relay.py        ควบคุมรีเลย์/ปั๊ม ผ่าน GPIO
    overlay.py      วาดผลลัพธ์ลงภาพ
    stats.py        วัด FPS และบันทึกเหตุการณ์ลง CSV
    main.py         ลูปหลักที่ประกอบทุกอย่างเข้าด้วยกัน
"""

import sys

__version__ = "1.0.0"


def _force_utf8_console() -> None:
    """บังคับให้ stdout/stderr เป็น UTF-8 เสมอ

    ทุกโมดูลในโปรเจกต์พิมพ์ข้อความภาษาไทยลงคอนโซล แต่ Python บน Windows จะเลือก
    encoding ของ stdout ตาม "โค้ดเพจ" ของเทอร์มินัลที่เรียกมันขึ้นมา — PowerShell
    มักได้ UTF-8 แต่ Git Bash/cmd.exe บางเครื่องได้ cp1252 ซึ่งเข้ารหัสภาษาไทยไม่ได้
    ทำให้โปรแกรม crash ด้วย UnicodeEncodeError ทันทีที่ print ข้อความไทยครั้งแรก
    เรียกฟังก์ชันนี้ตั้งแต่ import แพ็กเกจ src เพื่อกันปัญหานี้ไม่ว่าจะรันจากเทอร์มินัลไหน
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


_force_utf8_console()
