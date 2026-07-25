"""
โครงสร้างข้อมูล "สิ่งที่ AI ตรวจพบ 1 ชิ้น"

แยกออกมาเป็นไฟล์เล็กๆ เพื่อให้ทั้งฝั่ง ultralytics (PC) และฝั่ง TFLite (Pi)
ส่งผลลัพธ์กลับมาในรูปแบบเดียวกัน โค้ดตรรกะการพ่นจึงไม่ต้องรู้เลยว่า
เบื้องหลังใช้เอนจิ้นตัวไหน
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """วัตถุที่ตรวจพบ 1 ชิ้น พิกัดเป็นพิกเซลของ "ภาพต้นฉบับ" เสมอ"""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_int_box(self) -> tuple[int, int, int, int]:
        """พิกัดจำนวนเต็มสำหรับวาดกรอบด้วย OpenCV"""
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    def __str__(self) -> str:
        return f"{self.class_name} {self.confidence:.2f} @ ({self.x1:.0f},{self.y1:.0f})-({self.x2:.0f},{self.y2:.0f})"
