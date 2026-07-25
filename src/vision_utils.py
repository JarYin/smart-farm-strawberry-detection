"""
ฟังก์ชันประมวลผลภาพและกล่องพื้นฐาน (ใช้ NumPy ล้วน ไม่พึ่ง PyTorch)

โมดูลนี้สำคัญมากสำหรับฝั่ง Raspberry Pi เพราะ TFLite ให้ผลลัพธ์ดิบมา
เราต้องแปลงพิกัดกลับเอง ต่างจาก ultralytics ที่จัดการให้หมดแล้ว
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def imread(path: str | Path, flags: int | None = None) -> np.ndarray | None:
    """อ่านไฟล์ภาพ — ใช้แทน cv2.imread เสมอ

    cv2.imread บน Windows อ่านพาธที่มีตัวอักษรไทยไม่ได้ และไม่ฟ้อง error ด้วย
    มันคืนค่า None เฉยๆ ทำให้ตามหาสาเหตุยากมาก
    จึงอ่านไฟล์เป็นไบต์ด้วย NumPy ก่อน แล้วให้ OpenCV ถอดรหัสภาพจากไบต์แทน
    """
    import cv2

    if flags is None:
        flags = cv2.IMREAD_COLOR
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, image: np.ndarray) -> bool:
    """เขียนไฟล์ภาพ — ใช้แทน cv2.imwrite เสมอ (เหตุผลเดียวกับ imread)"""
    import cv2

    path = Path(path)
    suffix = path.suffix if path.suffix else ".jpg"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer.tofile(str(path))
    except OSError:
        return False
    return True


def letterbox(
    image: np.ndarray,
    new_shape: int | Tuple[int, int] = 320,
    color: Tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """ย่อภาพให้พอดีกรอบสี่เหลี่ยมจัตุรัสโดย "ไม่บิดสัดส่วน" แล้วเติมขอบสีเทา

    ทำไมต้อง letterbox แทนการ resize ตรงๆ:
      ภาพจากกล้องเป็น 4:3 (640x480) ถ้าบีบเป็น 320x320 ตรงๆ ต้นสตรอว์เบอร์รี
      จะถูกบีบแบน ผิดจากภาพที่ใช้ตอนเทรน ทำให้ AI แม่นยำลดลง

    คืนค่า:
        img   : ภาพขนาด new_shape พร้อมใช้งาน
        ratio : อัตราส่วนที่ย่อ (ใช้แปลงพิกัดกลับ)
        pad   : (pad_left, pad_top) จำนวนพิกเซลขอบที่เติมเข้าไป
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    height, width = image.shape[:2]
    ratio = min(new_shape[0] / height, new_shape[1] / width)
    if not scaleup:
        ratio = min(ratio, 1.0)

    new_w = int(round(width * ratio))
    new_h = int(round(height * ratio))
    pad_w = (new_shape[1] - new_w) / 2
    pad_h = (new_shape[0] - new_h) / 2

    import cv2  # นำเข้าตรงนี้เพื่อให้ทดสอบส่วนอื่นได้แม้ไม่มี OpenCV

    if (width, height) != (new_w, new_h):
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (left, top)


def scale_boxes_back(
    boxes: np.ndarray,
    ratio: float,
    pad: Tuple[int, int],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """แปลงกล่อง xyxy จากพิกัดภาพ letterbox กลับเป็นพิกัดภาพต้นฉบับ

    boxes          : (N, 4) รูปแบบ x1,y1,x2,y2 ในระบบพิกัดของภาพที่ผ่าน letterbox
    original_shape : (height, width) ของภาพต้นฉบับ
    """
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)

    boxes = boxes.astype(np.float32).copy()
    pad_left, pad_top = pad
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= max(ratio, 1e-9)

    height, width = original_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """แปลงกล่องจากรูปแบบ (จุดกึ่งกลาง x, y, กว้าง, สูง) เป็น (x1, y1, x2, y2)"""
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    out = np.empty_like(boxes, dtype=np.float32)
    half_w = boxes[:, 2] / 2.0
    half_h = boxes[:, 3] / 2.0
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """คำนวณค่า IoU ระหว่างกล่องเดียวกับกล่องหลายกล่อง (ทั้งหมดเป็น xyxy)"""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return (inter / (area_a + area_b - inter + 1e-9)).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-Maximum Suppression แบบ NumPy ล้วน

    กำจัดกล่องซ้ำซ้อนที่ทับกันเกิน iou_threshold โดยเก็บกล่องที่คะแนนสูงสุดไว้
    คืนค่าเป็นรายการ index ของกล่องที่เก็บไว้ (เรียงตามคะแนนมากไปน้อย)
    """
    if boxes.size == 0:
        return []

    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return keep


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_det: int = 300,
) -> list[int]:
    """NMS ที่แยกทำทีละคลาส

    กล่อง "สตรอว์เบอร์รี" กับ "วัชพืช" ที่ทับกันต้องไม่ตัดกันเอง เพราะเป็นคนละสิ่ง
    เทคนิค: เลื่อนพิกัดกล่องของแต่ละคลาสออกไปไกลๆ ตาม class id
    ทำให้กล่องต่างคลาสไม่มีทางทับกัน แล้วเรียก NMS รอบเดียว
    """
    if boxes.size == 0:
        return []
    offset = class_ids.astype(np.float32) * 8192.0
    shifted = boxes.copy().astype(np.float32)
    shifted[:, [0, 2]] += offset[:, None]
    shifted[:, [1, 3]] += offset[:, None]
    return nms(shifted, scores, iou_threshold)[:max_det]


def expand_box(box: np.ndarray | tuple, margin: float) -> np.ndarray:
    """ขยายกล่องออกทุกด้านเท่ากับ margin พิกเซล (ใช้ทำระยะกันชนรอบพืชเป้าหมาย)"""
    x1, y1, x2, y2 = box
    return np.array([x1 - margin, y1 - margin, x2 + margin, y2 + margin], dtype=np.float32)


def boxes_overlap(box_a, box_b) -> bool:
    """เช็คว่ากล่องสองกล่องซ้อนทับกันหรือไม่ (แตะขอบพอดีไม่นับ)"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def point_in_box(x: float, y: float, box) -> bool:
    """เช็คว่าจุด (x, y) อยู่ในกล่องหรือไม่"""
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2
