"""
โหลดและตรวจสอบไฟล์ตั้งค่า config.yaml

เหตุผลที่แยกเป็นโมดูล: ทุกส่วนของระบบ (กล้อง, โมเดล, ตรรกะพ่น, รีเลย์)
อ่านค่าจากที่เดียวกัน และตรวจความถูกต้องตั้งแต่ตอนโหลด
ถ้าพิมพ์ชื่อคีย์ผิดจะฟ้องทันที ไม่ใช่ไปพังกลางป่า
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List

# รากของโปรเจกต์ = โฟลเดอร์ที่อยู่เหนือ src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigError(Exception):
    """ข้อผิดพลาดที่เกิดจากไฟล์ตั้งค่าไม่ถูกต้อง"""


# ---------------------------------------------------------------------------
# โครงสร้างของแต่ละหมวดใน config.yaml
# ---------------------------------------------------------------------------


@dataclass
class CameraCfg:
    source: str = "webcam"
    index: int = 0
    path: str = ""
    loop: bool = True
    hold_frames: int = 15
    width: int = 640
    height: int = 480
    fps: int = 30
    flip_horizontal: bool = False
    flip_vertical: bool = False
    warmup_frames: int = 5

    def validate(self) -> None:
        allowed = {"webcam", "picamera2", "video", "images"}
        if self.source not in allowed:
            raise ConfigError(
                f"camera.source ต้องเป็นหนึ่งใน {sorted(allowed)} (ได้รับ '{self.source}')"
            )
        if self.source in {"video", "images"} and not self.path:
            raise ConfigError(f"camera.source = '{self.source}' ต้องระบุ camera.path ด้วย")
        if self.width <= 0 or self.height <= 0:
            raise ConfigError("camera.width และ camera.height ต้องมากกว่า 0")


@dataclass
class ModelCfg:
    backend: str = "auto"
    weights: str = "models/best.pt"
    imgsz: int = 320
    conf: float = 0.45
    iou: float = 0.45
    max_det: int = 20
    num_threads: int = 4

    def validate(self) -> None:
        allowed = {"auto", "ultralytics", "tflite", "roboflow", "color"}
        if self.backend not in allowed:
            raise ConfigError(
                f"model.backend ต้องเป็นหนึ่งใน {sorted(allowed)} (ได้รับ '{self.backend}')"
            )
        if self.imgsz <= 0 or self.imgsz % 32 != 0:
            raise ConfigError(f"model.imgsz ต้องเป็นจำนวนเต็มบวกที่หารด้วย 32 ลงตัว (ได้รับ {self.imgsz})")
        if not 0.0 < self.conf < 1.0:
            raise ConfigError(f"model.conf ต้องอยู่ระหว่าง 0 ถึง 1 (ได้รับ {self.conf})")
        if not 0.0 < self.iou < 1.0:
            raise ConfigError(f"model.iou ต้องอยู่ระหว่าง 0 ถึง 1 (ได้รับ {self.iou})")

    def resolved_backend(self) -> str:
        """แปลง backend='auto' เป็นค่าจริงโดยดูจากนามสกุลไฟล์น้ำหนักโมเดล

        หมายเหตุ: backend 'color' (ตรวจจับด้วยสีล้วน ไม่ใช้โมเดล AI) ต้องระบุตรงๆ เท่านั้น
        เพราะไม่มีไฟล์น้ำหนักให้เดาจากนามสกุล — ดูหมวด color ใน config.yaml
        """
        if self.backend != "auto":
            return self.backend
        return "tflite" if self.weights.lower().endswith(".tflite") else "ultralytics"

    def uses_weights_file(self) -> bool:
        """เอนจิ้นที่เลือกไว้ต้องมีไฟล์น้ำหนักโมเดลหรือไม่"""
        return self.resolved_backend() in {"ultralytics", "tflite"}


@dataclass
class RoboflowCfg:
    """ตั้งค่าสำหรับเรียกใช้โมเดลที่ฝากไว้บน Roboflow ผ่าน API (เอนจิ้นคลาวด์)

    เปิดใช้โดยตั้ง model.backend: roboflow ใน config.yaml
    ห้ามใส่ api key ไว้ในไฟล์นี้เด็ดขาด — ตั้งเป็นตัวแปรสภาพแวดล้อมแทน (ดู api_key_env)
    """

    api_url: str = "https://serverless.roboflow.com"
    model_id: str = "strawberry-detection-msf0m/3"
    api_key_env: str = "ROBOFLOW_API_KEY"
    timeout_seconds: float = 10.0

    def validate(self) -> None:
        if not self.api_url.strip():
            raise ConfigError("roboflow.api_url ห้ามว่างเปล่า")
        if "/" not in self.model_id:
            raise ConfigError(
                f"roboflow.model_id ต้องอยู่ในรูปแบบ 'ชื่อโมเดล/เวอร์ชัน' (ได้รับ '{self.model_id}')"
            )
        if not self.api_key_env.strip():
            raise ConfigError("roboflow.api_key_env ห้ามว่างเปล่า (ต้องระบุชื่อตัวแปรสภาพแวดล้อม)")
        if self.timeout_seconds <= 0:
            raise ConfigError("roboflow.timeout_seconds ต้องมากกว่า 0")


@dataclass
class ClassCfg:
    names: List[str] = field(default_factory=lambda: ["strawberry", "weed"])
    target: List[str] = field(default_factory=lambda: ["strawberry"])
    weed: List[str] = field(default_factory=lambda: ["weed"])
    unknown_policy: str = "ignore"
    per_class_conf: Dict[str, float] = field(default_factory=dict)

    # โหมดผกผัน (Inverse / Crop-vs-Non-crop detection):
    #   false (ค่าเริ่มต้น) -> ต้องมีคลาสวัชพืชในโมเดลจริงๆ ถึงจะพ่น (ตรรกะปกติ)
    #   true  -> "พืชเป้าหมายเท่านั้นที่ห้ามแตะ" — พ่นทั้งเขต ROI ทันทีที่ไม่พบพืชเป้าหมาย
    #            แม้แต่ต้นเดียวในเขตนั้น เหมาะกับโมเดลที่สอนให้รู้จักแค่พืชหลัก
    #            (เช่น strawberry-ripe/unripe) ไม่มีคลาสวัชพืชแยก ประหยัดการประมวลผลกว่า
    #            เพราะไม่ต้องคำนวณระยะห่าง/ซ้อนทับเป็นรายกล่อง แค่เช็คว่า "เจอพืชเป้าหมาย
    #            ในเขตพ่นหรือไม่" ครั้งเดียวต่อเฟรม
    #            ถ้าโมเดลบังเอิญมีคลาสวัชพืชด้วย (เช่น โหมดสีของ prototype-v1 ที่แดง=พืชหลัก
    #            เขียว=วัชพืช) กล่องวัชพืชเหล่านั้นจะถูกใช้เป็นเป้าหมายการพ่นแทนกล่องสังเคราะห์
    #            ทั้งเขต ทำให้ log และภาพผลลัพธ์ชี้ตำแหน่งจริงได้ แต่ "ผลการตัดสิน" ยังเหมือนเดิม
    #            ข้อควรระวัง: จะพ่นโดนดินเปล่า/พื้นที่ว่างที่ไม่มีทั้งพืชหลักและวัชพืชด้วย
    inverse_mode: bool = False

    # กันไม่ให้พิมพ์คำเตือนซ้ำ เพราะ validate() ถูกเรียกหลายครั้ง
    # (ตอนโหลด และตอนเอาค่าจาก command line มาทับ)
    _warned: bool = field(default=False, repr=False, compare=False)

    def validate(self) -> None:
        if not self.names:
            raise ConfigError("classes.names ว่างเปล่า ต้องระบุชื่อคลาสตามลำดับ id ใน data.yaml")
        if len(set(self.names)) != len(self.names):
            raise ConfigError(f"classes.names มีชื่อซ้ำกัน: {self.names}")
        if self.unknown_policy not in {"ignore", "weed"}:
            raise ConfigError(
                f"classes.unknown_policy ต้องเป็น 'ignore' หรือ 'weed' (ได้รับ '{self.unknown_policy}')"
            )
        overlap = set(self.target) & set(self.weed)
        if overlap:
            raise ConfigError(f"คลาสเหล่านี้ถูกจัดเป็นทั้งพืชเป้าหมายและวัชพืชพร้อมกัน: {sorted(overlap)}")
        if not self.target:
            raise ConfigError(
                "classes.target ว่างเปล่า ต้องระบุอย่างน้อย 1 คลาส "
                "(ในโหมดผกผัน classes.target คือคลาสเดียวที่ระบบใช้ตัดสินใจทั้งหมด)"
            )
        known = set(self.names)
        if not self._warned:
            for key, values in (("target", self.target), ("weed", self.weed)):
                unknown = [v for v in values if v not in known]
                if unknown:
                    # แค่แจ้งให้ทราบ ไม่ถือเป็นข้อผิดพลาด เพราะ config เผื่อคลาสไว้ล่วงหน้าได้
                    # (เช่น ใส่ 'grape' ไว้รอ แต่ dataset รอบนี้มีแค่ strawberry)
                    print(
                        f"[config] หมายเหตุ: classes.{key} มี {unknown} "
                        f"ซึ่งยังไม่มีในชุดข้อมูลปัจจุบัน -> ข้ามไปก่อน"
                    )
            if self.inverse_mode and self.weed:
                print(
                    "[config] หมายเหตุ: classes.inverse_mode = true -> ผลการตัดสินใจมาจาก "
                    f"classes.target เท่านั้น ส่วน classes.weed {self.weed} ใช้แค่ชี้ตำแหน่ง"
                    " ที่จะพ่น/ยิงในภาพและใน log (ไม่เปลี่ยนว่าจะทำงานหรือไม่)"
                )
            self._warned = True

        for name, value in self.per_class_conf.items():
            if not 0.0 < float(value) < 1.0:
                raise ConfigError(f"classes.per_class_conf['{name}'] ต้องอยู่ระหว่าง 0 ถึง 1")

    def role_of(self, class_name: str) -> str:
        """คืนค่าบทบาทของคลาส: 'target' | 'weed' | 'unknown'"""
        if class_name in self.target:
            return "target"
        if class_name in self.weed:
            return "weed"
        return "weed" if self.unknown_policy == "weed" else "unknown"

    def conf_for(self, class_name: str, default_conf: float) -> float:
        """ค่า confidence ขั้นต่ำของคลาสนั้น (ถ้าไม่ตั้งไว้ ใช้ค่ากลางของโมเดล)"""
        return float(self.per_class_conf.get(class_name, default_conf))


@dataclass
class ColorBand:
    """แถบสีหนึ่งสีที่ถอดค่าแล้วพร้อมใช้งาน (ผลลัพธ์ของ ColorCfg.bands())"""

    class_id: int
    name: str
    hue_ranges: List[tuple]   # [(hue_min, hue_max), ...] — สีแดงคร่อมเลข 0 จึงมี 2 ช่วง
    sat_min: int
    val_min: int


@dataclass
class ColorCfg:
    """ตัวตรวจจับด้วยสีล้วน (HSV Color Detector) — ใช้เมื่อ model.backend: color

    ทำไมต้องมี: prototype-v1 เป็นเวอร์ชันสำหรับ "นำเสนอผลงาน" ที่ต้องสาธิตให้เห็นทันที
    หน้างานโดยไม่ต้องแบกโมเดล/ชุดข้อมูลไปด้วย จึงแทนวัตถุจริงด้วยแผ่นสี:
        สีแดง  = ต้นสตรอว์เบอร์รี (พืชเป้าหมาย) -> ห้ามยิงเลเซอร์
        สีเขียว = วัชพืช                          -> ยิงเลเซอร์
    ตรรกะการตัดสินใจ ROI/ยืนยันหลายเฟรม/กันค้าง ยังเป็นชุดเดียวกับระบบจริงทุกบรรทัด
    เปลี่ยนแค่ "ตาที่ใช้มอง" เท่านั้น (ดู src/detector.py -> ColorDetector)

    ค่า HSV ตามมาตรฐาน OpenCV: H = 0-179, S = 0-255, V = 0-255
    (ต่างจากตำราทั่วไปที่ H = 0-359 — ของ OpenCV หารสอง เพราะเก็บใน uint8)
    """

    red_name: str = "red-object"
    # สีแดงอยู่คร่อมจุดเริ่มต้นของวงล้อสี (hue ~0) จึงต้องใช้สองช่วงเสมอ
    red_hue_ranges: List[List[int]] = field(default_factory=lambda: [[0, 10], [170, 179]])
    red_sat_min: int = 90
    red_val_min: int = 60

    green_name: str = "green-object"
    green_hue_ranges: List[List[int]] = field(default_factory=lambda: [[35, 85]])
    green_sat_min: int = 70
    green_val_min: int = 50

    blur_ksize: int = 5        # เบลอก่อนแปลงเป็น HSV ลด noise ของเซนเซอร์ (0 = ไม่เบลอ)
    morph_ksize: int = 5       # ขนาด kernel ของ open/close ลบจุดรบกวนและอุดรู (0 = ไม่ทำ)
    min_area_frac: float = 0.004   # สัดส่วนพื้นที่กล่องต่อเฟรมขั้นต่ำ (ตัดจุดสีเล็กๆ ทิ้ง)
    max_area_frac: float = 0.90    # สัดส่วนสูงสุด (ตัดกรณีแสงทั้งฉากเป็นสีนั้น)

    def validate(self) -> None:
        if not self.red_name.strip() or not self.green_name.strip():
            raise ConfigError("color.red_name และ color.green_name ห้ามว่างเปล่า")
        if self.red_name == self.green_name:
            raise ConfigError("color.red_name และ color.green_name ต้องไม่ซ้ำกัน")

        for key, ranges in (("red_hue_ranges", self.red_hue_ranges), ("green_hue_ranges", self.green_hue_ranges)):
            if not isinstance(ranges, (list, tuple)) or not ranges:
                raise ConfigError(f"color.{key} ต้องเป็นรายการช่วงสีอย่างน้อย 1 ช่วง เช่น [[35, 85]]")
            for item in ranges:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ConfigError(f"color.{key} แต่ละช่วงต้องเป็น [hue_min, hue_max] (ได้รับ {item!r})")
                try:
                    low, high = int(item[0]), int(item[1])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"color.{key} มีค่าที่ไม่ใช่จำนวนเต็ม: {item!r}") from exc
                if not 0 <= low <= high <= 179:
                    raise ConfigError(
                        f"color.{key} ต้องอยู่ในช่วง 0-179 และ hue_min <= hue_max (ได้รับ {item!r})\n"
                        "  หมายเหตุ: OpenCV ใช้ H 0-179 ไม่ใช่ 0-359"
                    )

        for key in ("red_sat_min", "red_val_min", "green_sat_min", "green_val_min"):
            value = int(getattr(self, key))
            if not 0 <= value <= 255:
                raise ConfigError(f"color.{key} ต้องอยู่ระหว่าง 0 ถึง 255 (ได้รับ {value})")

        for key in ("blur_ksize", "morph_ksize"):
            value = int(getattr(self, key))
            if value < 0:
                raise ConfigError(f"color.{key} ต้องไม่ติดลบ (0 = ปิดการทำงานของขั้นนี้)")
            if value and value % 2 == 0:
                raise ConfigError(f"color.{key} ต้องเป็นเลขคี่ (ข้อกำหนดของ OpenCV) หรือ 0 (ได้รับ {value})")

        if not 0.0 < self.min_area_frac < self.max_area_frac <= 1.0:
            raise ConfigError(
                "color.min_area_frac/max_area_frac ต้องอยู่ระหว่าง 0-1 และ min < max "
                f"(ได้รับ min={self.min_area_frac}, max={self.max_area_frac})"
            )

    def bands(self) -> List[ColorBand]:
        """คืนแถบสีทั้งหมดพร้อมใช้ เรียงตาม class id (0 = แดง, 1 = เขียว)

        ลำดับนี้ต้องตรงกับ classes.names ใน config.yaml เพื่อให้ class_id ที่บันทึกลง
        log สื่อความหมายเดียวกันกับฝั่งโมเดล AI
        """
        return [
            ColorBand(
                class_id=0,
                name=self.red_name,
                hue_ranges=[(int(a), int(b)) for a, b in self.red_hue_ranges],
                sat_min=int(self.red_sat_min),
                val_min=int(self.red_val_min),
            ),
            ColorBand(
                class_id=1,
                name=self.green_name,
                hue_ranges=[(int(a), int(b)) for a, b in self.green_hue_ranges],
                sat_min=int(self.green_sat_min),
                val_min=int(self.green_val_min),
            ),
        ]


@dataclass
class RoiCfg:
    enabled: bool = True
    x1: float = 0.0
    y1: float = 0.45
    x2: float = 1.0
    y2: float = 1.0
    require: str = "center"

    def validate(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"roi.{name} ต้องอยู่ระหว่าง 0.0 ถึง 1.0 (ได้รับ {value})")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ConfigError("roi ต้องมี x1 < x2 และ y1 < y2")
        if self.require not in {"center", "overlap"}:
            raise ConfigError(f"roi.require ต้องเป็น 'center' หรือ 'overlap' (ได้รับ '{self.require}')")

    def pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """แปลง ROI จากสัดส่วน 0-1 เป็นพิกัดพิกเซลจริงของเฟรม"""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height),
        )


@dataclass
class GeomFilterCfg:
    """ตัวกรองรูปทรงกล่องเสริม — ตัด detection ที่ aspect ratio/ขนาดผิดปกติชัดๆ ทิ้ง

    เป็นเลเยอร์เสริมลด false positive เท่านั้น ไม่ใช่ทางแก้ที่ต้นตอ (ดูคอมเมนต์ใน
    config.yaml -> geometry_filter สำหรับที่มาของตัวเลขและข้อจำกัด)

    ค่า default ของคลาสนี้ตั้ง enabled=False โดยตั้งใจ (ปิดไว้ก่อน) เพราะขอบเขต
    min/max_area_frac คำนวณจากสัดส่วนกล่องเทียบกับ "เฟรมกล้องจริง" (640x480+) —
    โค้ดที่สร้าง SprayController ตรงๆ โดยไม่ผ่าน config.yaml (เช่น เทสต์ที่ใช้กล่อง
    สังเคราะห์ขนาดเล็กทดสอบ logic ตำแหน่ง/ROI) จะไม่ได้ตั้งใจให้ผ่านตัวกรองนี้
    ระบบจริงเปิดใช้งานผ่าน config.yaml -> geometry_filter.enabled: true แทน
    """

    enabled: bool = False
    min_aspect: float = 0.26
    max_aspect: float = 2.15
    min_area_frac: float = 0.075
    max_area_frac: float = 0.511

    def validate(self) -> None:
        if self.min_aspect <= 0 or self.min_aspect >= self.max_aspect:
            raise ConfigError("geometry_filter.min_aspect ต้องมากกว่า 0 และน้อยกว่า max_aspect")
        if not 0.0 < self.min_area_frac < self.max_area_frac <= 1.0:
            raise ConfigError(
                "geometry_filter.min_area_frac/max_area_frac ต้องอยู่ระหว่าง 0-1 และ min < max"
            )


@dataclass
class SprayCfg:
    confirm_frames: int = 3
    release_frames: int = 5
    min_on_seconds: float = 0.6
    max_on_seconds: float = 5.0
    cooldown_seconds: float = 1.0
    protect_crop: bool = True
    protect_margin_px: int = 20

    def validate(self) -> None:
        if self.confirm_frames < 1:
            raise ConfigError("spray.confirm_frames ต้องมีค่าอย่างน้อย 1")
        if self.release_frames < 1:
            raise ConfigError("spray.release_frames ต้องมีค่าอย่างน้อย 1")
        if self.min_on_seconds < 0:
            raise ConfigError("spray.min_on_seconds ต้องไม่ติดลบ")
        if self.max_on_seconds <= 0:
            raise ConfigError("spray.max_on_seconds ต้องมากกว่า 0 (เป็นระบบตัดการพ่นค้าง)")
        if self.max_on_seconds < self.min_on_seconds:
            raise ConfigError(
                f"spray.max_on_seconds ({self.max_on_seconds}) ต้องไม่น้อยกว่า "
                f"spray.min_on_seconds ({self.min_on_seconds})"
            )
        if self.cooldown_seconds < 0:
            raise ConfigError("spray.cooldown_seconds ต้องไม่ติดลบ")


# อุปกรณ์ที่รีเลย์ไปต่ออยู่ — มีผลกับ "ข้อความที่แสดง" เท่านั้น ไม่เปลี่ยนตรรกะหรือสัญญาณ GPIO
ACTUATORS = ("pump", "laser", "led")


@dataclass
class RelayCfg:
    backend: str = "auto"
    pin: int = 17
    active_high: bool = False

    # actuator: pump  -> ปั๊มพ่นยา (ระบบจริง)
    #           laser -> เลเซอร์กำจัดวัชพืช (prototype-v1 ที่ใช้นำเสนอผลงาน)
    # เปลี่ยนแค่คำที่แสดงบนหน้าจอ/คอนโซล ("ปั๊ม/พ่น" <-> "เลเซอร์/ยิง") เพื่อให้ภาพที่ผู้ชม
    # เห็นตรงกับอุปกรณ์จริงที่ต่ออยู่ — ตรรกะการตัดสินใจ, สัญญาณ GPIO และคอลัมน์ในไฟล์ CSV
    # ยังเหมือนเดิมทุกประการ (คอลัมน์ pump/เหตุการณ์ SPRAY_START ถูกเก็บไว้เพื่อให้เครื่องมือ
    # วิเคราะห์ผลเดิม เช่น tools/report_events.py อ่านไฟล์เก่ากับไฟล์ใหม่ด้วยสคริปต์เดียวกันได้)
    actuator: str = "pump"

    # power: ความแรงของอุปกรณ์ 0.0-1.0 (ใช้ได้เฉพาะ backend gpio/gpiozero)
    #   1.0        = เปิด-ปิดธรรมดา (ค่าเริ่มต้น ไม่เปลี่ยนพฤติกรรมเดิม)
    #   น้อยกว่า 1  = ขับด้วย PWM ให้กำลังเฉลี่ยลดลง
    # มีไว้สำหรับเลเซอร์เป็นหลัก: ลดกำลังแสงตอนสาธิตในห้องให้ตาปลอดภัยขึ้น โดยยังเห็นจุดแดงชัด
    # ⚠️ ไม่ได้แปลว่า "ปลอดภัยแล้ว" — ลดกำลังเฉลี่ยเท่านั้น ยังห้ามมองลำแสงตรงๆ อยู่ดี
    # ปั๊มน้ำผ่านรีเลย์ใช้ค่านี้ไม่ได้ (รีเลย์เป็นสวิตช์กลไก PWM จะทำให้หน้าสัมผัสพัง) -> ตั้ง 1.0
    power: float = 1.0

    def validate(self) -> None:
        allowed = {"auto", "gpio", "gpiozero", "rpigpio", "mock"}
        if self.backend not in allowed:
            raise ConfigError(
                f"relay.backend ต้องเป็นหนึ่งใน {sorted(allowed)} (ได้รับ '{self.backend}')"
            )
        if not 0 <= self.pin <= 27:
            raise ConfigError(f"relay.pin ต้องเป็นเลข GPIO แบบ BCM ระหว่าง 0-27 (ได้รับ {self.pin})")
        if self.actuator not in ACTUATORS:
            raise ConfigError(
                f"relay.actuator ต้องเป็นหนึ่งใน {sorted(ACTUATORS)} (ได้รับ '{self.actuator}')"
            )
        if not 0.0 < self.power <= 1.0:
            raise ConfigError(
                f"relay.power ต้องมากกว่า 0 และไม่เกิน 1.0 (ได้รับ {self.power})\n"
                "  1.0 = เปิด-ปิดธรรมดา | น้อยกว่านั้น = ขับด้วย PWM ลดกำลัง"
            )
        if self.power < 1.0 and self.backend in {"rpigpio", "mock"}:
            raise ConfigError(
                f"relay.power = {self.power} ใช้กับ relay.backend '{self.backend}' ไม่ได้\n"
                "  การลดกำลังด้วย PWM รองรับเฉพาะ backend 'gpio' หรือ 'gpiozero' เท่านั้น"
            )


@dataclass
class RuntimeCfg:
    show_window: bool = True
    save_video: str = ""
    log_csv: str = "logs/events.csv"
    snapshot_dir: str = "logs/snapshots"
    save_snapshots: bool = True
    print_every: int = 30

    def validate(self) -> None:
        if self.print_every < 0:
            raise ConfigError("runtime.print_every ต้องไม่ติดลบ")


@dataclass
class TrainCfg:
    data: str = "datasets/data.yaml"
    base_model: str = "yolov8n.pt"
    epochs: int = 120
    batch: int = 16
    imgsz: int = 320
    patience: int = 30
    device: str = ""
    project: str = "runs/train"
    name: str = "yolov8n_320"

    def validate(self) -> None:
        if self.epochs < 1:
            raise ConfigError("train.epochs ต้องมีค่าอย่างน้อย 1")
        if self.batch == 0:
            raise ConfigError("train.batch ต้องไม่เป็น 0 (ใช้ -1 เพื่อให้เลือกอัตโนมัติ)")


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    roboflow: RoboflowCfg = field(default_factory=RoboflowCfg)
    classes: ClassCfg = field(default_factory=ClassCfg)
    color: ColorCfg = field(default_factory=ColorCfg)
    roi: RoiCfg = field(default_factory=RoiCfg)
    geometry_filter: GeomFilterCfg = field(default_factory=GeomFilterCfg)
    spray: SprayCfg = field(default_factory=SprayCfg)
    relay: RelayCfg = field(default_factory=RelayCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    # path ของไฟล์ config ที่โหลดมา ใช้แปลง path แบบสัมพัทธ์ให้เป็น absolute
    source_path: Path = field(default=DEFAULT_CONFIG_PATH)

    def validate(self) -> None:
        for f in fields(self):
            section = getattr(self, f.name)
            if hasattr(section, "validate"):
                section.validate()

    def resolve_path(self, value: str) -> Path:
        """แปลง path ในไฟล์ config ให้อ้างอิงจากรากโปรเจกต์เสมอ

        ทำให้รันจากโฟลเดอร์ไหนก็ได้ผลเหมือนกัน
        """
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()


# ---------------------------------------------------------------------------
# การโหลด
# ---------------------------------------------------------------------------


def _build_section(cls, raw: Any, section_name: str):
    """สร้าง dataclass หนึ่งหมวดจาก dict พร้อมฟ้องถ้ามีคีย์แปลกปลอม"""
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"หมวด '{section_name}' ต้องเป็น mapping (key: value)")

    valid_keys = {f.name for f in fields(cls)}
    unknown = set(raw) - valid_keys
    if unknown:
        raise ConfigError(
            f"หมวด '{section_name}' มีคีย์ที่ไม่รู้จัก: {sorted(unknown)}\n"
            f"  คีย์ที่ใช้ได้: {sorted(valid_keys)}"
        )

    # แปลงชนิดข้อมูลให้ตรงกับที่ dataclass ประกาศไว้ (YAML บางทีอ่าน 1 เป็น int แต่เราต้องการ float)
    kwargs = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        try:
            if f.type in ("int", int):
                value = int(value)
            elif f.type in ("float", float):
                value = float(value)
            elif f.type in ("bool", bool):
                value = bool(value)
            elif f.type in ("str", str):
                value = "" if value is None else str(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{section_name}.{f.name} มีค่าไม่ถูกต้อง: {value!r} ({exc})") from exc
        kwargs[f.name] = value
    return cls(**kwargs)


def _load_dotenv_if_present(root: Path) -> None:
    """โหลดตัวแปรจากไฟล์ .env ที่รากโปรเจกต์ (ถ้ามี) เข้า os.environ

    เขียนแบบง่ายๆ เองแทนที่จะเพิ่มไลบรารี python-dotenv สำหรับแค่ฟีเจอร์เดียวนี้
    ไม่ทับตัวแปรที่ตั้งไว้แล้วในเชลล์ ทำให้ยังสั่ง set ชั่วคราวแทนค่าใน .env ได้เสมอ
    ไฟล์ .env ถูกกันไว้ใน .gitignore แล้ว — ใช้เก็บ ROBOFLOW_API_KEY เป็นต้น
    """
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | Path | None = None) -> Config:
    """อ่าน config.yaml แล้วคืนอ็อบเจกต์ Config ที่ตรวจสอบความถูกต้องแล้ว"""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - ขึ้นกับสภาพแวดล้อม
        raise ConfigError(
            "ไม่พบไลบรารี PyYAML — ติดตั้งด้วยคำสั่ง:  pip install pyyaml"
        ) from exc

    _load_dotenv_if_present(PROJECT_ROOT)

    cfg_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"ไม่พบไฟล์ตั้งค่า: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} ต้องเป็น mapping ที่ระดับบนสุด")

    section_types = {f.name: f.type for f in fields(Config) if f.name != "source_path"}
    unknown_sections = set(raw) - set(section_types)
    if unknown_sections:
        raise ConfigError(
            f"พบหมวดที่ไม่รู้จักใน {cfg_path.name}: {sorted(unknown_sections)}\n"
            f"  หมวดที่ใช้ได้: {sorted(section_types)}"
        )

    cfg = Config(
        camera=_build_section(CameraCfg, raw.get("camera"), "camera"),
        model=_build_section(ModelCfg, raw.get("model"), "model"),
        roboflow=_build_section(RoboflowCfg, raw.get("roboflow"), "roboflow"),
        classes=_build_section(ClassCfg, raw.get("classes"), "classes"),
        color=_build_section(ColorCfg, raw.get("color"), "color"),
        roi=_build_section(RoiCfg, raw.get("roi"), "roi"),
        geometry_filter=_build_section(GeomFilterCfg, raw.get("geometry_filter"), "geometry_filter"),
        spray=_build_section(SprayCfg, raw.get("spray"), "spray"),
        relay=_build_section(RelayCfg, raw.get("relay"), "relay"),
        runtime=_build_section(RuntimeCfg, raw.get("runtime"), "runtime"),
        train=_build_section(TrainCfg, raw.get("train"), "train"),
        source_path=cfg_path.resolve(),
    )
    cfg.validate()
    return cfg
