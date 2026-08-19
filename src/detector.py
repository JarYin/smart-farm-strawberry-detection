"""
ส่วนตรวจจับวัตถุ (The Brain) — รองรับ 3 เอนจิ้น

  UltralyticsDetector : ใช้บน PC ตอนพัฒนา/ทดสอบ สะดวก แต่ลาก PyTorch มาด้วย (~2 GB)
  TFLiteDetector      : ใช้บน Raspberry Pi 4 ใช้แค่ tflite-runtime (~5 MB) เร็วและเบากว่ามาก
  RoboflowDetector    : เรียกโมเดลบนคลาวด์ผ่าน API ของ Roboflow — ใช้ทดสอบเท่านั้น
                        (ดูคำเตือนเรื่อง latency/อินเทอร์เน็ตในคลาสด้านล่าง)

ทั้งสามตัวคืนค่าเป็น list[Detection] เหมือนกัน โค้ดหลักจึงสลับใช้ได้ทันที
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Config
from .detection import Detection
from .vision_utils import class_aware_nms, letterbox, scale_boxes_back, xywh2xyxy


class DetectorError(Exception):
    """ข้อผิดพลาดที่เกิดจากการโหลดหรือรันโมเดล"""


class BaseDetector(ABC):
    """สัญญาว่าตัวตรวจจับทุกตัวต้องทำอะไรได้บ้าง"""

    def __init__(self, class_names: Sequence[str], conf: float, iou: float, max_det: int) -> None:
        self.class_names = list(class_names)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        # เก็บเวลาที่ใช้ในแต่ละขั้น (มิลลิวินาที) ไว้รายงานผลและทำ benchmark
        self.last_timing_ms: dict[str, float] = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

    @abstractmethod
    def infer(self, frame: np.ndarray) -> list[Detection]:
        """รับภาพ BGR จาก OpenCV คืนรายการวัตถุที่พบ (พิกัดเป็นพิกเซลของภาพต้นฉบับ)"""

    def close(self) -> None:
        """ปิด/คืนทรัพยากร (ตัวที่ไม่ต้องทำอะไรก็ปล่อยว่างไว้ได้)"""

    def name_of(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"id_{class_id}"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# เอนจิ้นที่ 1: ultralytics (สำหรับ PC)
# ---------------------------------------------------------------------------


class UltralyticsDetector(BaseDetector):
    """ใช้ไลบรารี ultralytics โดยตรง รองรับทั้งไฟล์ .pt และ .tflite"""

    def __init__(
        self,
        weights: str | Path,
        class_names: Sequence[str],
        imgsz: int = 320,
        conf: float = 0.45,
        iou: float = 0.45,
        max_det: int = 20,
    ) -> None:
        super().__init__(class_names, conf, iou, max_det)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "ไม่พบไลบรารี ultralytics — ติดตั้งด้วย:  pip install ultralytics\n"
                "  (ถ้าอยู่บน Raspberry Pi ให้ใช้ backend 'tflite' แทน จะเบากว่ามาก)"
            ) from exc

        weights = Path(weights)
        if not weights.exists():
            raise DetectorError(
                f"ไม่พบไฟล์โมเดล: {weights}\n"
                "  ยังไม่ได้เทรน? รัน:  python tools/train.py"
            )

        self.imgsz = int(imgsz)
        self.model = YOLO(str(weights))

        # ชื่อคลาสจากตัวโมเดลคือความจริงที่เชื่อถือได้ที่สุด — เตือนถ้าไม่ตรงกับ config
        model_names = getattr(self.model, "names", None)
        if isinstance(model_names, dict) and model_names:
            ordered = [model_names[k] for k in sorted(model_names)]
            if ordered != self.class_names:
                print(
                    "[detector] เตือน: classes.names ใน config ไม่ตรงกับชื่อคลาสในโมเดล\n"
                    f"  config : {self.class_names}\n"
                    f"  โมเดล  : {ordered}\n"
                    "  -> จะใช้ชื่อจากโมเดลเป็นหลัก กรุณาแก้ config.yaml ให้ตรงกัน"
                )
            self.class_names = ordered

    def infer(self, frame: np.ndarray) -> list[Detection]:
        started = time.perf_counter()
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - started) * 1000.0

        post_started = time.perf_counter()
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), confidence, class_id in zip(xyxy, confs, class_ids):
                detections.append(
                    Detection(
                        class_id=int(class_id),
                        class_name=self.name_of(int(class_id)),
                        confidence=float(confidence),
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    )
                )

        # ultralytics รวมเวลา preprocess/inference ไว้ด้วยกัน แยกออกมาถ้ามีข้อมูล
        speed = getattr(results[0], "speed", None) if results else None
        if isinstance(speed, dict):
            self.last_timing_ms = {
                "preprocess": float(speed.get("preprocess", 0.0)),
                "inference": float(speed.get("inference", inference_ms)),
                "postprocess": float(speed.get("postprocess", 0.0)),
            }
        else:
            self.last_timing_ms = {
                "preprocess": 0.0,
                "inference": inference_ms,
                "postprocess": (time.perf_counter() - post_started) * 1000.0,
            }
        return detections


# ---------------------------------------------------------------------------
# เอนจิ้นที่ 2: TFLite ล้วน (สำหรับ Raspberry Pi)
# ---------------------------------------------------------------------------


class TFLiteDetector(BaseDetector):
    """รันโมเดล .tflite เองทั้งหมด: letterbox -> invoke -> ถอดผลลัพธ์ -> NMS

    รองรับทั้งโมเดล float32 และโมเดล int8 ที่ผ่าน quantization แล้ว
    """

    def __init__(
        self,
        weights: str | Path,
        class_names: Sequence[str],
        imgsz: int = 320,
        conf: float = 0.45,
        iou: float = 0.45,
        max_det: int = 20,
        num_threads: int = 4,
    ) -> None:
        super().__init__(class_names, conf, iou, max_det)

        weights = Path(weights)
        if not weights.exists():
            raise DetectorError(
                f"ไม่พบไฟล์โมเดล: {weights}\n"
                "  ยังไม่ได้แปลงเป็น TFLite? รัน:  python tools/export_tflite.py"
            )

        interpreter_cls = self._load_interpreter_class()
        self.interpreter = interpreter_cls(model_path=str(weights), num_threads=int(num_threads))
        self.interpreter.allocate_tensors()

        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

        # รูปร่างอินพุตของ YOLOv8 TFLite ปกติคือ NHWC = (1, H, W, 3) แต่บาง export
        # (พบจริงกับโมเดลที่ export จาก Colab เดือน 2026-08) ให้เป็น NCHW = (1, 3, H, W)
        # แทน ต้องตรวจให้ถูกทั้งสองแบบ ไม่งั้น set_tensor จะพังด้วย dimension mismatch
        input_shape = self.input_detail["shape"]
        if len(input_shape) == 4 and input_shape[3] == 3:
            self.input_layout = "nhwc"
            self.imgsz = (int(input_shape[1]), int(input_shape[2]))
        elif len(input_shape) == 4 and input_shape[1] == 3:
            self.input_layout = "nchw"
            self.imgsz = (int(input_shape[2]), int(input_shape[3]))
        else:
            print(f"[detector] เตือน: รูปร่างอินพุตผิดจากที่คาด {input_shape} -> ใช้ค่าจาก config แทน")
            self.input_layout = "nhwc"
            self.imgsz = (int(imgsz), int(imgsz))

        if self.imgsz != (int(imgsz), int(imgsz)):
            print(
                f"[detector] หมายเหตุ: โมเดลรับภาพขนาด {self.imgsz} "
                f"แต่ config ตั้ง imgsz = {imgsz} -> ใช้ขนาดของโมเดลเป็นหลัก"
            )

        self.input_dtype = self.input_detail["dtype"]
        self.input_quant = self.input_detail.get("quantization", (0.0, 0))
        self.output_dtype = self.output_detail["dtype"]
        self.output_quant = self.output_detail.get("quantization", (0.0, 0))
        self.is_quantized_input = self.input_dtype in (np.int8, np.uint8)

        print(
            f"[detector] โหลด TFLite สำเร็จ: {weights.name} | "
            f"อินพุต {self.imgsz} {np.dtype(self.input_dtype).name} | "
            f"เอาต์พุต {tuple(self.output_detail['shape'])} {np.dtype(self.output_dtype).name} | "
            f"{num_threads} threads"
        )

    @staticmethod
    def _load_interpreter_class():
        """หา Interpreter จาก tflite_runtime ก่อน ถ้าไม่มีค่อยลอง ai-edge-litert แล้วค่อย TensorFlow เต็มตัว

        tflite_runtime เลิกพัฒนาแล้วและไม่มี wheel สำหรับ Python รุ่นใหม่ (เช่น 3.13 บน
        Raspberry Pi OS Trixie) — Google เปลี่ยนไปดูแลต่อในชื่อ ai-edge-litert (LiteRT) แทน
        """
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore

            return Interpreter
        except ImportError:
            pass
        try:
            from ai_edge_litert.interpreter import Interpreter  # type: ignore

            return Interpreter
        except ImportError:
            pass
        try:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

            return Interpreter
        except ImportError as exc:
            raise DetectorError(
                "ไม่พบตัวรัน TFLite — ติดตั้งอย่างใดอย่างหนึ่ง:\n"
                "  บน Raspberry Pi (Python รุ่นใหม่) :  pip install ai-edge-litert\n"
                "  บน Raspberry Pi (Python รุ่นเก่า) :  pip install tflite-runtime\n"
                "  บน PC (Windows)                  :  pip install tensorflow"
            ) from exc

    # -- ขั้นตอนที่ 1: เตรียมภาพ -------------------------------------------------

    def _preprocess(self, frame: np.ndarray):
        import cv2

        padded, ratio, pad = letterbox(frame, self.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        if self.is_quantized_input:
            scale, zero_point = self.input_quant
            if scale and scale > 0:
                # โมเดล int8 คาดหวังค่าที่ quantize มาจากภาพ float ช่วง 0-1
                normalized = rgb.astype(np.float32) / 255.0
                tensor = np.clip(
                    np.round(normalized / scale + zero_point),
                    np.iinfo(self.input_dtype).min,
                    np.iinfo(self.input_dtype).max,
                ).astype(self.input_dtype)
            else:
                tensor = rgb.astype(self.input_dtype)
        else:
            tensor = (rgb.astype(np.float32) / 255.0).astype(self.input_dtype)

        if self.input_layout == "nchw":
            tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW

        return np.expand_dims(tensor, axis=0), ratio, pad

    # -- ขั้นตอนที่ 3: ถอดผลลัพธ์ -------------------------------------------------

    def _decode(self, raw: np.ndarray, ratio: float, pad, original_shape) -> list[Detection]:
        # ถ้าเป็นโมเดล quantized ต้องแปลงค่ากลับเป็น float ก่อน
        if raw.dtype in (np.int8, np.uint8):
            scale, zero_point = self.output_quant
            if scale and scale > 0:
                raw = (raw.astype(np.float32) - zero_point) * scale
            else:
                raw = raw.astype(np.float32)

        pred = raw[0] if raw.ndim == 3 else raw
        num_classes = len(self.class_names)
        expected = 4 + num_classes

        # YOLOv8 ให้ผลลัพธ์เป็น (4+nc, anchors) แต่บางเวอร์ชันสลับแกน — รองรับทั้งคู่
        if pred.shape[0] == expected:
            pred = pred.T
        elif pred.shape[1] != expected:
            raise DetectorError(
                f"รูปร่างเอาต์พุตของโมเดลไม่ตรงกับจำนวนคลาส\n"
                f"  เอาต์พุต : {pred.shape} (คาดว่าจะมีแกนขนาด {expected} = 4 + {num_classes} คลาส)\n"
                f"  config   : classes.names = {self.class_names}\n"
                "  -> แก้ classes.names ใน config.yaml ให้ตรงกับตอนเทรน"
            )

        boxes_xywh = pred[:, :4].astype(np.float32)
        class_scores = pred[:, 4:].astype(np.float32)
        if class_scores.size == 0:
            return []

        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores.max(axis=1)

        keep_mask = confidences >= self.conf
        if not np.any(keep_mask):
            return []

        boxes_xywh = boxes_xywh[keep_mask]
        class_ids = class_ids[keep_mask]
        confidences = confidences[keep_mask]

        # TFLite ที่ export จาก ultralytics ให้พิกัดแบบ normalize (0-1)
        # แต่บางเส้นทางการ export ให้เป็นพิกเซลเลย — ตรวจจากขนาดค่าสูงสุด
        if float(boxes_xywh.max()) <= 1.5:
            boxes_xywh = boxes_xywh * np.array(
                [self.imgsz[1], self.imgsz[0], self.imgsz[1], self.imgsz[0]], dtype=np.float32
            )

        boxes_xyxy = xywh2xyxy(boxes_xywh)
        keep = class_aware_nms(boxes_xyxy, confidences, class_ids, self.iou, self.max_det)
        if not keep:
            return []

        boxes_xyxy = scale_boxes_back(boxes_xyxy[keep], ratio, pad, original_shape)
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        detections: list[Detection] = []
        for (x1, y1, x2, y2), confidence, class_id in zip(boxes_xyxy, confidences, class_ids):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=self.name_of(int(class_id)),
                    confidence=float(confidence),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
        return detections

    # -- รวมทุกขั้นตอน -----------------------------------------------------------

    def infer(self, frame: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        tensor, ratio, pad = self._preprocess(frame)

        t1 = time.perf_counter()
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_detail["index"])

        t2 = time.perf_counter()
        detections = self._decode(raw, ratio, pad, frame.shape[:2])
        t3 = time.perf_counter()

        self.last_timing_ms = {
            "preprocess": (t1 - t0) * 1000.0,
            "inference": (t2 - t1) * 1000.0,
            "postprocess": (t3 - t2) * 1000.0,
        }
        return detections


# ---------------------------------------------------------------------------
# เอนจิ้นที่ 3: Roboflow hosted inference (คลาวด์)
# ---------------------------------------------------------------------------


class RoboflowDetector(BaseDetector):
    """เรียกใช้โมเดลที่ฝากไว้บน Roboflow ผ่าน API แบบคลาวด์ (Roboflow serverless inference)

    ใช้ตอนที่ยังไม่มีโมเดลของตัวเอง หรืออยากเทียบผลกับโมเดลสำเร็จรูปก่อนตัดสินใจเทรนเอง
    ทำงานเหมือน detector อีก 2 ตัว (คืนค่า list[Detection] เหมือนกัน) จึงเสียบเข้ากับ
    src/main.py, src/overlay.py, src/logic.py ได้ทันทีโดยไม่ต้องแก้อะไรที่อื่น

    ⚠️ ไม่แนะนำให้ใช้ควบคุมปั๊มแบบเรียลไทม์บน Raspberry Pi จริง เพราะ:
      1. ต้องมีอินเทอร์เน็ตเสถียรตลอดเวลา (โรงเรือนหลายแห่งสัญญาณไม่ดี)
      2. แต่ละเฟรมมี latency เครือข่ายเพิ่ม ~0.3-2 วินาที ทำให้ได้ FPS ต่ำกว่า
         เอนจิ้น local (ultralytics/tflite) มาก ไม่ทันต่อการสั่งพ่นขณะรถเคลื่อนที่
      3. บัญชีฟรีของ Roboflow มีโควตาการเรียกใช้จำกัดต่อเดือน
    เหมาะกับการทดสอบภาพนิ่งทีละภาพ (ดู tools/roboflow_infer.py) หรือทดสอบผ่าน
    src/main.py --source images ที่เดินช้าๆ มากกว่าการใช้งานจริงต่อเนื่อง

    การจัดการ API key: รับค่ามาทาง __init__ เท่านั้น ไม่อ่านจากไฟล์ config หรือ env
    เอง — ผู้เรียก (build_detector / tools/roboflow_infer.py) เป็นคนอ่านจาก
    ตัวแปรสภาพแวดล้อมแล้วส่งเข้ามา เพื่อให้จุดที่แตะค่าลับมีที่เดียวและตรวจสอบง่าย
    """

    def __init__(
        self,
        model_id: str,
        api_url: str,
        api_key: str,
        class_names: Sequence[str] = (),
        conf: float = 0.45,
        iou: float = 0.45,
        max_det: int = 20,
        timeout: float = 10.0,
        api_key_env_name: str = "ROBOFLOW_API_KEY",
    ) -> None:
        super().__init__(class_names, conf, iou, max_det)

        if not api_key:
            raise DetectorError(
                "ไม่พบ Roboflow API key\n"
                f"  ตั้งตัวแปรสภาพแวดล้อม {api_key_env_name} ก่อนรัน:\n"
                f"    PowerShell :  $env:{api_key_env_name} = \"คีย์ของคุณ\"\n"
                f"    Bash       :  export {api_key_env_name}=\"คีย์ของคุณ\"\n"
                "  หรือสร้างไฟล์ .env ที่รากโปรเจกต์ (คัดลอกจาก .env.example)\n"
                "  ห้ามเขียนคีย์ลงไฟล์ config.yaml หรือโค้ดโดยตรง เพราะจะหลุดไปกับ git ได้"
            )

        try:
            from inference_sdk import InferenceHTTPClient
        except ImportError as exc:
            raise DetectorError(
                "ไม่พบไลบรารี inference-sdk — ติดตั้งด้วย:  pip install inference-sdk"
            ) from exc

        self.model_id = model_id
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.timeout = float(timeout)
        self._dynamic_names = len(self.class_names) == 0

        print(f"[detector] ใช้ Roboflow hosted inference | model_id={model_id} | api_url={api_url}")
        print("[detector] คำเตือน: เอนจิ้นนี้พึ่งอินเทอร์เน็ตและมี latency สูง ไม่เหมาะกับควบคุมปั๊มเรียลไทม์")
        if self._dynamic_names:
            print("[detector] classes.names ว่างเปล่า -> จะใช้ชื่อคลาสที่ API ส่งกลับมาโดยตรง")

    def _decode(self, predictions: list[dict[str, Any]]) -> list[Detection]:
        """แปลงผลลัพธ์ JSON จาก Roboflow เป็น list[Detection]

        รูปแบบมาตรฐานของ Roboflow object detection ต่อ 1 รายการ:
            {"class": "strawberry", "confidence": 0.91,
             "x": 246.5, "y": 300.5, "width": 60.0, "height": 82.0, ...}
        โดย x,y คือจุดกึ่งกลางกล่อง หน่วยเป็นพิกเซลของภาพต้นฉบับที่ส่งไปตรงๆ
        (ไม่ต้อง letterbox/แปลงพิกัดกลับเหมือน TFLite เพราะ Roboflow จัดการฝั่งเซิร์ฟเวอร์ให้แล้ว)

        แยกเป็นเมธอดต่างหากจาก infer() เพื่อให้ทดสอบได้โดยไม่ต้องเรียก API จริง
        """
        detections: list[Detection] = []
        for pred in predictions:
            try:
                confidence = float(pred["confidence"])
                cx, cy = float(pred["x"]), float(pred["y"])
                width, height = float(pred["width"]), float(pred["height"])
            except (KeyError, TypeError, ValueError):
                # ข้ามรายการที่รูปแบบผิดปกติ ไม่ให้ช่องเดียวทำทั้งเฟรมพัง
                continue
            if confidence < self.conf:
                continue

            class_name = pred.get("class") or pred.get("class_name") or f"id_{pred.get('class_id', '?')}"
            half_w, half_h = width / 2.0, height / 2.0
            detections.append(
                Detection(
                    class_id=int(pred.get("class_id", -1)),
                    class_name=str(class_name),
                    confidence=confidence,
                    x1=cx - half_w,
                    y1=cy - half_h,
                    x2=cx + half_w,
                    y2=cy + half_h,
                )
            )
        return detections

    def infer(self, frame: np.ndarray) -> list[Detection]:
        import cv2
        from PIL import Image

        t0 = time.perf_counter()
        # แปลง BGR (OpenCV) -> RGB ก่อนส่ง เพราะโมเดลถูกเทรนด้วยภาพ RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        t1 = time.perf_counter()

        try:
            result = self.client.infer(pil_image, model_id=self.model_id)
        except Exception as exc:
            raise DetectorError(f"เรียก Roboflow API ไม่สำเร็จ: {exc}") from exc
        t2 = time.perf_counter()

        predictions = result.get("predictions", []) if isinstance(result, dict) else []
        detections = self._decode(predictions)
        t3 = time.perf_counter()

        self.last_timing_ms = {
            "preprocess": (t1 - t0) * 1000.0,
            # ส่วนใหญ่ของเวลานี้คือรอเครือข่าย ไม่ใช่เวลาประมวลผลจริงของโมเดล
            "inference": (t2 - t1) * 1000.0,
            "postprocess": (t3 - t2) * 1000.0,
        }
        return detections


# ---------------------------------------------------------------------------
# ตัวช่วยสร้าง detector ตาม config
# ---------------------------------------------------------------------------


def effective_conf(cfg: Config) -> float:
    """ค่า confidence ที่ควรใช้กรองในชั้น detector

    ถ้า config ตั้ง per_class_conf ของบางคลาสไว้ "ต่ำกว่า" ค่ากลาง แล้ว detector
    ยังกรองด้วยค่ากลางอยู่ วัตถุคลาสนั้นจะถูกทิ้งตั้งแต่ต้นทาง ไม่มีวันไปถึงชั้นตรรกะ
    จึงต้องกรองด้วยค่าที่ "ต่ำที่สุด" ไว้ก่อน แล้วปล่อยให้ชั้นตรรกะบังคับเกณฑ์
    ที่เข้มกว่าของแต่ละคลาสอีกที (ดู logic.SprayController)
    """
    values = [cfg.model.conf, *(float(v) for v in cfg.classes.per_class_conf.values())]
    return max(0.01, min(values))


def build_detector(cfg: Config) -> BaseDetector:
    """สร้างตัวตรวจจับตามที่ตั้งค่าไว้ใน config.yaml"""
    backend = cfg.model.resolved_backend()
    conf = effective_conf(cfg)

    if backend == "roboflow":
        # จุดเดียวที่อ่าน API key จากตัวแปรสภาพแวดล้อม — ไม่มีที่อื่นในโปรเจกต์แตะค่านี้
        api_key = os.environ.get(cfg.roboflow.api_key_env, "")
        return RoboflowDetector(
            model_id=cfg.roboflow.model_id,
            api_url=cfg.roboflow.api_url,
            api_key=api_key,
            class_names=cfg.classes.names,
            conf=conf,
            iou=cfg.model.iou,
            max_det=cfg.model.max_det,
            timeout=cfg.roboflow.timeout_seconds,
            api_key_env_name=cfg.roboflow.api_key_env,
        )

    weights = cfg.resolve_path(cfg.model.weights)

    if backend == "tflite":
        return TFLiteDetector(
            weights=weights,
            class_names=cfg.classes.names,
            imgsz=cfg.model.imgsz,
            conf=conf,
            iou=cfg.model.iou,
            max_det=cfg.model.max_det,
            num_threads=cfg.model.num_threads,
        )
    return UltralyticsDetector(
        weights=weights,
        class_names=cfg.classes.names,
        imgsz=cfg.model.imgsz,
        conf=conf,
        iou=cfg.model.iou,
        max_det=cfg.model.max_det,
    )
