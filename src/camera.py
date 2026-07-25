"""
ส่วนรับภาพ (The Vision) — รองรับหลายแหล่งภาพด้วยอินเทอร์เฟซเดียว

  WebcamSource      : กล้อง USB / เว็บแคมบน PC          (ตอนพัฒนา)
  PiCameraSource    : Pi Camera OV5647 ผ่าน picamera2   (ตอน deploy จริง)
  VideoFileSource   : ไฟล์วิดีโอ .mp4                    (ทดสอบซ้ำด้วยข้อมูลเดิม)
  ImageFolderSource : โฟลเดอร์รูปภาพ                     (Testbed เลื่อนภาพผ่านหน้ากล้อง)

ทุกตัวมีเมธอด read() ที่คืนภาพ BGR (แบบเดียวกับ OpenCV) หรือ None เมื่อจบ
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .config import CameraCfg
from .vision_utils import imread

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CameraError(Exception):
    """ข้อผิดพลาดที่เกิดจากการเปิดหรืออ่านกล้อง"""


class FrameSource(ABC):
    """แหล่งภาพหนึ่งแหล่ง"""

    def __init__(self, cfg: CameraCfg) -> None:
        self.cfg = cfg
        self.frame_index = 0

    @abstractmethod
    def _read_raw(self) -> Optional[np.ndarray]:
        """อ่านภาพดิบ 1 เฟรม (คลาสลูกไปทำเอง)"""

    def read(self) -> Optional[np.ndarray]:
        """อ่าน 1 เฟรม พร้อมพลิกภาพตามที่ตั้งค่าไว้"""
        frame = self._read_raw()
        if frame is None:
            return None
        self.frame_index += 1
        return self._apply_flips(frame)

    def _apply_flips(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        if self.cfg.flip_horizontal and self.cfg.flip_vertical:
            return cv2.flip(frame, -1)
        if self.cfg.flip_horizontal:
            return cv2.flip(frame, 1)
        if self.cfg.flip_vertical:
            return cv2.flip(frame, 0)
        return frame

    def warmup(self) -> None:
        """ทิ้งเฟรมแรกๆ ให้กล้องปรับ auto-exposure / white balance ก่อนเริ่มงานจริง"""
        for _ in range(max(0, self.cfg.warmup_frames)):
            if self._read_raw() is None:
                break

    def release(self) -> None:
        """คืนทรัพยากร"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class WebcamSource(FrameSource):
    """กล้องเว็บแคม / USB ผ่าน OpenCV"""

    def __init__(self, cfg: CameraCfg) -> None:
        super().__init__(cfg)
        import cv2

        # บน Windows ใช้ backend DirectShow จะเปิดกล้องเร็วกว่าค่า default มาก
        import sys

        if sys.platform == "win32":
            self.capture = cv2.VideoCapture(cfg.index, cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(cfg.index)

        if not self.capture.isOpened():
            raise CameraError(
                f"เปิดกล้องหมายเลข {cfg.index} ไม่ได้\n"
                "  - ตรวจว่ากล้องเสียบอยู่และไม่มีโปรแกรมอื่นใช้งานค้าง\n"
                "  - ลองเปลี่ยน camera.index เป็น 1 หรือ 2 ใน config.yaml"
            )

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.capture.set(cv2.CAP_PROP_FPS, cfg.fps)
        # บัฟเฟอร์เล็กที่สุด = ได้ภาพสดที่สุด สำคัญมากสำหรับงานสั่งพ่นแบบเรียลไทม์
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (cfg.width, cfg.height):
            print(
                f"[camera] หมายเหตุ: กล้องไม่รองรับ {cfg.width}x{cfg.height} "
                f"-> ใช้ {actual_w}x{actual_h} แทน"
            )

    def _read_raw(self) -> Optional[np.ndarray]:
        ok, frame = self.capture.read()
        return frame if ok else None

    def release(self) -> None:
        if getattr(self, "capture", None) is not None:
            self.capture.release()


class PiCameraSource(FrameSource):
    """Pi Camera OV5647 ผ่านไลบรารี picamera2 (ใช้บน Raspberry Pi OS Bookworm ขึ้นไป)"""

    def __init__(self, cfg: CameraCfg) -> None:
        super().__init__(cfg)
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise CameraError(
                "ไม่พบไลบรารี picamera2 — บน Raspberry Pi ติดตั้งด้วย:\n"
                "  sudo apt install -y python3-picamera2\n"
                "  (แล้วสร้าง venv ด้วย --system-site-packages เพื่อให้มองเห็น)"
            ) from exc

        self.picam = Picamera2()
        # ขอภาพเป็น RGB888 แล้วค่อยแปลงเป็น BGR เอง เพื่อให้เข้ากับ OpenCV
        config = self.picam.create_preview_configuration(
            main={"size": (cfg.width, cfg.height), "format": "RGB888"}
        )
        self.picam.configure(config)
        self.picam.start()

    def _read_raw(self) -> Optional[np.ndarray]:
        import cv2

        frame = self.picam.capture_array()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        picam = getattr(self, "picam", None)
        if picam is not None:
            try:
                picam.stop()
                picam.close()
            except Exception:  # pragma: no cover - ปิดกล้องล้มเหลวไม่ควรทำให้โปรแกรมพัง
                pass


class VideoFileSource(FrameSource):
    """อ่านจากไฟล์วิดีโอ — เหมาะกับการทดสอบซ้ำด้วยข้อมูลชุดเดิมทุกครั้ง"""

    def __init__(self, cfg: CameraCfg, path: Path) -> None:
        super().__init__(cfg)
        import cv2

        if not path.is_file():
            raise CameraError(f"ไม่พบไฟล์วิดีโอ: {path}")
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise CameraError(f"เปิดไฟล์วิดีโอไม่ได้ (codec อาจไม่รองรับ): {path}")
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def _read_raw(self) -> Optional[np.ndarray]:
        import cv2

        ok, frame = self.capture.read()
        if not ok:
            if not self.cfg.loop:
                return None
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
            if not ok:
                return None
        return frame

    def release(self) -> None:
        if getattr(self, "capture", None) is not None:
            self.capture.release()


class ImageFolderSource(FrameSource):
    """เลื่อนภาพจากโฟลเดอร์ทีละใบ — ใช้ทำ Testbed Simulation (ขั้นตอนที่ 5)

    ยึดกล้องไว้กับที่แล้วเลื่อนภาพผ่านหน้ากล้อง = จำลองด้วยโฟลเดอร์ภาพได้เลย
    แต่ละภาพจะถูกแสดงค้างไว้ hold_frames เฟรม เพื่อจำลองว่ารถวิ่งผ่านต้นพืช
    """

    def __init__(self, cfg: CameraCfg, folder: Path) -> None:
        super().__init__(cfg)
        if not folder.is_dir():
            raise CameraError(f"ไม่พบโฟลเดอร์รูปภาพ: {folder}")

        self.files = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.files:
            raise CameraError(
                f"ไม่พบไฟล์รูปในโฟลเดอร์ {folder}\n"
                f"  นามสกุลที่รองรับ: {sorted(IMAGE_EXTENSIONS)}"
            )

        self.folder = folder
        self.cursor = 0
        self.hold_counter = 0
        self.current: Optional[np.ndarray] = None
        self.current_name = ""
        print(f"[camera] โหมดโฟลเดอร์ภาพ: พบ {len(self.files)} ไฟล์ใน {folder.name}")

    def _load_next(self) -> Optional[np.ndarray]:
        while self.cursor < len(self.files):
            path = self.files[self.cursor]
            self.cursor += 1
            image = imread(path)
            if image is None:
                print(f"[camera] เตือน: อ่านไฟล์ไม่ได้ ข้ามไป -> {path.name}")
                continue
            self.current_name = path.name
            return image

        if self.cfg.loop and self.files:
            self.cursor = 0
            return self._load_next()
        return None

    def _read_raw(self) -> Optional[np.ndarray]:
        if self.current is None or self.hold_counter <= 0:
            image = self._load_next()
            if image is None:
                return None
            self.current = image
            self.hold_counter = max(1, self.cfg.hold_frames)
        self.hold_counter -= 1
        return self.current.copy()


def open_source(cfg: CameraCfg, project_root: Path | None = None) -> FrameSource:
    """สร้างแหล่งภาพตามที่ตั้งค่าไว้ใน config.yaml"""
    if cfg.source == "webcam":
        return WebcamSource(cfg)
    if cfg.source == "picamera2":
        return PiCameraSource(cfg)

    path = Path(cfg.path).expanduser()
    if not path.is_absolute() and project_root is not None:
        path = (project_root / path).resolve()

    if cfg.source == "video":
        return VideoFileSource(cfg, path)
    if cfg.source == "images":
        return ImageFolderSource(cfg, path)

    raise CameraError(f"ไม่รู้จัก camera.source = '{cfg.source}'")
