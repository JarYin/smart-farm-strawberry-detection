"""
ตรรกะการตัดสินใจพ่นยา — หัวใจของงานวิจัยนี้

ถ้าเขียนแบบง่ายที่สุดคือ "เจอวัชพืช = เปิดปั๊ม" ซึ่งใช้งานจริงไม่ได้ เพราะ:
  - AI อาจตรวจผิดแค่เฟรมเดียวจากภาพเบลอ (Motion Blur) แล้วปั๊มจะกระตุกทันที
  - หัวฉีดครอบคลุมแค่บริเวณหนึ่งของภาพ ไม่ใช่ทั้งเฟรม
  - ถ้าวัชพืชขึ้นชิดต้นสตรอว์เบอร์รี การพ่นจะทำให้พืชหลักเสียหาย
  - ถ้าโปรแกรมค้าง ปั๊มต้องไม่เปิดค้างจนยาหมดถัง

โมดูลนี้จึงแยกออกมาเป็นตรรกะบริสุทธิ์ (ไม่ยุ่งกับกล้อง/GPIO เลย)
รับนาฬิกาเข้ามาทางพารามิเตอร์ได้ จึงเขียนเทสต์ครอบคลุมได้ทั้งหมดโดยไม่ต้องรอเวลาจริง
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .config import ClassCfg, GeomFilterCfg, RoiCfg, SprayCfg
from .detection import Detection
from .vision_utils import boxes_overlap, expand_box, plausible_box_geometry, point_in_box

# สถานะของเครื่องสถานะ
STATE_IDLE = "IDLE"
STATE_SPRAYING = "SPRAYING"
STATE_COOLDOWN = "COOLDOWN"


@dataclass
class FrameAnalysis:
    """ผลการจำแนกวัตถุในเฟรมหนึ่งๆ ก่อนเข้าเครื่องสถานะ"""

    targets: list[Detection] = field(default_factory=list)
    """พืชเป้าหมายที่ผ่านเกณฑ์ความเชื่อมั่น"""

    weeds: list[Detection] = field(default_factory=list)
    """วัชพืชทั้งหมดที่ผ่านเกณฑ์ความเชื่อมั่น (ยังไม่กรอง ROI)"""

    actionable_weeds: list[Detection] = field(default_factory=list)
    """วัชพืชที่ "สมควรพ่น" จริง — อยู่ในระยะหัวฉีดและไม่ติดพืชหลัก"""

    suppressed_weeds: list[tuple[Detection, str]] = field(default_factory=list)
    """วัชพืชที่ถูกระงับ พร้อมเหตุผล"""

    rejected: list[tuple[Detection, str]] = field(default_factory=list)
    """วัตถุที่ถูกตัดทิ้ง (ความเชื่อมั่นต่ำ หรือเป็นคลาสที่ไม่สนใจ)"""

    @property
    def has_actionable_weed(self) -> bool:
        return bool(self.actionable_weeds)

    @property
    def best_weed(self) -> Detection | None:
        """วัชพืชที่มั่นใจที่สุดในเฟรมนี้ (ใช้บันทึก log)"""
        if not self.actionable_weeds:
            return None
        return max(self.actionable_weeds, key=lambda d: d.confidence)


@dataclass
class Decision:
    """คำตัดสินของเฟรมหนึ่งๆ"""

    spray_on: bool
    state: str
    reason: str
    analysis: FrameAnalysis
    changed: bool = False          # สถานะปั๊มเปลี่ยนในเฟรมนี้หรือไม่
    hit_streak: int = 0            # จำนวนเฟรมติดกันที่เจอวัชพืช
    miss_streak: int = 0           # จำนวนเฟรมติดกันที่ไม่เจอวัชพืช
    spray_elapsed: float = 0.0     # พ่นมาแล้วกี่วินาที (ถ้ากำลังพ่น)


class SprayController:
    """เครื่องสถานะควบคุมการพ่น: IDLE -> SPRAYING -> COOLDOWN -> IDLE"""

    def __init__(
        self,
        spray_cfg: SprayCfg,
        roi_cfg: RoiCfg,
        class_cfg: ClassCfg,
        base_conf: float = 0.45,
        clock: Callable[[], float] = time.monotonic,
        geom_cfg: GeomFilterCfg | None = None,
    ) -> None:
        self.spray = spray_cfg
        self.roi = roi_cfg
        self.classes = class_cfg
        self.base_conf = float(base_conf)
        self.clock = clock
        self.geom = geom_cfg or GeomFilterCfg()

        self.state = STATE_IDLE
        self.hit_streak = 0
        self.miss_streak = 0
        self._spray_started_at: float | None = None
        self._spray_stopped_at: float | None = None

        # สถิติสะสม ใช้สรุปผลการทดลอง
        self.frames_processed = 0
        self.frames_with_weed = 0
        self.spray_events = 0
        self.total_spray_seconds = 0.0
        self.cutoff_events = 0  # จำนวนครั้งที่ถูกตัดเพราะพ่นนานเกิน max_on_seconds

    # -- ขั้นที่ 1: จำแนกวัตถุในเฟรม -------------------------------------------

    def analyze(self, detections: Sequence[Detection], frame_shape: tuple[int, int]) -> FrameAnalysis:
        """แยกวัตถุที่ตรวจพบเป็น พืชเป้าหมาย / วัชพืชที่ควรพ่น / สิ่งที่ถูกระงับ

        frame_shape : (height, width) ของเฟรม ใช้คำนวณ ROI เป็นพิกเซล
        """
        height, width = frame_shape
        analysis = FrameAnalysis()

        roi_box = self.roi.pixel_box(width, height) if self.roi.enabled else None

        # 1.1 กรองด้วยเกณฑ์ความเชื่อมั่นเฉพาะคลาส แล้วแยกตามบทบาท
        for det in detections:
            required_conf = self.classes.conf_for(det.class_name, self.base_conf)
            if det.confidence < required_conf:
                analysis.rejected.append((det, f"ความเชื่อมั่น {det.confidence:.2f} < {required_conf:.2f}"))
                continue

            if self.geom.enabled and not plausible_box_geometry(
                det.x2 - det.x1,
                det.y2 - det.y1,
                frame_shape,
                self.geom.min_aspect,
                self.geom.max_aspect,
                self.geom.min_area_frac,
                self.geom.max_area_frac,
            ):
                aspect = (det.x2 - det.x1) / max(det.y2 - det.y1, 1e-9)
                area_frac = ((det.x2 - det.x1) * (det.y2 - det.y1)) / max(width * height, 1)
                analysis.rejected.append(
                    (det, f"รูปทรงกล่องผิดปกติ (aspect={aspect:.2f}, area_frac={area_frac:.3f})")
                )
                continue

            role = self.classes.role_of(det.class_name)
            if role == "target":
                analysis.targets.append(det)
            elif role == "weed":
                analysis.weeds.append(det)
            else:
                analysis.rejected.append((det, "คลาสที่ไม่อยู่ในรายการเป้าหมายหรือวัชพืช"))

        if self.classes.inverse_mode:
            self._analyze_inverse(analysis, roi_box, width, height)
        else:
            self._analyze_explicit_weed(analysis, roi_box)

        return analysis

    def _analyze_explicit_weed(self, analysis: FrameAnalysis, roi_box) -> None:
        """ตรรกะปกติ: ต้องมีคลาสวัชพืชในโมเดลจริงๆ ถึงจะพ่น กรองเป็นรายกล่อง

        ตรวจ ROI และระยะห่างจากพืชเป้าหมายทีละกล่อง — แม่นยำกว่าโหมดผกผัน
        แต่ใช้การคำนวณมากกว่า (ขยายกล่อง + เช็คซ้อนทับต่อคู่วัชพืช-พืชเป้าหมาย)
        """
        protected_boxes = []
        if self.spray.protect_crop:
            margin = float(self.spray.protect_margin_px)
            protected_boxes = [expand_box(t.box, margin) for t in analysis.targets]

        for weed in analysis.weeds:
            if roi_box is not None and not self._in_roi(weed, roi_box):
                analysis.suppressed_weeds.append((weed, "อยู่นอกระยะหัวฉีด (ROI)"))
                continue

            if protected_boxes and any(boxes_overlap(weed.box, box) for box in protected_boxes):
                analysis.suppressed_weeds.append((weed, "อยู่ชิดพืชเป้าหมาย งดพ่นเพื่อไม่ให้พืชหลักเสียหาย"))
                continue

            analysis.actionable_weeds.append(weed)

    def _analyze_inverse(self, analysis: FrameAnalysis, roi_box, width: int, height: int) -> None:
        """โหมดผกผัน (Inverse / Crop-vs-Non-crop): ไม่มีคลาสวัชพืชในโมเดล

        ถือว่า "สิ่งที่ไม่ใช่พืชเป้าหมาย" ทั้งหมดคือสิ่งที่ต้องพ่น ตรวจแค่ว่ามีพืช
        เป้าหมายอยู่ในเขตพ่น (ROI) หรือไม่ — เช็คบูลีนครั้งเดียวต่อเฟรม ไม่ต้องคำนวณ
        ระยะห่างหรือ IoU เป็นรายคู่กล่องเหมือนโหมดปกติ เบากว่ามากบนบอร์ดที่ทรัพยากรจำกัด
        เช่น Raspberry Pi 4

        ข้อควรระวัง: จะพ่นโดนดินเปล่า/พื้นที่ว่างที่ไม่มีทั้งพืชหลักและวัชพืชด้วย
        เพราะระบบแยกไม่ออกระหว่าง "วัชพืช" กับ "ไม่มีอะไรเลย" — เป็นการยอมแลก
        ความละเอียดเพื่อความเร็วและไม่ต้องมีชุดข้อมูลวัชพืชแยกต่างหาก
        """
        zone_box = roi_box if roi_box is not None else (0, 0, width, height)
        target_in_zone = any(self._in_roi(t, zone_box) for t in analysis.targets)

        # สร้างกล่องสังเคราะห์ครอบคลุมทั้งเขตพ่น เพื่อให้ไหลผ่าน pipeline เดิม
        # (เครื่องสถานะ, การวาดกรอบ, การบันทึก log) ได้เหมือนกล่องวัชพืชจริง
        zone_detection = Detection(
            class_id=-1,
            class_name="non-crop-area",
            confidence=1.0,
            x1=float(zone_box[0]),
            y1=float(zone_box[1]),
            x2=float(zone_box[2]),
            y2=float(zone_box[3]),
        )

        if target_in_zone:
            analysis.suppressed_weeds.append(
                (zone_detection, "พบพืชเป้าหมายในเขตพ่น งดพ่นทั้งเขตเพื่อป้องกันพืชหลัก (โหมดผกผัน)")
            )
        else:
            analysis.actionable_weeds.append(zone_detection)

    def _in_roi(self, det: Detection, roi_box) -> bool:
        if self.roi.require == "center":
            cx, cy = det.center
            return point_in_box(cx, cy, roi_box)
        return boxes_overlap(det.box, roi_box)

    # -- ขั้นที่ 2: เครื่องสถานะ -------------------------------------------------

    def update(
        self,
        detections: Sequence[Detection],
        frame_shape: tuple[int, int],
        now: float | None = None,
    ) -> Decision:
        """ประมวลผล 1 เฟรม แล้วคืนคำตัดสินว่าปั๊มควรเปิดหรือปิด"""
        now = self.clock() if now is None else float(now)
        analysis = self.analyze(detections, frame_shape)

        self.frames_processed += 1
        weed_present = analysis.has_actionable_weed
        if weed_present:
            self.frames_with_weed += 1
            self.hit_streak += 1
            self.miss_streak = 0
        else:
            self.miss_streak += 1
            self.hit_streak = 0

        was_on = self.state == STATE_SPRAYING
        reason = ""

        # --- อยู่ในช่วงพัก: รอจนครบเวลาแล้วค่อยกลับไปพร้อมทำงาน ---
        if self.state == STATE_COOLDOWN:
            # ต้องเทียบกับ None ตรงๆ ห้ามใช้ `or` เพราะเวลา 0.0 วินาทีเป็นค่าเท็จใน Python
            # จะทำให้เวลาที่ผ่านไปถูกคำนวณผิดเป็น 0 เสมอในเฟรมแรกของการทำงาน
            stopped_at = now if self._spray_stopped_at is None else self._spray_stopped_at
            if now - stopped_at >= self.spray.cooldown_seconds:
                self.state = STATE_IDLE
            else:
                remaining = self.spray.cooldown_seconds - (now - stopped_at)
                reason = f"อยู่ในช่วงพักหลังพ่น เหลืออีก {remaining:.1f} วินาที"

        # --- ว่างอยู่: เจอวัชพืชติดกันครบตามกำหนดหรือยัง ---
        if self.state == STATE_IDLE:
            if self.hit_streak >= self.spray.confirm_frames:
                self.state = STATE_SPRAYING
                self._spray_started_at = now
                self.spray_events += 1
                best = analysis.best_weed
                detail = f" ({best.class_name} {best.confidence:.2f})" if best else ""
                reason = f"ยืนยันพบวัชพืชครบ {self.hit_streak} เฟรมติดกัน{detail} -> เริ่มพ่น"
            elif not reason:
                if weed_present:
                    reason = f"กำลังยืนยัน {self.hit_streak}/{self.spray.confirm_frames} เฟรม"
                elif analysis.suppressed_weeds:
                    reason = analysis.suppressed_weeds[0][1]
                elif analysis.targets:
                    names = ", ".join(sorted({t.class_name for t in analysis.targets}))
                    reason = f"พบพืชเป้าหมาย ({names}) -> ปล่อยผ่าน"
                else:
                    reason = "ไม่พบวัตถุที่ต้องดำเนินการ"

        # --- กำลังพ่น: ตรวจเงื่อนไขหยุด ---
        elif self.state == STATE_SPRAYING:
            started_at = now if self._spray_started_at is None else self._spray_started_at
            elapsed = now - started_at

            if elapsed >= self.spray.max_on_seconds:
                self._stop_spraying(now, elapsed)
                self.cutoff_events += 1
                reason = (
                    f"ตัดการพ่นอัตโนมัติ: พ่นต่อเนื่องครบ {self.spray.max_on_seconds:.1f} วินาที "
                    "(ระบบกันปั๊มค้าง)"
                )
            elif self.miss_streak >= self.spray.release_frames and elapsed >= self.spray.min_on_seconds:
                self._stop_spraying(now, elapsed)
                reason = f"ไม่พบวัชพืชติดกัน {self.miss_streak} เฟรม -> หยุดพ่น"
            elif weed_present:
                reason = f"กำลังพ่น {elapsed:.1f} วินาที (ยังพบวัชพืช {len(analysis.actionable_weeds)} จุด)"
            elif elapsed < self.spray.min_on_seconds:
                reason = f"กำลังพ่น {elapsed:.1f} วินาที (ยังไม่ถึงเวลาพ่นขั้นต่ำ)"
            else:
                reason = f"กำลังพ่น {elapsed:.1f} วินาที (รอครบ {self.spray.release_frames} เฟรมก่อนหยุด)"

        is_on = self.state == STATE_SPRAYING
        spray_elapsed = (
            now - self._spray_started_at if (is_on and self._spray_started_at is not None) else 0.0
        )

        return Decision(
            spray_on=is_on,
            state=self.state,
            reason=reason,
            analysis=analysis,
            changed=(is_on != was_on),
            hit_streak=self.hit_streak,
            miss_streak=self.miss_streak,
            spray_elapsed=spray_elapsed,
        )

    def _stop_spraying(self, now: float, elapsed: float) -> None:
        self.state = STATE_COOLDOWN
        self._spray_stopped_at = now
        self._spray_started_at = None
        self.total_spray_seconds += elapsed

    # -- ส่วนเสริม ---------------------------------------------------------------

    def force_stop(self, now: float | None = None) -> None:
        """สั่งหยุดพ่นทันที (ใช้ตอนปิดโปรแกรมหรือกดปุ่มฉุกเฉิน)"""
        now = self.clock() if now is None else float(now)
        if self.state == STATE_SPRAYING:
            started_at = now if self._spray_started_at is None else self._spray_started_at
            self._stop_spraying(now, now - started_at)
        self.hit_streak = 0
        self.miss_streak = 0

    def summary(self) -> dict:
        """สรุปสถิติการทำงาน ใช้เขียนลงรายงานผลการทดลอง"""
        weed_ratio = (self.frames_with_weed / self.frames_processed * 100.0) if self.frames_processed else 0.0
        return {
            "frames_processed": self.frames_processed,
            "frames_with_weed": self.frames_with_weed,
            "weed_frame_percent": round(weed_ratio, 2),
            "spray_events": self.spray_events,
            "total_spray_seconds": round(self.total_spray_seconds, 2),
            "avg_spray_seconds": round(
                self.total_spray_seconds / self.spray_events if self.spray_events else 0.0, 2
            ),
            "max_duration_cutoffs": self.cutoff_events,
        }
