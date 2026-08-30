"""
โปรแกรมหลัก — วนลูป: รับภาพ -> วิเคราะห์ -> ตัดสินใจ -> สั่งรีเลย์

รันบน PC (ทดสอบ):
    python -m src.main
รันบน Raspberry Pi (ใช้งานจริง):
    python -m src.main --source picamera2 --weights models/best_float32.tflite --headless
รันโหมดต้นแบบนำเสนอ (prototype-v1) ที่ตรวจจับด้วยสีแดง/เขียวแทนโมเดล AI:
    python -m src.main --backend color

ปุ่มลัดขณะรัน (เฉพาะเมื่อเปิดหน้าต่างแสดงผล):
    q / ESC = ออก        p = หยุดชั่วคราว
    s       = บันทึกภาพ   x = หยุดทำงานฉุกเฉิน (ปิดปั๊ม/เลเซอร์ทันที)
"""

from __future__ import annotations

import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path

# รองรับทั้งการรันแบบ  python -m src.main  และ  python src/main.py
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import camera as camera_module
    from src import overlay
    from src.config import PROJECT_ROOT, ConfigError, load_config
    from src.detector import DetectorError, build_detector, effective_conf
    from src.logic import SprayController
    from src.relay import build_relay
    from src.stats import EventLogger, FpsMeter
    from src.vision_utils import imwrite
else:
    from . import camera as camera_module
    from . import overlay
    from .config import PROJECT_ROOT, ConfigError, load_config
    from .detector import DetectorError, build_detector, effective_conf
    from .logic import SprayController
    from .relay import build_relay
    from .stats import EventLogger, FpsMeter
    from .vision_utils import imwrite


# ธงบอกให้ลูปหลักหยุด ตั้งค่าโดยตัวจัดการสัญญาณระบบ
_should_stop = False


def _handle_signal(signum, _frame):  # pragma: no cover - ขึ้นกับระบบปฏิบัติการ
    """รับสัญญาณปิดโปรแกรมจากระบบ (เช่น systemctl stop) แล้วออกจากลูปอย่างเรียบร้อย

    สำคัญมาก: ถ้าถูกฆ่าดื้อๆ ปั๊มจะค้างเปิด การจับสัญญาณไว้ทำให้ finally ได้ทำงาน
    """
    global _should_stop
    print(f"\n[main] ได้รับสัญญาณ {signum} -> กำลังปิดระบบอย่างปลอดภัย...")
    _should_stop = True


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Farm Autonomous Robot — ระบบตรวจจับวัชพืชและสั่งพ่นยาอัตโนมัติ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="พาธไฟล์ตั้งค่า (ค่าเริ่มต้น: config.yaml)")
    parser.add_argument(
        "--source",
        choices=["webcam", "picamera2", "video", "images"],
        help="แหล่งภาพ (แทนค่าใน config)",
    )
    parser.add_argument("--path", help="พาธไฟล์วิดีโอหรือโฟลเดอร์ภาพ (ใช้กับ --source video/images)")
    parser.add_argument("--index", type=int, help="หมายเลขกล้องเว็บแคม")
    parser.add_argument("--weights", help="พาธไฟล์โมเดล (.pt หรือ .tflite)")
    parser.add_argument(
        "--backend",
        choices=["auto", "ultralytics", "tflite", "roboflow", "color"],
        help="เอนจิ้นตรวจจับ (color = ตรวจจับด้วยสีแดง/เขียว ไม่ใช้โมเดล AI)",
    )
    parser.add_argument("--conf", type=float, help="ค่า confidence ขั้นต่ำ (แทนค่าใน config)")
    parser.add_argument("--headless", action="store_true", help="ไม่เปิดหน้าต่างแสดงผล (ใช้บน Pi)")
    parser.add_argument("--show", action="store_true", help="บังคับเปิดหน้าต่างแสดงผล")
    parser.add_argument("--mock-relay", action="store_true", help="บังคับใช้รีเลย์จำลอง ไม่แตะ GPIO จริง")
    parser.add_argument("--record", metavar="PATH", help="บันทึกวิดีโอผลลัพธ์ลงไฟล์ที่ระบุ")
    parser.add_argument("--max-frames", type=int, default=0, help="ประมวลผลกี่เฟรมแล้วหยุด (0 = ไม่จำกัด)")
    parser.add_argument("--no-log", action="store_true", help="ไม่เขียนไฟล์ CSV บันทึกเหตุการณ์")
    return parser.parse_args(argv)


def apply_overrides(cfg, args) -> None:
    """เอาค่าจาก command line ไปทับค่าใน config.yaml"""
    if args.source:
        cfg.camera.source = args.source
    if args.path:
        cfg.camera.path = args.path
    if args.index is not None:
        cfg.camera.index = args.index
    if args.weights:
        cfg.model.weights = args.weights
    if args.backend:
        cfg.model.backend = args.backend
    if args.conf is not None:
        cfg.model.conf = args.conf
    if args.headless:
        cfg.runtime.show_window = False
    if args.show:
        cfg.runtime.show_window = True
    if args.record:
        cfg.runtime.save_video = args.record
    if args.no_log:
        cfg.runtime.log_csv = ""
        cfg.runtime.save_snapshots = False
    cfg.validate()


def _open_video_writer(path: Path, fps: float, size: tuple[int, int]):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), size)
    if not writer.isOpened():
        print(f"[main] เตือน: เปิดไฟล์บันทึกวิดีโอไม่ได้ -> {path}")
        return None
    print(f"[main] บันทึกวิดีโอผลลัพธ์ไปที่ {path}")
    return writer


def _log_decision(logger: EventLogger, frame_no: int, decision, fps: float, timing: dict, event: str) -> None:
    best = decision.analysis.best_weed
    logger.log(
        frame=frame_no,
        event=event,
        state=decision.state,
        pump="ON" if decision.spray_on else "OFF",
        class_name=best.class_name if best else "",
        confidence=f"{best.confidence:.4f}" if best else "",
        x1=f"{best.x1:.1f}" if best else "",
        y1=f"{best.y1:.1f}" if best else "",
        x2=f"{best.x2:.1f}" if best else "",
        y2=f"{best.y2:.1f}" if best else "",
        weed_count=len(decision.analysis.actionable_weeds),
        target_count=len(decision.analysis.targets),
        fps=f"{fps:.2f}",
        infer_ms=f"{timing.get('inference', 0.0):.1f}",
        reason=decision.reason,
    )


def run(cfg, args) -> int:
    global _should_stop

    labels = overlay.labels_for(cfg.relay.actuator)
    backend = cfg.model.resolved_backend()

    print("=" * 72)
    print(" Smart Farm Autonomous Robot — ระบบตรวจจับวัชพืชและสั่งการอัตโนมัติ")
    print("=" * 72)
    print(f" ไฟล์ตั้งค่า : {cfg.source_path}")
    print(f" แหล่งภาพ   : {cfg.camera.source}")
    if backend == "color":
        print(" เอนจิ้น    : color — ตรวจจับด้วยสี (HSV) ไม่ใช้โมเดล AI")
        print(f" แถบสี      : แดง='{cfg.color.red_name}' | เขียว='{cfg.color.green_name}'")
    else:
        print(f" โมเดล      : {cfg.model.weights}  (เอนจิ้น {backend})")
    print(f" คลาส       : เป้าหมาย={cfg.classes.target} | วัชพืช={cfg.classes.weed}")
    print(f" เกณฑ์ conf : กลาง={cfg.model.conf} | รายคลาส={cfg.classes.per_class_conf or 'ไม่ได้ตั้ง'}")
    print(f" อุปกรณ์    : {labels.device_th} (relay.actuator = {cfg.relay.actuator})")
    print("-" * 72)

    detector = build_detector(cfg)
    source = camera_module.open_source(cfg.camera, PROJECT_ROOT)
    relay = build_relay(cfg.relay, force_mock=args.mock_relay)
    controller = SprayController(
        spray_cfg=cfg.spray,
        roi_cfg=cfg.roi,
        class_cfg=cfg.classes,
        base_conf=effective_conf(cfg),
        geom_cfg=cfg.geometry_filter,
        action_word=labels.action_th,
        device_word=labels.device_th,
    )
    logger = EventLogger(
        cfg.resolve_path(cfg.runtime.log_csv) if cfg.runtime.log_csv else "",
        enabled=bool(cfg.runtime.log_csv),
    )
    fps_meter = FpsMeter(window=30)

    snapshot_dir = cfg.resolve_path(cfg.runtime.snapshot_dir)
    if cfg.runtime.save_snapshots:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    paused = False
    frame_no = 0
    window_name = "Smart Farm Autonomous Robot"

    cv2 = None
    if cfg.runtime.show_window:
        import cv2 as _cv2

        cv2 = _cv2

    print("[main] อุ่นเครื่องกล้อง...")
    source.warmup()
    print("[main] เริ่มทำงาน — กด Ctrl+C เพื่อหยุด\n")

    try:
        while not _should_stop:
            if cfg.runtime.show_window and paused:
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = False
                    print("[main] ทำงานต่อ")
                continue

            frame = source.read()
            if frame is None:
                print("[main] ไม่มีภาพแล้ว (แหล่งภาพจบ) -> หยุดทำงาน")
                break

            frame_no += 1
            detections = detector.infer(frame)
            decision = controller.update(detections, frame.shape[:2])

            # สั่งรีเลย์ตามคำตัดสิน — จุดเดียวในโปรแกรมที่แตะฮาร์ดแวร์ปั๊ม
            relay.set(decision.spray_on)

            fps = fps_meter.tick()
            timing = detector.last_timing_ms

            # บันทึกเฉพาะตอนสถานะปั๊มเปลี่ยน เพื่อให้ไฟล์ log อ่านง่าย
            if decision.changed:
                event = "SPRAY_START" if decision.spray_on else "SPRAY_STOP"
                _log_decision(logger, frame_no, decision, fps, timing, event)
                verb = "เริ่ม" if decision.spray_on else "หยุด"
                print(
                    f"[{datetime.now():%H:%M:%S}] {event} ({verb}{labels.action_th}): {decision.reason}"
                )

                if decision.spray_on and cfg.runtime.save_snapshots:
                    shot = overlay.render(
                        frame,
                        decision,
                        fps,
                        cfg.roi.pixel_box(frame.shape[1], frame.shape[0]) if cfg.roi.enabled else None,
                        timing,
                        labels=labels,
                    )
                    shot_path = snapshot_dir / f"spray_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                    imwrite(shot_path, shot)

            if cfg.runtime.print_every and frame_no % cfg.runtime.print_every == 0:
                print(
                    f"[{frame_no:>6}] {fps:5.1f} FPS | {decision.state:<9}"
                    f" | {labels.device_th} {'เปิด' if decision.spray_on else 'ปิด '}"
                    f" | วัชพืช {len(decision.analysis.actionable_weeds)}"
                    f" พืชหลัก {len(decision.analysis.targets)}"
                    f" | {decision.reason}"
                )

            need_canvas = cfg.runtime.show_window or cfg.runtime.save_video
            if need_canvas:
                roi_box = cfg.roi.pixel_box(frame.shape[1], frame.shape[0]) if cfg.roi.enabled else None
                footer = (
                    f"frame {frame_no} | hit {decision.hit_streak}/{cfg.spray.confirm_frames}"
                    f" | miss {decision.miss_streak}/{cfg.spray.release_frames}"
                )
                canvas = overlay.render(frame, decision, fps, roi_box, timing, footer, labels)

                if cfg.runtime.save_video:
                    if writer is None:
                        writer = _open_video_writer(
                            cfg.resolve_path(cfg.runtime.save_video),
                            cfg.camera.fps,
                            (canvas.shape[1], canvas.shape[0]),
                        )
                    if writer is not None:
                        writer.write(canvas)

                if cfg.runtime.show_window:
                    try:
                        cv2.imshow(window_name, canvas)
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error:
                        # opencv-python-headless ไม่มีส่วนแสดงผลหน้าต่าง (ใช้บน Pi)
                        # ไม่ควรทำให้ทั้งระบบหยุด แค่ปิดการแสดงผลแล้วทำงานต่อ
                        print("[main] แสดงหน้าต่างไม่ได้ -> เปลี่ยนเป็นโหมด headless โดยอัตโนมัติ")
                        print("       ถ้าต้องการดูภาพ ให้ติดตั้ง opencv-python (ไม่ใช่ opencv-python-headless)")
                        cfg.runtime.show_window = False
                        key = 255  # ไม่มีปุ่มถูกกด

                    if key in (ord("q"), 27):
                        break
                    if key == ord("p"):
                        paused = True
                        print("[main] หยุดชั่วคราว (กด p อีกครั้งเพื่อทำงานต่อ)")
                    elif key == ord("s"):
                        shot_path = snapshot_dir / f"manual_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                        imwrite(shot_path, canvas)
                        print(f"[main] บันทึกภาพ -> {shot_path}")
                    elif key == ord("x"):
                        controller.force_stop()
                        relay.off()
                        print(f"[main] หยุด{labels.action_th}ฉุกเฉิน — ปิด{labels.device_th}แล้ว")

            if args.max_frames and frame_no >= args.max_frames:
                print(f"[main] ครบ {args.max_frames} เฟรมตามที่กำหนด -> หยุดทำงาน")
                break

    except KeyboardInterrupt:
        print("\n[main] ผู้ใช้กด Ctrl+C -> กำลังปิดระบบ")
    finally:
        # ลำดับนี้สำคัญ: ปิดปั๊มก่อนเสมอ แล้วค่อยคืนทรัพยากรอื่น
        try:
            controller.force_stop()
            relay.off()
        finally:
            relay.close()

        source.release()
        detector.close()
        logger.close()
        if writer is not None:
            writer.release()
        if cfg.runtime.show_window and cv2 is not None:
            cv2.destroyAllWindows()

        _print_summary(controller, relay, fps_meter, frame_no, cfg, labels)

    return 0


def _print_summary(controller, relay, fps_meter, frame_no: int, cfg, labels) -> None:
    stats = controller.summary()
    action = labels.action_th
    print("\n" + "=" * 72)
    print(" สรุปผลการทำงาน")
    print("=" * 72)
    print(f" เวลาทำงานรวม        : {fps_meter.elapsed_seconds:.1f} วินาที")
    print(f" จำนวนเฟรมที่ประมวลผล : {frame_no}")
    print(f" FPS เฉลี่ย            : {fps_meter.average_fps:.2f}")
    print(f" เฟรมที่พบวัชพืช       : {stats['frames_with_weed']} ({stats['weed_frame_percent']}%)")
    print(f" จำนวนครั้งที่สั่ง{action}    : {stats['spray_events']}")
    print(f" เวลา{action}รวม           : {stats['total_spray_seconds']} วินาที")
    print(f" เวลา{action}เฉลี่ยต่อครั้ง   : {stats['avg_spray_seconds']} วินาที")
    print(f" ถูกตัดเพราะ{action}นานเกิน  : {stats['max_duration_cutoffs']} ครั้ง")
    print(
        f" สถานะ{labels.device_th}ตอนจบ       : "
        f"{'เปิดอยู่ (ผิดปกติ!)' if relay.is_on else 'ปิดเรียบร้อย'}"
    )
    if cfg.runtime.log_csv:
        print(f" ไฟล์บันทึกเหตุการณ์   : {cfg.resolve_path(cfg.runtime.log_csv)}")
    print("=" * 72)


def main(argv=None) -> int:
    args = parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    try:
        cfg = load_config(args.config)
        apply_overrides(cfg, args)
    except ConfigError as exc:
        print(f"\n[ผิดพลาด] ไฟล์ตั้งค่าไม่ถูกต้อง:\n  {exc}\n", file=sys.stderr)
        return 2

    try:
        return run(cfg, args)
    except (DetectorError, camera_module.CameraError) as exc:
        print(f"\n[ผิดพลาด] {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
