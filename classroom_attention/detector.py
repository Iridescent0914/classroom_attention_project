import logging
import os
import warnings
from datetime import timedelta

import cv2
import torch
from ultralytics import YOLO

from .reporting import ReportBuilder
from .scoring import calculate_attention_score, is_reliable_pose_target
from .tracker import StudentStateTracker

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def draw_ascii_label(img, text, pos, color, scale=0.55, thickness=2):
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    top = max(0, y - th - baseline - 6)
    bottom = max(th + baseline + 6, y + 4)
    right = min(img.shape[1] - 1, x + tw + 10)
    cv2.rectangle(img, (x, top), (right, bottom), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (x + 4, bottom - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return img


def resolve_tracker_config(tracker_cfg_path):
    resolved_path = tracker_cfg_path
    try:
        if not (isinstance(resolved_path, str) and os.path.exists(resolved_path)):
            import inspect
            import ultralytics as ultralytics_pkg

            pkg_dir = os.path.dirname(inspect.getfile(ultralytics_pkg))
            default_tracker = os.path.join(pkg_dir, "cfg", "trackers", "bytetrack.yaml")
            if os.path.exists(default_tracker):
                resolved_path = default_tracker
    except Exception:
        pass
    return resolved_path


def create_best_effort_video_writer(output_path, fps, frame_size):
    ext = os.path.splitext(str(output_path))[1].lower()
    if ext == ".avi":
        codec_candidates = ["XVID", "MJPG", "mp4v"]
    else:
        # Prefer mp4v first to avoid repeated H264 encoder probe errors on minimal FFmpeg builds.
        codec_candidates = ["mp4v", "avc1", "XVID", "MJPG"]
    for codec in codec_candidates:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, frame_size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    return None, None


class ClassroomMonitor:
    def __init__(self, video_path, config):
        self.video_path = video_path
        self.config = config
        self.attention_records = []
        self.video_metadata = {}
        self.state_tracker = StudentStateTracker()
        self.raise_hand_events = []
        self.report_builder = ReportBuilder(video_path, config)

        if torch.cuda.is_available() and config.DEVICE != -1:
            gpu_name = torch.cuda.get_device_name(config.DEVICE)
            gpu_memory = torch.cuda.get_device_properties(config.DEVICE).total_memory / 1024**3
            print(f"GPU enabled: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("Using CPU mode")

    def process(self, max_frames=0, progress_callback=None):
        print("\n" + "=" * 60)
        print("Classroom Attention Detection System v2.0".center(60))
        print("Added long head down, eyes closed, stillness and raise-hand rules".center(60))
        print("=" * 60 + "\n")

        print("Step 1: loading YOLO pose model...")
        yolo = YOLO(self.config.POSE_MODEL)
        if torch.cuda.is_available() and self.config.DEVICE != -1:
            yolo.to(f"cuda:{self.config.DEVICE}")
        print("Model loaded\n")

        print("Step 2: opening video...")
        video_path = os.path.abspath(self.video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_metadata = {
            "video_path": video_path,
            "fps": round(float(fps), 3) if fps else 0.0,
            "total_frames": int(total_frames),
            "width": int(width),
            "height": int(height),
            "duration_sec": round((float(total_frames) / float(fps)), 2) if fps else 0.0,
            "skip_frames": int(self.config.SKIP_FRAMES),
            "attention_threshold": int(self.config.ATTENTION_SCORE_THRESHOLD),
        }

        if max_frames > 0:
            total_frames = min(total_frames, max_frames)
            print(f"Video: {total_frames} frames (test mode), {fps:.2f}fps, {width}x{height}\n")
        else:
            print(f"Video: {total_frames} frames, {fps:.2f}fps, {width}x{height}\n")

        video_writer = None
        if self.config.OUTPUT_VIDEO:
            output_fps = fps / (self.config.SKIP_FRAMES + 1)
            video_writer, codec_used = create_best_effort_video_writer(
                self.config.OUTPUT_VIDEO_PATH,
                output_fps,
                (width, height),
            )
            if not video_writer:
                raise ValueError(f"Unable to create video file: {self.config.OUTPUT_VIDEO_PATH}")
            print(f"Video codec: {codec_used}")
            print(f"Annotated video output: {os.path.abspath(self.config.OUTPUT_VIDEO_PATH)}\n")

        print("Step 3: running detection and tracking...")
        tracker_cfg_path = resolve_tracker_config(self.config.TRACKER_CFG)
        print(f"Tracker config: {tracker_cfg_path}")

        frame_idx = 0
        processed_count = 0
        warn_no_tracking_printed = False

        try:
            while True:
                if max_frames > 0 and frame_idx >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % (self.config.SKIP_FRAMES + 1) != 0:
                    frame_idx += 1
                    continue

                results = yolo.track(
                    frame,
                    classes=[0],
                    conf=self.config.CONFIDENCE_THRESHOLD,
                    imgsz=self.config.IMG_SIZE,
                    iou=self.config.IOU_THRESHOLD,
                    max_det=self.config.MAX_DET,
                    half=(self.config.HALF and torch.cuda.is_available() and self.config.DEVICE != -1),
                    agnostic_nms=self.config.AGNOSTIC_NMS,
                    persist=True,
                    tracker=tracker_cfg_path,
                    device=(self.config.DEVICE if self.config.DEVICE != -1 else "cpu"),
                    verbose=False,
                )

                rendered_frame, tracking_active = self._process_frame_results(results, fps, frame_idx)
                if rendered_frame is None:
                    rendered_frame = frame
                elif not tracking_active and not warn_no_tracking_printed:
                    print("Tracking did not return track IDs, falling back to detection-only IDs.")
                    warn_no_tracking_printed = True

                if video_writer:
                    video_writer.write(rendered_frame)

                if processed_count % 50 == 0:
                    progress = (frame_idx / total_frames) * 100 if total_frames else 0
                    detected_people = len(results[0].boxes) if results and results[0].boxes else 0
                    not_focus_count = sum(1 for r in self.attention_records if r["frame"] == frame_idx)
                    progress_line = (
                        f"Progress: {progress:.1f}% [{frame_idx}/{total_frames}] | "
                        f"Detected: {detected_people} | Not focus: {not_focus_count}"
                    )
                    print(f"  {progress_line}")
                    if progress_callback:
                        progress_callback(progress_line)

                processed_count += 1
                frame_idx += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user, saving current outputs...")
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.report_builder.generate_report(self.attention_records)

    def build_structured_report(self, df, summary):
        return self.report_builder.build_structured_report(
            df, summary, self.video_metadata, self.raise_hand_events
        )

    def print_report(self, summary):
        self.report_builder.print_report(summary, self.config)

    def _process_frame_results(self, results, fps, frame_idx):
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return None, True

        boxes = results[0].boxes
        keypoints = results[0].keypoints
        orig_frame = results[0].orig_img

        try:
            tracking_active = any(box.id is not None for box in boxes)
        except Exception:
            tracking_active = False

        for i, box in enumerate(boxes):
            track_id = int(box.id[0]) if box.id is not None else 0
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_height = y2 - y1

            if not keypoints or i >= len(keypoints.data):
                continue

            kpts = keypoints.data[i].cpu().numpy()
            if not is_reliable_pose_target(kpts, (x1, y1, x2, y2), orig_frame.shape, self.config):
                cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                continue

            attention_score, reasons = calculate_attention_score(
                kpts, bbox_height, self.config, self.state_tracker, track_id, fps
            )

            if any("Hand raised" in reason for reason in reasons):
                self.raise_hand_events.append(
                    {
                        "student_id": track_id,
                        "time_sec": round(frame_idx / fps, 2),
                        "time_str": str(timedelta(seconds=int(frame_idx / fps))),
                        "frame": frame_idx,
                        "event": "raise_hand",
                        "bbox": (x1, y1, x2, y2),
                    }
                )

            short_term_count = 0
            for reason in reasons:
                if reason.startswith("Head down") or reason.startswith("Leaning") or reason == "Hand low":
                    short_term_count += 1
            long_term_flag = any(
                reason.startswith("Long head down")
                or reason.startswith("Eyes closed")
                or reason.startswith("Still")
                for reason in reasons
            )
            is_not_focused = long_term_flag or (
                attention_score < self.config.ATTENTION_SCORE_THRESHOLD and short_term_count >= 1
            )

            show_raise_hand = any("Hand raised" in reason for reason in reasons)
            if show_raise_hand:
                is_not_focused = False

            color = (0, 0, 255) if is_not_focused else (0, 255, 0)
            border_thickness = 4 if long_term_flag else 2
            cv2.rectangle(orig_frame, (x1, y1), (x2, y2), color, border_thickness)

            if self.config.SHOW_LABELS:
                status = "NOT FOCUS" if is_not_focused else "FOCUS"
                if show_raise_hand:
                    prioritized = [r for r in reasons if "Hand raised" in r] + [
                        r for r in reasons if "Hand raised" not in r
                    ]
                else:
                    prioritized = list(reasons)
                main_reasons = prioritized[:2]
                reason_text = f"({'; '.join(main_reasons)})" if main_reasons else ""
                label = f"ID:{track_id} {status}({attention_score}) {reason_text}"
                label_y = max(20, y1 - 10)
                orig_frame = draw_ascii_label(orig_frame, label, (x1, label_y), color)
                if show_raise_hand:
                    orig_frame = draw_ascii_label(
                        orig_frame,
                        "HAND RAISED",
                        (x1, max(20, label_y - 30)),
                        (0, 255, 255),
                    )

            if is_not_focused:
                self.attention_records.append(
                    {
                        "student_id": track_id,
                        "time_sec": round(frame_idx / fps, 2),
                        "time_str": str(timedelta(seconds=int(frame_idx / fps))),
                        "frame": frame_idx,
                        "score": attention_score,
                        "reason": ";".join(reasons),
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        return orig_frame, tracking_active
