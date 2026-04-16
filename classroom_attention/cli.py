import argparse
import os
import sys

from .config import Config


def build_parser():
    parser = argparse.ArgumentParser(
        description="Classroom attention detection v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ca_gpu.py test.mp4 --save-video
  python ca_gpu.py test.mp4 --threshold 40 --save-video
  python ca_gpu.py test.mp4 --save-video --max-frames 500
""",
    )

    parser.add_argument("video_path", help="Input video path")
    parser.add_argument("--threshold", type=int, default=85, help="Attention threshold (0-100)")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every N+1 frames")
    parser.add_argument("--save-video", action="store_true", help="Save the annotated video")
    parser.add_argument("--no-labels", action="store_true", help="Hide text labels in output video")
    parser.add_argument("-o", "--output", default="output_annotated.mp4", help="Annotated video output path")
    parser.add_argument("--export-json", action="store_true", help="Export structured JSON summary")
    parser.add_argument("--json-output", default="attention_summary.json", help="Structured JSON output path")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process; 0 means all")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO input image size")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per frame")
    parser.add_argument("--agnostic-nms", action="store_true", help="Use class-agnostic NMS")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 inference")
    parser.add_argument("--kpt-vis", type=float, default=0.35, help="Minimum keypoint visibility")
    parser.add_argument(
        "--short-head-down-duration", type=float, default=0.45, help="Minimum short head-down duration in seconds"
    )
    parser.add_argument(
        "--head-down-threshold", type=float, default=0.025, help="Relative head-down threshold versus shoulders"
    )
    parser.add_argument(
        "--min-pose-bbox-height", type=int, default=95, help="Minimum bbox height for behavior scoring"
    )
    parser.add_argument("--edge-box-margin", type=int, default=2, help="Ignore boxes too close to the image edge")
    parser.add_argument("--min-valid-kpts", type=int, default=4, help="Minimum visible keypoints for scoring")
    parser.add_argument("--disable-sideways", action="store_true", help="Disable leaning detection")
    parser.add_argument("--disable-hands-low", action="store_true", help="Disable hand-low detection")
    parser.add_argument("--model", default="yolov8m-pose.pt", help="Pose model weights path")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index; use -1 for CPU")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config path")
    parser.add_argument("--disable-stillness", action="store_true", help="Disable stillness detection")
    parser.add_argument("--enable-stillness", action="store_true", help="Enable stillness detection")
    parser.add_argument("--min-range-sec", type=float, default=2.0, help="Minimum merged not-focus segment length")
    parser.add_argument("--min-total-sec", type=float, default=3.0, help="Minimum per-student total duration")
    parser.add_argument("--min-events", type=int, default=1, help="Minimum per-student event count")
    parser.add_argument("--font", type=str, default=Config.FONT_PATH, help="Font path for Chinese rendering")
    parser.add_argument("--raise-delta", type=float, default=0.10, help="Raise-hand delta relative to torso")
    parser.add_argument("--raise-duration", type=float, default=1.5, help="Raise-hand minimum duration in seconds")
    parser.add_argument("--raise-vis", type=float, default=0.6, help="Raise-hand keypoint visibility threshold")
    return parser


def build_config(args):
    config = Config()
    config.ATTENTION_SCORE_THRESHOLD = args.threshold
    config.SKIP_FRAMES = args.skip_frames
    config.OUTPUT_VIDEO = args.save_video
    config.OUTPUT_VIDEO_PATH = args.output
    config.SHOW_LABELS = not args.no_labels
    config.CONFIDENCE_THRESHOLD = args.conf
    config.IMG_SIZE = args.imgsz
    config.IOU_THRESHOLD = args.iou
    config.MAX_DET = args.max_det
    config.AGNOSTIC_NMS = args.agnostic_nms
    config.HALF = not args.no_half
    config.KPT_VIS = args.kpt_vis
    config.SHORT_HEAD_DOWN_DURATION = args.short_head_down_duration
    config.HEAD_DOWN_THRESHOLD = args.head_down_threshold
    config.MIN_POSE_BBOX_HEIGHT = args.min_pose_bbox_height
    config.EDGE_BOX_MARGIN = args.edge_box_margin
    config.MIN_VALID_KPTS = args.min_valid_kpts
    config.ENABLE_SIDEWAYS = not args.disable_sideways
    config.ENABLE_HANDS_LOW = not args.disable_hands_low
    config.POSE_MODEL = args.model
    config.DEVICE = args.device
    config.TRACKER_CFG = args.tracker
    config.STILLNESS_ENABLED = args.enable_stillness and not args.disable_stillness
    config.MIN_RANGE_SEC = args.min_range_sec
    config.MIN_NOTFOCUS_TOTAL_SEC = args.min_total_sec
    config.MIN_NOTFOCUS_EVENT_COUNT = args.min_events
    config.FONT_PATH = args.font
    config.RAISE_HAND_DELTA = args.raise_delta
    config.RAISE_HAND_DURATION = args.raise_duration
    config.RAISE_HAND_VISIBILITY = args.raise_vis
    return config


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: file not found: {args.video_path}")
        sys.exit(1)

    import torch

    from .pipeline import run_detection_pipeline

    config = build_config(args)
    print("\n" + "-" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and config.DEVICE != -1:
        print(f"GPU: {torch.cuda.get_device_name(config.DEVICE)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(config.DEVICE).total_memory / 1024**3:.1f} GB")
    else:
        print("Runtime device: CPU")
    print(
        "Rule config: "
        f"kpt_vis={config.KPT_VIS}, "
        f"short_head_down={config.SHORT_HEAD_DOWN_DURATION}s, "
        f"head_down_threshold={config.HEAD_DOWN_THRESHOLD}, "
        f"min_bbox_h={config.MIN_POSE_BBOX_HEIGHT}, "
        f"edge_margin={config.EDGE_BOX_MARGIN}, "
        f"min_valid_kpts={config.MIN_VALID_KPTS}, "
        f"sideways={config.ENABLE_SIDEWAYS}, "
        f"hands_low={config.ENABLE_HANDS_LOW}"
    )
    print("-" * 60 + "\n")

    try:
        outputs = run_detection_pipeline(args)
        if outputs["csv_path"]:
            print(f"CSV report saved: {outputs['csv_path']}")
        if outputs["json_path"]:
            print(f"JSON summary saved: {outputs['json_path']}")
        if outputs["video_path"]:
            print(f"Annotated video saved: {outputs['video_path']}")
        if outputs["raise_hand_csv_path"]:
            print(f"Raise-hand events saved: {outputs['raise_hand_csv_path']}")

        print("\n" + "=" * 60)
        print("All tasks completed")
        print("=" * 60 + "\n")

    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
