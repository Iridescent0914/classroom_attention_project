from dataclasses import dataclass
import os


def resolve_default_font() -> str:
    """Pick a reasonable default font path for the current OS."""
    candidates = []
    if os.name == "nt":
        candidates.extend(
            [
                r"C:/Windows/Fonts/simhei.ttf",
                r"C:/Windows/Fonts/msyh.ttc",
                r"C:/Windows/Fonts/simsun.ttc",
                r"C:/Windows/Fonts/arial.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJKSC-Regular.otf",
                "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc",
                "/usr/share/fonts/truetype/arphic/ukai.ttc",
                "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )

    for font_path in candidates:
        if os.path.exists(font_path):
            return font_path
    return candidates[0]


@dataclass
class Config:
    POSE_MODEL: str = "yolov8m-pose.pt"
    DEVICE: int = 0
    HEAD_DOWN_THRESHOLD: float = 0.025
    HEAD_DOWN_DURATION: float = 5.0
    SHORT_HEAD_DOWN_DURATION: float = 0.45
    EYE_CLOSED_THRESHOLD: float = 0.18
    EYE_CLOSED_DURATION: float = 4.0
    STILLNESS_THRESHOLD: float = 5.0
    STILLNESS_DURATION: float = 4.0
    GAZE_FIXED_THRESHOLD: float = 0.02
    STILLNESS_ENABLED: bool = False
    SHOULDER_TILT_THRESHOLD: int = 25
    HAND_BELOW_HIP_THRESHOLD: float = 0.02
    ATTENTION_SCORE_THRESHOLD: int = 50
    RAISE_HAND_DELTA: float = 0.10
    RAISE_HAND_DURATION: float = 1.5
    RAISE_HAND_VISIBILITY: float = 0.6
    OUTPUT_VIDEO: bool = True
    OUTPUT_VIDEO_PATH: str = "output_annotated.mp4"
    SHOW_LABELS: bool = True
    SKIP_FRAMES: int = 2
    CONFIDENCE_THRESHOLD: float = 0.25
    IMG_SIZE: int = 1280
    TRACKER_CFG: str = "bytetrack.yaml"
    IOU_THRESHOLD: float = 0.5
    MAX_DET: int = 300
    HALF: bool = True
    AGNOSTIC_NMS: bool = False
    KPT_VIS: float = 0.35
    MIN_POSE_BBOX_HEIGHT: int = 95
    EDGE_BOX_MARGIN: int = 2
    MIN_VALID_KPTS: int = 4
    ENABLE_SIDEWAYS: bool = True
    ENABLE_HANDS_LOW: bool = True
    FONT_PATH: str = resolve_default_font()
    MIN_RANGE_SEC: float = 2.0
    MIN_NOTFOCUS_TOTAL_SEC: float = 3.0
    MIN_NOTFOCUS_EVENT_COUNT: int = 1
