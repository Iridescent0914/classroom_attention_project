import numpy as np


def calculate_attention_score(keypoints, bbox_height, config, state_tracker, student_id, fps):
    """Return (score, reasons) for a single tracked student."""
    if keypoints is None or len(keypoints) < 17:
        return 0, []

    score = 100
    reasons = []

    try:
        state_tracker.update(student_id, keypoints, config, fps)
        long_term_behaviors = state_tracker.check_long_term_behaviors(
            student_id, keypoints, fps, config
        )

        for behavior in long_term_behaviors:
            if behavior.startswith("Long head down"):
                score -= 80
            elif behavior.startswith("Eyes closed"):
                score -= 70
            elif behavior.startswith("Still"):
                score -= 50
            reasons.append(behavior)

        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hand = keypoints[9]
        right_hand = keypoints[10]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        if (
            not any(reason.startswith("Long head down") for reason in reasons)
            and nose[2] > config.KPT_VIS
            and left_shoulder[2] > config.KPT_VIS
            and right_shoulder[2] > config.KPT_VIS
        ):
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            head_drop = (nose[1] - shoulder_center_y) * bbox_height
            if head_drop > config.HEAD_DOWN_THRESHOLD * bbox_height:
                state_tracker.short_head_down_timer[student_id] += 1 / fps
                if (
                    state_tracker.short_head_down_timer[student_id]
                    >= config.SHORT_HEAD_DOWN_DURATION
                ):
                    score -= 30
                    reasons.append("Head down")
            else:
                state_tracker.short_head_down_timer[student_id] = 0
        else:
            state_tracker.short_head_down_timer[student_id] = 0

        if (
            config.ENABLE_SIDEWAYS
            and left_shoulder[2] > config.KPT_VIS
            and right_shoulder[2] > config.KPT_VIS
        ):
            shoulder_vec = np.array(
                [right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]]
            )
            angle = np.degrees(np.arctan2(abs(shoulder_vec[1]), abs(shoulder_vec[0])))
            if angle > config.SHOULDER_TILT_THRESHOLD:
                score -= 20
                reasons.append(f"Leaning({int(angle)}deg)")

        torso_h = None
        if (
            left_shoulder[2] > config.KPT_VIS
            and right_shoulder[2] > config.KPT_VIS
            and left_hip[2] > config.KPT_VIS
            and right_hip[2] > config.KPT_VIS
        ):
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
            hip_center_y = (left_hip[1] + right_hip[1]) / 2.0
            torso_h = max(hip_center_y - shoulder_center_y, 1.0)

        hand_delta = config.HAND_BELOW_HIP_THRESHOLD * (
            torso_h if torso_h is not None else max(bbox_height, 1.0)
        )
        hand_low = False
        if left_hand[2] > config.KPT_VIS and left_hip[2] > config.KPT_VIS:
            hand_low = left_hand[1] > left_hip[1] + hand_delta
        if right_hand[2] > config.KPT_VIS and right_hip[2] > config.KPT_VIS:
            hand_low = hand_low or right_hand[1] > right_hip[1] + hand_delta

        if config.ENABLE_HANDS_LOW and hand_low:
            score -= 15
            reasons.append("Hand low")

    except Exception as exc:
        print(f"Scoring error: {exc}")
        return max(0, score), ["Scoring error"]

    return max(0, min(100, score)), reasons


def is_reliable_pose_target(keypoints, bbox, frame_shape, config):
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape[:2]
    if (x2 - x1) < 20 or (y2 - y1) < config.MIN_POSE_BBOX_HEIGHT:
        return False
    if x1 <= config.EDGE_BOX_MARGIN or y1 <= config.EDGE_BOX_MARGIN:
        return False
    if x2 >= frame_w - config.EDGE_BOX_MARGIN or y2 >= frame_h - config.EDGE_BOX_MARGIN:
        return False
    if keypoints is None or len(keypoints) < 17:
        return False
    visible_kpts = int(np.sum(keypoints[:, 2] > config.KPT_VIS))
    return visible_kpts >= config.MIN_VALID_KPTS
