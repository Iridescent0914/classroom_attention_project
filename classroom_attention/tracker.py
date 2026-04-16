from collections import defaultdict, deque

import numpy as np


class StudentStateTracker:
    """Keep per-student temporal state for behavior rules."""

    def __init__(self):
        self.head_position = defaultdict(lambda: deque(maxlen=30))
        self.eye_openness = defaultdict(lambda: deque(maxlen=30))
        self.gaze_position = defaultdict(lambda: deque(maxlen=30))
        self.head_down_timer = defaultdict(float)
        self.short_head_down_timer = defaultdict(float)
        self.eye_closed_timer = defaultdict(float)
        self.stillness_timer = defaultdict(float)
        self.raise_hand_timer = defaultdict(float)

    def update(self, student_id, keypoints, config, fps):
        if keypoints is None or len(keypoints) == 0:
            return

        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]

        if nose[2] > config.KPT_VIS:
            self.head_position[student_id].append(nose[:2])

        self.eye_openness[student_id].append(
            self.calculate_eye_aspect_ratio(left_eye, right_eye, config)
        )

        if left_eye[2] > config.KPT_VIS and right_eye[2] > config.KPT_VIS:
            gaze_x = (left_eye[0] + right_eye[0]) / 2
            gaze_y = (left_eye[1] + right_eye[1]) / 2
            self.gaze_position[student_id].append((gaze_x, gaze_y))

    @staticmethod
    def calculate_eye_aspect_ratio(left_eye, right_eye, config):
        try:
            if left_eye[2] > config.KPT_VIS and right_eye[2] > config.KPT_VIS:
                return (left_eye[2] + right_eye[2]) / 2
        except Exception:
            pass
        return 1.0

    def check_long_term_behaviors(self, student_id, keypoints, fps, config):
        behaviors = []
        if keypoints is None or len(keypoints) < 13:
            return behaviors

        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        if (
            nose[2] > config.KPT_VIS
            and left_shoulder[2] > config.KPT_VIS
            and right_shoulder[2] > config.KPT_VIS
        ):
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            if nose[1] - shoulder_center_y > config.HEAD_DOWN_THRESHOLD:
                self.head_down_timer[student_id] += 1 / fps
                if self.head_down_timer[student_id] >= config.HEAD_DOWN_DURATION:
                    behaviors.append(
                        f"Long head down({self.head_down_timer[student_id]:.1f}s)"
                    )
            else:
                self.head_down_timer[student_id] = 0
        else:
            self.head_down_timer[student_id] = 0

        if self.eye_openness[student_id]:
            ear = self.eye_openness[student_id][-1]
            if ear < config.EYE_CLOSED_THRESHOLD:
                self.eye_closed_timer[student_id] += 1 / fps
                if self.eye_closed_timer[student_id] >= config.EYE_CLOSED_DURATION:
                    behaviors.append(
                        f"Eyes closed({self.eye_closed_timer[student_id]:.1f}s)"
                    )
            else:
                self.eye_closed_timer[student_id] = 0
        else:
            self.eye_closed_timer[student_id] = 0

        if config.STILLNESS_ENABLED and len(self.head_position[student_id]) >= 5:
            recent_positions = list(self.head_position[student_id])[-5:]
            head_movement = 0.0
            for i in range(1, len(recent_positions)):
                dx = recent_positions[i][0] - recent_positions[i - 1][0]
                dy = recent_positions[i][1] - recent_positions[i - 1][1]
                head_movement += float(np.sqrt(dx * dx + dy * dy))

            if head_movement < config.STILLNESS_THRESHOLD:
                self.stillness_timer[student_id] += 1 / fps
                if self.stillness_timer[student_id] >= config.STILLNESS_DURATION:
                    behaviors.append(f"Still({self.stillness_timer[student_id]:.1f}s)")
            else:
                self.stillness_timer[student_id] = 0
        else:
            self.stillness_timer[student_id] = 0

        try:
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
            hip_center_y = (left_hip[1] + right_hip[1]) / 2.0
            torso_h = max(hip_center_y - shoulder_center_y, 1.0)
            delta_pixels = config.RAISE_HAND_DELTA * torso_h
            vis_ok_left = (
                left_wrist[2] > config.KPT_VIS
                and left_elbow[2] > config.KPT_VIS
                and left_shoulder[2] > config.KPT_VIS
            )
            vis_ok_right = (
                right_wrist[2] > config.KPT_VIS
                and right_elbow[2] > config.KPT_VIS
                and right_shoulder[2] > config.KPT_VIS
            )
            left_raise = vis_ok_left and (
                left_wrist[1] < left_shoulder[1] - delta_pixels
                and left_elbow[1] < left_shoulder[1] - delta_pixels
            )
            right_raise = vis_ok_right and (
                right_wrist[1] < right_shoulder[1] - delta_pixels
                and right_elbow[1] < right_shoulder[1] - delta_pixels
            )

            if left_raise or right_raise:
                self.raise_hand_timer[student_id] += 1 / fps
                if self.raise_hand_timer[student_id] >= config.RAISE_HAND_DURATION:
                    behaviors.append(
                        f"Hand raised({self.raise_hand_timer[student_id]:.1f}s)"
                    )
            else:
                self.raise_hand_timer[student_id] = 0
        except Exception:
            pass

        return behaviors
