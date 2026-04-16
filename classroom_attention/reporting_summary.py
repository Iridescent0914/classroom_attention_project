import os
from collections import Counter, defaultdict
from datetime import timedelta

import pandas as pd


class ReportBuilder:
    def __init__(self, video_path, config):
        self.video_path = video_path
        self.config = config

    def generate_report(self, attention_records):
        if not attention_records:
            return None, {}

        df = pd.DataFrame(attention_records)
        summary = {}

        for student_id in sorted(df["student_id"].unique()):
            student_data = df[df["student_id"] == student_id].sort_values("time_sec")
            time_ranges = []
            if not student_data.empty:
                start_time = student_data.iloc[0]["time_sec"]
                end_time = student_data.iloc[0]["time_sec"]

                for _, row in student_data.iterrows():
                    if row["time_sec"] - end_time > 3:
                        if end_time - start_time >= self.config.MIN_RANGE_SEC:
                            time_ranges.append((start_time, end_time))
                        start_time = row["time_sec"]
                    end_time = row["time_sec"]

                if end_time - start_time >= self.config.MIN_RANGE_SEC:
                    time_ranges.append((start_time, end_time))

            formatted_ranges = []
            total_duration = 0.0
            for start, end in time_ranges:
                duration = end - start
                time_range_data = student_data[
                    (student_data["time_sec"] >= start) & (student_data["time_sec"] <= end)
                ]
                if not time_range_data.empty:
                    all_reasons = []
                    for reason_str in time_range_data["reason"]:
                        all_reasons.extend(str(reason_str).split(";"))
                    reason_counts = Counter(all_reasons)
                    main_reason = reason_counts.most_common(1)[0][0] if reason_counts else "Unknown"
                else:
                    main_reason = "Unknown"

                formatted_ranges.append(
                    {
                        "start": str(timedelta(seconds=int(start))),
                        "end": str(timedelta(seconds=int(end))),
                        "duration_sec": round(duration, 1),
                        "reason": main_reason,
                    }
                )
                total_duration += duration

            if formatted_ranges:
                summary[student_id] = {
                    "time_ranges": formatted_ranges,
                    "total_duration_sec": round(total_duration, 1),
                    "event_count": len(formatted_ranges),
                }

        return df, summary

    def build_structured_report(self, df, summary, video_metadata, raise_hand_events):
        payload = {
            "video": dict(video_metadata) if video_metadata else {"video_path": os.path.abspath(self.video_path)},
            "overview": {
                "students_with_not_focus_events": 0,
                "not_focus_record_count": 0,
                "not_focus_event_count": 0,
                "raise_hand_event_count": len(raise_hand_events),
                "top_students": [],
            },
            "students": [],
            "keyframe_candidates": [],
            "raise_hand_events": raise_hand_events[:50],
        }

        if df is None or df.empty or not summary:
            return payload

        payload["overview"]["students_with_not_focus_events"] = len(summary)
        payload["overview"]["not_focus_record_count"] = int(len(df))
        payload["overview"]["not_focus_event_count"] = int(
            sum(student_summary["event_count"] for student_summary in summary.values())
        )

        top_students = []
        keyframe_candidates = []

        for student_id in sorted(summary.keys()):
            student_rows = df[df["student_id"] == student_id].sort_values("time_sec")
            reason_counter = defaultdict(int)
            event_ranges = []

            for reason_str in student_rows["reason"].tolist():
                for reason in str(reason_str).split(";"):
                    reason = reason.strip()
                    if reason:
                        reason_counter[reason] += 1

            for idx, time_range in enumerate(summary[student_id]["time_ranges"], start=1):
                start_parts = [int(part) for part in str(time_range["start"]).split(":")]
                end_parts = [int(part) for part in str(time_range["end"]).split(":")]
                start_sec = round(float(start_parts[0] * 3600 + start_parts[1] * 60 + start_parts[2]), 2)
                end_sec = round(float(end_parts[0] * 3600 + end_parts[1] * 60 + end_parts[2]), 2)
                segment_rows = student_rows[
                    (student_rows["time_sec"] >= max(0, start_sec - 0.01))
                    & (student_rows["time_sec"] <= end_sec + 0.01)
                ]
                if not segment_rows.empty:
                    midpoint_sec = round(
                        (float(segment_rows["time_sec"].min()) + float(segment_rows["time_sec"].max())) / 2,
                        2,
                    )
                    midpoint_row = segment_rows.iloc[
                        (segment_rows["time_sec"] - midpoint_sec).abs().argsort()[:1]
                    ].iloc[0]
                else:
                    midpoint_sec = round((start_sec + end_sec) / 2, 2)
                    midpoint_row = student_rows.iloc[
                        (student_rows["time_sec"] - midpoint_sec).abs().argsort()[:1]
                    ].iloc[0]

                bbox_nums = self._parse_bbox(midpoint_row["bbox"])
                bbox_area = max(0, bbox_nums[2] - bbox_nums[0]) * max(0, bbox_nums[3] - bbox_nums[1])
                event_payload = {
                    "event_index": idx,
                    "start": time_range["start"],
                    "end": time_range["end"],
                    "duration_sec": float(time_range["duration_sec"]),
                    "main_reason": time_range["reason"],
                    "midpoint_sec": midpoint_sec,
                    "midpoint_frame": int(midpoint_row["frame"]),
                    "score": int(midpoint_row["score"]),
                    "bbox": str(midpoint_row["bbox"]),
                }
                event_ranges.append(event_payload)
                keyframe_candidates.append(
                    {
                        "student_id": int(student_id),
                        "priority": float(time_range["duration_sec"]),
                        "time_sec": midpoint_sec,
                        "frame": int(midpoint_row["frame"]),
                        "reason": time_range["reason"],
                        "score": int(midpoint_row["score"]),
                        "bbox": str(midpoint_row["bbox"]),
                        "bbox_area": int(bbox_area),
                    }
                )

            dominant_reasons = sorted(reason_counter.items(), key=lambda item: (-item[1], item[0]))[:5]
            payload["students"].append(
                {
                    "student_id": int(student_id),
                    "event_count": int(summary[student_id]["event_count"]),
                    "total_duration_sec": float(summary[student_id]["total_duration_sec"]),
                    "dominant_reasons": [
                        {"reason": reason, "count": count} for reason, count in dominant_reasons
                    ],
                    "event_ranges": event_ranges,
                }
            )
            top_students.append(
                {
                    "student_id": int(student_id),
                    "event_count": int(summary[student_id]["event_count"]),
                    "total_duration_sec": float(summary[student_id]["total_duration_sec"]),
                }
            )

        top_students.sort(key=lambda item: (-item["total_duration_sec"], -item["event_count"], item["student_id"]))
        keyframe_candidates.sort(
            key=lambda item: (
                -item["priority"],
                -item.get("bbox_area", 0),
                item["score"],
                item["time_sec"],
                item["student_id"],
            )
        )
        payload["overview"]["top_students"] = top_students[:10]
        payload["keyframe_candidates"] = keyframe_candidates[:20]
        return payload

    @staticmethod
    def print_report(summary, config):
        if not summary:
            print("\n=== Detection Report ===")
            print("No not-focus behavior detected.")
            return

        print("\n" + "=" * 70)
        print("Classroom Attention Report v2.0".center(70))
        print("=" * 70)

        qualified_ids = []
        for student_id in sorted(summary.keys()):
            data = summary[student_id]
            qualifies = (
                data["total_duration_sec"] >= config.MIN_NOTFOCUS_TOTAL_SEC
                and data["event_count"] >= config.MIN_NOTFOCUS_EVENT_COUNT
            )
            if qualifies:
                qualified_ids.append(student_id)

            print(f"\n[Student ID: {student_id:02d}]")
            print(f"Not-focus events: {data['event_count']}")
            print(f"Total duration: {data['total_duration_sec']}s")
            print("Time ranges:")
            for i, time_range in enumerate(data["time_ranges"], 1):
                print(f"  {i}. {time_range['start']} ~ {time_range['end']} ({time_range['duration_sec']}s)")
                print(f"     Main reason: {time_range['reason']}")

        print("\n" + "=" * 70)
        print(f"Students over threshold: {len(qualified_ids)}")
        print("=" * 70 + "\n")

    @staticmethod
    def _parse_bbox(bbox_value):
        if isinstance(bbox_value, str):
            try:
                return [int(float(v.strip())) for v in bbox_value.strip("()[]").split(",")[:4]]
            except Exception:
                return [0, 0, 0, 0]
        try:
            return [int(float(v)) for v in list(bbox_value)[:4]]
        except Exception:
            return [0, 0, 0, 0]