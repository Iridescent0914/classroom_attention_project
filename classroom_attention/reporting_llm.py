import argparse
import base64
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.request


class LLMReportGenerator:
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    DEFAULT_MODEL = "qwen3.6-plus"
    DEFAULT_PROMPT_TEMPLATE = "prompt_template.txt"

    def __init__(self, prompt_template_path=None):
        self.prompt_template_path = prompt_template_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            self.DEFAULT_PROMPT_TEMPLATE,
        )

    @staticmethod
    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_text(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def load_csv_rows(path, limit=20):
        rows = []
        if not path or not os.path.exists(path):
            return rows
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx >= limit:
                    break
                rows.append(row)
        return rows

    @staticmethod
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    def extract_keyframes(self, video_path, summary_payload, output_dir, max_frames):
        import cv2

        self.ensure_dir(output_dir)
        candidates = summary_payload.get("keyframe_candidates", [])[:max_frames]
        if not candidates:
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        frames = []
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for idx, candidate in enumerate(candidates, start=1):
                frame_no = int(candidate.get("frame") or 0)
                frame_no = max(0, min(frame_no, max(0, total - 1)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ok, frame = cap.read()
                if not ok:
                    continue

                file_name = (
                    f"frame_{idx:02d}_student_{int(candidate.get('student_id', 0)):02d}_"
                    f"{frame_no:06d}.jpg"
                )
                file_path = os.path.join(output_dir, file_name)
                cv2.imwrite(file_path, frame)
                frames.append(
                    {
                        "path": file_path,
                        "image_path": file_path,
                        "crop_path": None,
                        "student_id": int(candidate.get("student_id", 0)),
                        "frame": frame_no,
                        "time_sec": candidate.get("time_sec", 0),
                        "reason": candidate.get("reason", ""),
                        "score": candidate.get("score"),
                        "priority": candidate.get("priority"),
                        "bbox": candidate.get("bbox"),
                    }
                )
        finally:
            cap.release()

        return frames

    @staticmethod
    def image_to_data_url(path):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def resolve_api_key(self, env_name, key_file):
        api_key = os.environ.get(env_name, "").strip()
        if api_key:
            return api_key

        if key_file and os.path.exists(key_file):
            api_key = self.load_text(key_file).strip()
            if api_key:
                return api_key

        raise RuntimeError(
            f"Missing API key. Set environment variable `{env_name}` "
            f"or place the key in `{key_file}`."
        )

    def build_prompt(self, summary_payload, csv_rows, keyframes):
        overview = summary_payload.get("overview", {})
        students = summary_payload.get("students", [])[:8]
        video_meta = summary_payload.get("video", {})
        report_basis = {
            "students_with_not_focus_events": overview.get("students_with_not_focus_events", 0),
            "not_focus_event_count": overview.get("not_focus_event_count", 0),
            "not_focus_record_count": overview.get("not_focus_record_count", 0),
            "raise_hand_event_count": overview.get("raise_hand_event_count", 0),
        }
        keyframe_desc = [
            {
                "index": idx + 1,
                "student_id": item["student_id"],
                "time_sec": item["time_sec"],
                "frame": item["frame"],
                "reason": item["reason"],
                "score": item["score"],
                "bbox": item["bbox"],
                "is_crop": False,
            }
            for idx, item in enumerate(keyframes)
        ]

        template = self.load_text(self.prompt_template_path)
        return template.format(
            video=json.dumps(video_meta, ensure_ascii=False, indent=2),
            report_basis=json.dumps(report_basis, ensure_ascii=False, indent=2),
            overview=json.dumps(overview, ensure_ascii=False, indent=2),
            students=json.dumps(students, ensure_ascii=False, indent=2),
            csv_rows=json.dumps(csv_rows, ensure_ascii=False, indent=2),
            keyframes=json.dumps(keyframe_desc, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def normalize_report_text(report_text):
        text = report_text.strip()
        text = re.sub(r"^(#+)\s*#+\s*", r"\1 ", text, flags=re.MULTILINE)
        text = re.sub(r"^(#+)\s{2,}", r"\1 ", text, flags=re.MULTILINE)
        return text

    def call_qwen(self, api_key, model, base_url, prompt, keyframes, temperature):
        content = [{"type": "text", "text": prompt}]
        for frame in keyframes:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self.image_to_data_url(frame.get("image_path") or frame["path"])}
                }
            )

        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }

        request = urllib.request.Request(
            base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qwen API request failed: HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qwen API request failed: {exc}") from exc

        data = json.loads(body)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected API response: {body}") from exc

    def write_report(self, path, report_text, keyframes):
        lines = [self.normalize_report_text(report_text), "", "## 附录：关键帧文件"]
        for frame in keyframes:
            lines.append(
                f"- `{os.path.basename(frame.get('image_path') or frame['path'])}` | "
                f"student_id={frame['student_id']} | time_sec={frame['time_sec']} | reason={frame['reason']}"
            )
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).strip() + "\n")

    def generate_markdown_report(
        self,
        summary_path="attention_summary.json",
        csv_path="attention_report.csv",
        video_path="",
        model=DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        api_key_env="DASHSCOPE_API_KEY",
        api_key_file="qwen_api_key.txt",
        max_keyframes=6,
        keyframe_dir="report_keyframes",
        output_path="report.md",
        temperature=0.2,
    ):
        api_key = self.resolve_api_key(api_key_env, api_key_file)
        summary_payload = self.load_json(summary_path)
        resolved_video_path = video_path or summary_payload.get("video", {}).get("video_path", "")
        if not resolved_video_path:
            raise RuntimeError("Video path not found. Pass --video or include video.video_path in the summary JSON.")
        if not os.path.exists(resolved_video_path):
            raise FileNotFoundError(f"Video file not found: {resolved_video_path}")

        csv_rows = self.load_csv_rows(csv_path)
        keyframes = self.extract_keyframes(resolved_video_path, summary_payload, keyframe_dir, max_keyframes)
        prompt = self.build_prompt(summary_payload, csv_rows, keyframes)
        report_text = self.call_qwen(api_key, model, base_url, prompt, keyframes, temperature)
        self.write_report(output_path, report_text, keyframes)
        return {
            "keyframe_dir": os.path.abspath(keyframe_dir),
            "output_path": os.path.abspath(output_path),
        }


def build_llm_report_parser():
    parser = argparse.ArgumentParser(description="Generate a Qwen report from classroom summary and keyframes.")
    parser.add_argument("--summary", default="attention_summary.json", help="Structured summary JSON path")
    parser.add_argument("--csv", default="attention_report.csv", help="Detailed CSV path")
    parser.add_argument("--video", default="", help="Original video path; falls back to summary metadata")
    parser.add_argument(
        "--model",
        default=LLMReportGenerator.DEFAULT_MODEL,
        help=f"Vision model name (default: {LLMReportGenerator.DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=LLMReportGenerator.DEFAULT_BASE_URL,
        help="DashScope compatible-mode endpoint",
    )
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY", help="Environment variable holding the API key")
    parser.add_argument("--api-key-file", default="qwen_api_key.txt", help="Local API key file path")
    parser.add_argument("--max-keyframes", type=int, default=6, help="Number of keyframes to send to the model")
    parser.add_argument("--keyframe-dir", default="report_keyframes", help="Directory to save extracted keyframes")
    parser.add_argument("--output", default="report.md", help="Markdown report output path")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    return parser


def main_llm_reporter():
    parser = build_llm_report_parser()
    args = parser.parse_args()

    generator = LLMReportGenerator()
    result = generator.generate_markdown_report(
        summary_path=args.summary,
        csv_path=args.csv,
        video_path=args.video,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        api_key_file=args.api_key_file,
        max_keyframes=args.max_keyframes,
        keyframe_dir=args.keyframe_dir,
        output_path=args.output,
        temperature=args.temperature,
    )

    print(f"Keyframes saved to: {result['keyframe_dir']}")
    print(f"Report saved to: {result['output_path']}")


def run_llm_reporter():
    try:
        main_llm_reporter()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)