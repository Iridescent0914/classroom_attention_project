import contextlib
import io
import json
import os


class _CallbackStream(io.TextIOBase):
    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self._buffer = ""

    def writable(self):
        return True

    def write(self, s):
        text = str(s)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._callback(line)
        return len(text)

    def flush(self):
        line = self._buffer.strip()
        if line:
            self._callback(line)
        self._buffer = ""


def _run_monitor_with_progress_capture(monitor, max_frames, progress_callback):
    if not progress_callback:
        try:
            return monitor.process(max_frames, progress_callback=None)
        except TypeError:
            return monitor.process(max_frames)

    stream = _CallbackStream(progress_callback)
    with contextlib.redirect_stdout(stream):
        try:
            result = monitor.process(max_frames, progress_callback=None)
        except TypeError:
            result = monitor.process(max_frames)
    stream.flush()
    return result


def run_detection_pipeline(args, progress_callback=None):
    """Run the existing detection flow using an argparse-like namespace."""
    from .cli import build_config
    from .detector import ClassroomMonitor

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

    import pandas as pd
    import torch

    config = build_config(args)
    if progress_callback:
        progress_callback("Preparing detection pipeline")

    monitor = ClassroomMonitor(args.video_path, config)
    if progress_callback:
        device_msg = "GPU" if torch.cuda.is_available() and config.DEVICE != -1 else "CPU"
        progress_callback(f"Running detection on {device_msg}")

    df, summary = _run_monitor_with_progress_capture(monitor, args.max_frames, progress_callback)
    monitor.print_report(summary)

    outputs = {
        "csv_path": None,
        "json_path": None,
        "video_path": None,
        "raise_hand_csv_path": None,
        "summary": summary,
        "dataframe": df,
    }

    if df is not None:
        csv_path = os.path.abspath("attention_report.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        outputs["csv_path"] = csv_path
        if progress_callback:
            progress_callback(f"Saved CSV report: {csv_path}")

        if getattr(args, "export_json", False):
            json_path = os.path.abspath(args.json_output)
            summary_payload = monitor.build_structured_report(df, summary)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, ensure_ascii=False, indent=2)
            outputs["json_path"] = json_path
            if progress_callback:
                progress_callback(f"Saved structured summary: {json_path}")

    if config.OUTPUT_VIDEO and os.path.exists(config.OUTPUT_VIDEO_PATH):
        outputs["video_path"] = os.path.abspath(config.OUTPUT_VIDEO_PATH)
        if progress_callback:
            progress_callback(f"Saved annotated video: {outputs['video_path']}")

    if monitor.raise_hand_events:
        raise_csv = os.path.abspath("raise_hand_events.csv")
        pd.DataFrame(monitor.raise_hand_events).to_csv(raise_csv, index=False, encoding="utf-8-sig")
        outputs["raise_hand_csv_path"] = raise_csv
        if progress_callback:
            progress_callback(f"Saved raise-hand events: {raise_csv}")

    return outputs


def run_llm_report_pipeline(
    summary_path,
    csv_path,
    video_path="",
    model="qwen3.6-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    api_key_env="DASHSCOPE_API_KEY",
    api_key_file="qwen_api_key.txt",
    max_keyframes=6,
    keyframe_dir="report_keyframes",
    output_path="report.md",
    temperature=0.2,
    progress_callback=None,
):
    from .reporting_llm import LLMReportGenerator

    if progress_callback:
        progress_callback("Preparing LLM report generation")

    generator = LLMReportGenerator()
    result = generator.generate_markdown_report(
        summary_path=summary_path,
        csv_path=csv_path,
        video_path=video_path,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key_file=api_key_file,
        max_keyframes=max_keyframes,
        keyframe_dir=keyframe_dir,
        output_path=output_path,
        temperature=temperature,
    )
    if progress_callback:
        progress_callback(f"Saved markdown report: {result['output_path']}")
    return result
