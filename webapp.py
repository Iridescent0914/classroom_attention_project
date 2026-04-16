#!/usr/bin/env python3
"""Local web UI for classroom attention analysis."""

import json
import mimetypes
import os
import re
import socket
import tempfile
import threading
import traceback
import urllib.parse
import uuid
import zipfile
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from classroom_attention.pipeline import run_detection_pipeline, run_llm_report_pipeline


WORKSPACE_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = WORKSPACE_ROOT / "uploads"
TASKS = {}
TASK_LOCK = threading.Lock()
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_SIZE = 512 * 1024 * 1024
MAX_LOG_LINES = 600
MAX_TASKS = 100


UPLOADS_DIR.mkdir(exist_ok=True)


def list_video_files():
    videos = []
    for base in (WORKSPACE_ROOT, UPLOADS_DIR):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                try:
                    rel = path.relative_to(WORKSPACE_ROOT)
                except ValueError:
                    continue
                videos.append(str(rel).replace("\\", "/"))
    return sorted(set(videos))


def ensure_workspace_path(path_value):
  if path_value is None:
    raise ValueError("Path is required.")
  path_value = str(path_value).strip()
  if not path_value:
    raise ValueError("Path is required.")

  candidate = (WORKSPACE_ROOT / path_value).resolve() if not os.path.isabs(path_value) else Path(path_value).resolve()
  if WORKSPACE_ROOT not in [candidate, *candidate.parents]:
    raise ValueError("Only files inside the project workspace are supported by the web UI.")
  return candidate


def relative_workspace_path(path_value):
    return str(ensure_workspace_path(path_value).relative_to(WORKSPACE_ROOT)).replace("\\", "/")


def create_task(payload):
    task_id = uuid.uuid4().hex
    task = {
        "id": task_id,
        "status": "queued",
        "logs": [],
        "result": None,
        "error": None,
        "payload": payload,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with TASK_LOCK:
        prune_old_tasks_locked()
        TASKS[task_id] = task
    return task


def append_log(task_id, message):
    with TASK_LOCK:
        if task_id in TASKS:
            timestamp = datetime.now().strftime("%H:%M:%S")
            logs = TASKS[task_id]["logs"]
            logs.append(f"[{timestamp}] {message}")
            if len(logs) > MAX_LOG_LINES:
                del logs[: len(logs) - MAX_LOG_LINES]


def set_task_state(task_id, **updates):
    with TASK_LOCK:
        if task_id in TASKS:
            TASKS[task_id].update(updates)


def get_task(task_id):
    with TASK_LOCK:
        return TASKS.get(task_id)


def prune_old_tasks_locked():
    if len(TASKS) < MAX_TASKS:
        return
    ordered_ids = sorted(TASKS.keys(), key=lambda tid: TASKS[tid].get("created_at", ""))
    to_remove = len(TASKS) - MAX_TASKS + 1
    for task_id in ordered_ids[:to_remove]:
        TASKS.pop(task_id, None)


def parse_int(value, default, *, min_value=None, max_value=None):
  try:
    parsed = int(value)
  except (TypeError, ValueError):
    parsed = int(default)
  if min_value is not None:
    parsed = max(min_value, parsed)
  if max_value is not None:
    parsed = min(max_value, parsed)
  return parsed


def parse_float(value, default, *, min_value=None, max_value=None):
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    parsed = float(default)
  if min_value is not None:
    parsed = max(min_value, parsed)
  if max_value is not None:
    parsed = min(max_value, parsed)
  return parsed


def parse_bool(value, default=False):
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
      return True
    if lowered in {"0", "false", "no", "off"}:
      return False
  if value is None:
    return bool(default)
  return bool(value)


def normalize_output_path(path_value, default_name):
  normalized = str(path_value).strip() if path_value is not None else ""
  target = ensure_workspace_path(normalized or default_name)
  return str(target)


def validate_payload(raw_payload):
  if not isinstance(raw_payload, dict):
    raise ValueError("Request payload must be a JSON object.")

  video_raw = raw_payload.get("video_path", "")
  video_path = ensure_workspace_path(video_raw)
  if not video_path.exists() or not video_path.is_file():
    raise ValueError("video_path does not point to an existing file.")
  if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
    raise ValueError("video_path must be a supported video file.")

  return {
    "video_path": str(video_path),
    "threshold": parse_int(raw_payload.get("threshold", 85), 85, min_value=0, max_value=100),
    "skip_frames": parse_int(raw_payload.get("skip_frames", 2), 2, min_value=0, max_value=120),
    "save_video": parse_bool(raw_payload.get("save_video", True), True),
    "no_labels": parse_bool(raw_payload.get("no_labels", False), False),
    "output": normalize_output_path(raw_payload.get("output"), "output_annotated.mp4"),
    "json_output": normalize_output_path(raw_payload.get("json_output"), "attention_summary.json"),
    "max_frames": parse_int(raw_payload.get("max_frames", 0), 0, min_value=0),
    "conf": parse_float(raw_payload.get("conf", 0.25), 0.25, min_value=0.0, max_value=1.0),
    "imgsz": parse_int(raw_payload.get("imgsz", 1280), 1280, min_value=160, max_value=4096),
    "iou": parse_float(raw_payload.get("iou", 0.5), 0.5, min_value=0.0, max_value=1.0),
    "max_det": parse_int(raw_payload.get("max_det", 300), 300, min_value=1, max_value=2000),
    "agnostic_nms": parse_bool(raw_payload.get("agnostic_nms", False), False),
    "no_half": parse_bool(raw_payload.get("no_half", False), False),
    "kpt_vis": parse_float(raw_payload.get("kpt_vis", 0.35), 0.35, min_value=0.0, max_value=1.0),
    "short_head_down_duration": parse_float(
      raw_payload.get("short_head_down_duration", 0.45),
      0.45,
      min_value=0.0,
      max_value=10.0,
    ),
    "head_down_threshold": parse_float(raw_payload.get("head_down_threshold", 0.025), 0.025, min_value=0.0),
    "min_pose_bbox_height": parse_int(raw_payload.get("min_pose_bbox_height", 95), 95, min_value=1),
    "edge_box_margin": parse_int(raw_payload.get("edge_box_margin", 2), 2, min_value=0),
    "min_valid_kpts": parse_int(raw_payload.get("min_valid_kpts", 4), 4, min_value=0, max_value=17),
    "disable_sideways": parse_bool(raw_payload.get("disable_sideways", False), False),
    "disable_hands_low": parse_bool(raw_payload.get("disable_hands_low", False), False),
    "model": str(raw_payload.get("model", "yolov8m-pose.pt") or "yolov8m-pose.pt").strip(),
    "device": parse_int(raw_payload.get("device", 0), 0, min_value=-1, max_value=16),
    "tracker": str(raw_payload.get("tracker", "bytetrack.yaml") or "bytetrack.yaml").strip(),
    "disable_stillness": parse_bool(raw_payload.get("disable_stillness", False), False),
    "enable_stillness": parse_bool(raw_payload.get("enable_stillness", False), False),
    "min_range_sec": parse_float(raw_payload.get("min_range_sec", 2.0), 2.0, min_value=0.0),
    "min_total_sec": parse_float(raw_payload.get("min_total_sec", 3.0), 3.0, min_value=0.0),
    "min_events": parse_int(raw_payload.get("min_events", 1), 1, min_value=0),
    "font": str(raw_payload.get("font", "") or "").strip(),
    "raise_delta": parse_float(raw_payload.get("raise_delta", 0.10), 0.10, min_value=0.0),
    "raise_duration": parse_float(raw_payload.get("raise_duration", 1.5), 1.5, min_value=0.0),
    "raise_vis": parse_float(raw_payload.get("raise_vis", 0.6), 0.6, min_value=0.0, max_value=1.0),
    "generate_report": parse_bool(raw_payload.get("generate_report", True), True),
    "llm_model": str(raw_payload.get("llm_model", "qwen3.6-plus") or "qwen3.6-plus").strip(),
    "api_key_file": normalize_output_path(raw_payload.get("api_key_file"), "qwen_api_key.txt"),
    "max_keyframes": parse_int(raw_payload.get("max_keyframes", 6), 6, min_value=1, max_value=40),
    "keyframe_dir": normalize_output_path(raw_payload.get("keyframe_dir"), "report_keyframes"),
    "report_output": normalize_output_path(raw_payload.get("report_output"), "report.md"),
    "temperature": parse_float(raw_payload.get("temperature", 0.2), 0.2, min_value=0.0, max_value=2.0),
  }


def build_detection_args(payload):
    return SimpleNamespace(
    video_path=payload["video_path"],
        threshold=int(payload.get("threshold", 85)),
        skip_frames=int(payload.get("skip_frames", 2)),
        save_video=bool(payload.get("save_video", True)),
        no_labels=bool(payload.get("no_labels", False)),
        output=payload.get("output", "output_annotated.mp4"),
        export_json=True,
        json_output=payload.get("json_output", "attention_summary.json"),
        max_frames=int(payload.get("max_frames", 0)),
        conf=float(payload.get("conf", 0.25)),
        imgsz=int(payload.get("imgsz", 1280)),
        iou=float(payload.get("iou", 0.5)),
        max_det=int(payload.get("max_det", 300)),
        agnostic_nms=bool(payload.get("agnostic_nms", False)),
        no_half=bool(payload.get("no_half", False)),
        kpt_vis=float(payload.get("kpt_vis", 0.35)),
        short_head_down_duration=float(payload.get("short_head_down_duration", 0.45)),
        head_down_threshold=float(payload.get("head_down_threshold", 0.025)),
        min_pose_bbox_height=int(payload.get("min_pose_bbox_height", 95)),
        edge_box_margin=int(payload.get("edge_box_margin", 2)),
        min_valid_kpts=int(payload.get("min_valid_kpts", 4)),
        disable_sideways=bool(payload.get("disable_sideways", False)),
        disable_hands_low=bool(payload.get("disable_hands_low", False)),
        model=payload.get("model", "yolov8m-pose.pt"),
        device=int(payload.get("device", 0)),
        tracker=payload.get("tracker", "bytetrack.yaml"),
        disable_stillness=bool(payload.get("disable_stillness", False)),
        enable_stillness=bool(payload.get("enable_stillness", False)),
        min_range_sec=float(payload.get("min_range_sec", 2.0)),
        min_total_sec=float(payload.get("min_total_sec", 3.0)),
        min_events=int(payload.get("min_events", 1)),
        font=payload.get("font", ""),
        raise_delta=float(payload.get("raise_delta", 0.10)),
        raise_duration=float(payload.get("raise_duration", 1.5)),
        raise_vis=float(payload.get("raise_vis", 0.6)),
    )


def task_runner(task_id):
    task = get_task(task_id)
    if not task:
        return
    payload = task["payload"]
    set_task_state(task_id, status="running")

    try:
        append_log(task_id, "Starting classroom attention analysis")
        detection_args = build_detection_args(payload)
        detection_result = run_detection_pipeline(detection_args, progress_callback=lambda msg: append_log(task_id, msg))

        result = {
            "detection": {
                "csv_path": detection_result["csv_path"],
                "json_path": detection_result["json_path"],
                "video_path": detection_result["video_path"],
                "raise_hand_csv_path": detection_result["raise_hand_csv_path"],
            },
            "report_markdown": None,
        }

        if payload.get("generate_report", True):
            append_log(task_id, "Starting Qwen report generation")
            llm_result = run_llm_report_pipeline(
                summary_path=detection_result["json_path"] or payload.get("json_output", "attention_summary.json"),
                csv_path=detection_result["csv_path"] or "attention_report.csv",
              video_path=payload["video_path"],
                model=payload.get("llm_model", "qwen3.6-plus"),
                api_key_file=payload.get("api_key_file", "qwen_api_key.txt"),
                max_keyframes=int(payload.get("max_keyframes", 6)),
                keyframe_dir=payload.get("keyframe_dir", "report_keyframes"),
                output_path=payload.get("report_output", "report.md"),
                temperature=float(payload.get("temperature", 0.2)),
                progress_callback=lambda msg: append_log(task_id, msg),
            )
            result["llm_report"] = llm_result
            try:
                report_path = ensure_workspace_path(llm_result["output_path"])
                result["report_markdown"] = report_path.read_text(encoding="utf-8")
            except Exception:
                result["report_markdown"] = None

        append_log(task_id, "Workflow completed")
        set_task_state(task_id, status="completed", result=result)
    except Exception as exc:
        append_log(task_id, f"Error: {exc}")
        append_log(task_id, traceback.format_exc())
        set_task_state(task_id, status="failed", error=str(exc))


def spawn_task(payload):
    task = create_task(payload)
    thread = threading.Thread(target=task_runner, args=(task["id"],), daemon=True)
    thread.start()
    return task["id"]


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>课堂专注度分析工作台</title>
  <style>
    :root {
      --bg: #f1efe8;
      --panel: rgba(255, 255, 255, 0.86);
      --ink: #22211c;
      --muted: #666154;
      --primary: #0d5f5e;
      --secondary: #bf5b2c;
      --focus: #1f8890;
      --line: rgba(34, 33, 28, 0.12);
      --shadow: 0 20px 52px rgba(54, 50, 40, 0.14);
      --radius: 24px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Source Han Serif SC", "Noto Serif SC", "Microsoft YaHei", serif;
      background:
        radial-gradient(circle at 8% 12%, rgba(13,95,94,0.18), transparent 28%),
        radial-gradient(circle at 92% 8%, rgba(191,91,44,0.2), transparent 24%),
        linear-gradient(140deg, #f8f5ef 0%, #f0ece3 55%, #ebe5da 100%);
    }

    .shell {
      width: min(1340px, calc(100vw - 28px));
      margin: 18px auto 36px;
      display: grid;
      gap: 16px;
    }

    .hero,
    .panel {
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.65);
      border-radius: var(--radius);
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow);
    }

    .hero {
      padding: 30px;
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 24px;
      align-items: end;
      overflow: hidden;
      position: relative;
    }

    .hero::after {
      content: "";
      position: absolute;
      right: -80px;
      bottom: -120px;
      width: 260px;
      height: 260px;
      background: radial-gradient(circle, rgba(13,95,94,0.15), rgba(13,95,94,0));
      pointer-events: none;
    }

    .eyebrow {
      font-size: 12px;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: var(--primary);
      font-weight: 800;
      margin-bottom: 4px;
    }

    h1 {
      margin: 8px 0 10px;
      font-size: clamp(30px, 4.5vw, 58px);
      line-height: 0.98;
      font-weight: 900;
    }

    .lead {
      margin: 0;
      color: var(--muted);
      line-height: 1.78;
      font-size: 16px;
      max-width: 70ch;
    }

    .hero-cards {
      display: grid;
      gap: 12px;
    }

    .hero-card {
      border-radius: 18px;
      border: 1px solid rgba(34,33,28,0.08);
      padding: 16px 18px;
      background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(244,238,228,0.8));
    }

    .hero-card strong {
      display: block;
      font-size: 22px;
      margin-bottom: 2px;
    }

    .main-grid {
      display: grid;
      grid-template-columns: 430px minmax(0, 1fr);
      gap: 16px;
    }

    .stack { display: grid; gap: 16px; }
    .panel { padding: 22px; }
    .panel h2 { margin: 0 0 14px; font-size: 22px; }

    .step-badge {
      display: inline-block;
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--primary);
      font-weight: 800;
      margin-bottom: 6px;
    }

    label {
      display: block;
      font-size: 13px;
      font-weight: 800;
      margin-bottom: 6px;
      color: var(--muted);
    }

    input,
    select {
      width: 100%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      color: var(--ink);
      font-size: 14px;
      transition: border-color .15s ease, box-shadow .15s ease;
    }

    input:focus,
    select:focus {
      border-color: var(--focus);
      box-shadow: 0 0 0 3px rgba(31,136,144,0.18);
      outline: none;
    }

    .helper { margin-top: 8px; color: var(--muted); font-size: 12px; line-height: 1.55; }

    .file-card {
      border: 1px dashed rgba(13,95,94,0.35);
      border-radius: 18px;
      background: rgba(13,95,94,0.05);
      padding: 14px;
      margin-top: 10px;
    }

    .file-name {
      font-weight: 800;
      font-size: 14px;
      word-break: break-all;
    }

    .file-meta {
      font-size: 12px;
      color: var(--muted);
      margin-top: 5px;
    }

    .choice-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
    }

    .mini-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .toggle-row {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 8px;
    }

    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      font-weight: 700;
    }

    .toggle input { width: auto; }

    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }

    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 20px;
      font-size: 14px;
      font-weight: 800;
      cursor: pointer;
      transition: transform .15s ease, opacity .15s ease, box-shadow .15s ease;
    }

    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: .5; cursor: not-allowed; transform: none; }

    .primary {
      color: #fff;
      background: linear-gradient(135deg, var(--secondary), #dc8757);
      box-shadow: 0 8px 18px rgba(191,91,44,0.32);
    }

    .secondary {
      color: var(--ink);
      background: rgba(34,33,28,0.08);
    }

    .status-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 10px 18px;
      background: rgba(13,95,94,0.12);
      color: var(--primary);
      font-size: 14px;
      font-weight: 800;
      border: 1px solid rgba(13,95,94,0.28);
      box-shadow: 0 4px 14px rgba(13,95,94,0.12);
    }

    .status-pill.status-uploading,
    .status-pill.status-queued,
    .status-pill.status-running {
      background: rgba(191,91,44,0.14);
      color: #9c3f15;
      border-color: rgba(191,91,44,0.42);
      animation: pulse-status 1.2s ease-in-out infinite;
    }

    .status-pill.status-completed {
      background: rgba(13,95,94,0.18);
      color: #0b5857;
      border-color: rgba(13,95,94,0.45);
    }

    .status-pill.status-failed {
      background: rgba(146,40,40,0.16);
      color: #8f2525;
      border-color: rgba(146,40,40,0.45);
    }

    .status-hint {
      margin-top: 10px;
      font-size: 13px;
      font-weight: 700;
      color: #9c3f15;
      min-height: 20px;
    }

    @keyframes pulse-status {
      0% { box-shadow: 0 4px 14px rgba(191,91,44,0.16); }
      50% { box-shadow: 0 6px 20px rgba(191,91,44,0.32); }
      100% { box-shadow: 0 4px 14px rgba(191,91,44,0.16); }
    }

    .content-grid {
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 16px;
    }

    .subpanel {
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(255,255,255,0.64);
      padding: 16px;
      min-height: 260px;
    }

    .subpanel h3 { margin: 0 0 12px; font-size: 18px; }

    .video-frame {
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 16px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #d5cfc4;
      background: linear-gradient(135deg, #1f1b17, #2f2923);
      border: 1px solid rgba(255,255,255,0.12);
    }

    video { width: 100%; height: 100%; object-fit: contain; background: #101010; }

    .results-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .result-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.74);
    }

    .result-card a {
      color: var(--primary);
      text-decoration: none;
      font-weight: 800;
    }

    .muted { color: var(--muted); }
    .small { font-size: 12px; }

    .logbox,
    .markdownbox {
      border-radius: 16px;
      padding: 14px;
      min-height: 240px;
      max-height: 430px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .logbox {
      background: #181818;
      color: #ece6dc;
      font-family: Consolas, monospace;
      font-size: 13px;
      line-height: 1.58;
    }

    .markdownbox {
      background: linear-gradient(180deg, rgba(255,255,255,.94), rgba(246,239,227,.94));
      border: 1px solid rgba(34,33,28,.08);
      line-height: 1.75;
      font-size: 14px;
    }

    .markdownbox h1,
    .markdownbox h2,
    .markdownbox h3 { line-height: 1.2; }

    @media (max-width: 1060px) {
      .hero,
      .main-grid,
      .content-grid { grid-template-columns: 1fr; }
      .results-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <div class="eyebrow">Smart Classroom Intelligence Lab</div>
        <h1>课堂专注度分析<br/>演示工作台</h1>
        <p class="lead">本页面将“姿态检测 + 注意力评分 + Qwen 报告生成”串成一步式流程。你只需从本地文件夹选择一个视频，系统会自动上传到服务端并执行分析，最后在同页查看日志、输出文件和报告正文。</p>
      </div>
      <div class="hero-cards">
        <div class="hero-card">
          <strong>1. 本地选视频</strong>
          <span class="muted">仅支持本地文件夹选择，不再使用服务端已有视频列表</span>
        </div>
        <div class="hero-card">
          <strong>2. 一键运行</strong>
          <span class="muted">自动完成上传、检测、统计与报告生成</span>
        </div>
        <div class="hero-card">
          <strong>3. 结果集中展示</strong>
          <span class="muted">标注视频、CSV/JSON 与 Markdown 报告统一查看</span>
        </div>
      </div>
    </section>

    <section class="main-grid">
      <div class="stack">
        <section class="panel">
          <div class="step-badge">Step 01</div>
          <h2>选择视频来源</h2>
          <label for="videoUpload">本地视频文件</label>
          <input id="videoUpload" type="file" accept="video/*" />
          <div class="helper">提示：浏览器仅允许你通过系统文件选择器挑选本地视频。选中后会先在页面预览，再在点击“开始分析”时上传到项目目录下的 uploads/。</div>

          <div style="margin-top:12px;">
            <label for="existingVideoSelect">或选择 uploads 中已有视频</label>
            <div class="choice-row">
              <select id="existingVideoSelect">
                <option value="">-- 请选择已有视频 --</option>
              </select>
              <button class="secondary" id="refreshExistingBtn" type="button">刷新</button>
            </div>
            <div class="helper">可直接复用 uploads/ 里的历史视频，无需重复上传。</div>
          </div>

          <div class="file-card">
            <div class="file-name" id="localFileName">尚未选择文件</div>
            <div class="file-meta" id="localFileMeta">支持 mp4 / avi / mov / mkv / webm</div>
          </div>
          <div class="actions">
            <button class="secondary" id="clearFileBtn">清空选择</button>
          </div>
        </section>

        <section class="panel">
          <div class="step-badge">Step 02</div>
          <h2>运行参数配置</h2>
          <div class="mini-grid">
            <div><label for="threshold">专注阈值</label><input id="threshold" type="number" value="85" min="0" max="100" /></div>
            <div><label for="skipFrames">跳帧数</label><input id="skipFrames" type="number" value="2" min="0" /></div>
            <div><label for="maxFrames">最大处理帧数</label><input id="maxFrames" type="number" value="0" min="0" /></div>
            <div><label for="device">运行设备</label><select id="device"><option value="0">GPU 0</option><option value="-1">CPU</option></select></div>
            <div><label for="outputVideo">标注视频输出</label><input id="outputVideo" value="output_annotated.mp4" /></div>
            <div><label for="outputJson">结构化 JSON 输出</label><input id="outputJson" value="attention_summary.json" /></div>
            <div><label for="reportOutput">Markdown 报告输出</label><input id="reportOutput" value="report.md" /></div>
            <div><label for="maxKeyframes">报告关键帧数量</label><input id="maxKeyframes" type="number" value="6" min="1" max="20" /></div>
            <div><label for="temperature">Qwen Temperature</label><input id="temperature" type="number" value="0.2" min="0" max="2" step="0.1" /></div>
          </div>
          <div class="toggle-row">
            <label class="toggle"><input id="saveVideo" type="checkbox" checked />保存标注视频</label>
            <label class="toggle"><input id="noLabels" type="checkbox" />隐藏框标签</label>
            <label class="toggle"><input id="generateReport" type="checkbox" checked />生成 Qwen 报告</label>
          </div>
          <div class="actions">
            <button class="primary" id="startBtn">开始分析</button>
          </div>
        </section>
      </div>

      <div class="stack">
        <section class="panel">
          <div class="status-bar">
            <h2 style="margin:0">执行与输出</h2>
            <div class="status-pill" id="statusPill">尚未启动</div>
          </div>
          <div class="status-hint" id="statusHint">等待启动任务…</div>
          <div class="content-grid">
            <div class="subpanel">
              <h3>原视频预览</h3>
              <div class="video-frame" id="videoFrame"><span class="muted">请选择本地视频进行预览</span></div>
            </div>
            <div class="subpanel">
              <h3>产出文件</h3>
              <div class="results-grid" id="results"></div>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="content-grid">
            <div class="subpanel">
              <h3>实时日志</h3>
              <div class="logbox" id="logs">等待任务开始…</div>
            </div>
            <div class="subpanel">
              <h3>报告预览</h3>
              <div class="markdownbox" id="reportPreview">生成 report.md 后会在这里直接展示。</div>
            </div>
          </div>
        </section>
      </div>
    </section>
  </div>

  <script>
    let currentTaskId = null;
    let timer = null;
    let selectedFile = null;
    let selectedExistingVideo = '';
    let localPreviewUrl = '';
    let uploadedVideoPath = '';

    function escapeHtml(text) {
      return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function markdownToHtml(markdown) {
      const lines = String(markdown || '').split(/\r?\n/);
      return lines.map(line => {
        if (/^###\s+/.test(line)) return `<h3>${escapeHtml(line.replace(/^###\s+/, ''))}</h3>`;
        if (/^##\s+/.test(line)) return `<h2>${escapeHtml(line.replace(/^##\s+/, ''))}</h2>`;
        if (/^#\s+/.test(line)) return `<h1>${escapeHtml(line.replace(/^#\s+/, ''))}</h1>`;
        if (/^-\s+/.test(line)) return `<div>• ${escapeHtml(line.replace(/^-\s+/, ''))}</div>`;
        if (!line.trim()) return '<div style="height:10px"></div>';
        return `<div>${escapeHtml(line)}</div>`;
      }).join('');
    }

    function formatFileSize(bytes) {
      const size = Number(bytes || 0);
      if (size < 1024) return `${size} B`;
      if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
      if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
      return `${(size / (1024 * 1024 * 1024)).toFixed(2)} GB`;
    }

    function setBusy(isBusy) {
      const btn = document.getElementById('startBtn');
      btn.disabled = isBusy;
      btn.textContent = isBusy ? '处理中...' : '开始分析';
    }

    function updateStatus(status, extraHint) {
      const pill = document.getElementById('statusPill');
      const hint = document.getElementById('statusHint');
      const key = String(status || 'idle').toLowerCase();
      pill.className = `status-pill status-${key}`;

      if (key === 'uploading') {
        pill.textContent = '上传中';
        hint.textContent = extraHint || '正在上传视频，请稍候…';
        return;
      }
      if (key === 'queued') {
        pill.textContent = '排队中';
        hint.textContent = extraHint || '任务已提交，等待开始执行。';
        return;
      }
      if (key === 'running') {
        pill.textContent = '执行中';
        hint.textContent = extraHint || '模型推理和报告生成中，处理耗时与视频长度相关，请耐心等待。';
        return;
      }
      if (key === 'completed') {
        pill.textContent = '已完成';
        hint.textContent = extraHint || '任务完成，可查看标注视频、CSV/JSON 与报告内容。';
        return;
      }
      if (key === 'failed') {
        pill.textContent = '失败';
        hint.textContent = extraHint || '任务失败，请查看实时日志中的报错信息。';
        return;
      }

      pill.textContent = '尚未启动';
      hint.textContent = extraHint || '等待启动任务…';
    }

    function renderVideoPreviewFromSource(src) {
      const frame = document.getElementById('videoFrame');
      if (!src) {
        frame.innerHTML = '<span class="muted">请选择本地视频进行预览</span>';
        return;
      }
      frame.innerHTML = `<video controls preload="metadata" src="${src}"></video>`;
    }

    function setSelectedFile(file) {
      selectedFile = file || null;
      selectedExistingVideo = '';
      uploadedVideoPath = '';
      document.getElementById('existingVideoSelect').value = '';

      if (localPreviewUrl) {
        URL.revokeObjectURL(localPreviewUrl);
        localPreviewUrl = '';
      }

      const nameNode = document.getElementById('localFileName');
      const metaNode = document.getElementById('localFileMeta');

      if (!selectedFile) {
        nameNode.textContent = '尚未选择文件';
        metaNode.textContent = '支持 mp4 / avi / mov / mkv / webm';
        renderVideoPreviewFromSource('');
        return;
      }

      nameNode.textContent = selectedFile.name;
      metaNode.textContent = `${formatFileSize(selectedFile.size)} · ${selectedFile.type || 'video/*'}`;
      localPreviewUrl = URL.createObjectURL(selectedFile);
      renderVideoPreviewFromSource(localPreviewUrl);
    }

    function clearSelection() {
      const input = document.getElementById('videoUpload');
      input.value = '';
      setSelectedFile(null);
      selectedExistingVideo = '';
      document.getElementById('existingVideoSelect').value = '';
    }

    function useExistingVideo(path) {
      selectedExistingVideo = path || '';
      if (!selectedExistingVideo) {
        return;
      }

      selectedFile = null;
      uploadedVideoPath = selectedExistingVideo;
      const uploadInput = document.getElementById('videoUpload');
      uploadInput.value = '';

      if (localPreviewUrl) {
        URL.revokeObjectURL(localPreviewUrl);
        localPreviewUrl = '';
      }

      const nameNode = document.getElementById('localFileName');
      const metaNode = document.getElementById('localFileMeta');
      nameNode.textContent = `已选择已有视频: ${selectedExistingVideo}`;
      metaNode.textContent = '来源: uploads 文件夹';
      renderVideoPreviewFromSource('/api/file?path=' + encodeURIComponent(selectedExistingVideo));
    }

    async function loadExistingVideos(preferred) {
      const select = document.getElementById('existingVideoSelect');
      const keep = preferred || selectedExistingVideo || '';
      const res = await fetch('/api/videos');
      const data = await res.json();
      const videos = (data.videos || []).filter(v => String(v).startsWith('uploads/'));

      select.innerHTML = '<option value="">-- 请选择已有视频 --</option>';
      videos.forEach(video => {
        const option = document.createElement('option');
        option.value = video;
        option.textContent = video;
        select.appendChild(option);
      });

      if (keep && videos.includes(keep)) {
        select.value = keep;
        useExistingVideo(keep);
      }
    }

    function renderResults(result) {
      const container = document.getElementById('results');
      container.innerHTML = '';
      if (!result) return;

      const items = [];
      if (result.detection) {
        items.push(['CSV 明细', result.detection.csv_path, true]);
        items.push(['JSON 摘要', result.detection.json_path, true]);
        items.push(['标注视频', result.detection.video_path, true]);
        items.push(['举手事件', result.detection.raise_hand_csv_path, true]);
      }
      if (result.llm_report) {
        items.push(['Markdown 报告', result.llm_report.output_path, true]);
        items.push(['关键帧目录', result.llm_report.keyframe_dir, true]);
      }

      items.filter(item => item[1]).forEach(([label, path, preferDownload]) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        const openHref = '/api/file?path=' + encodeURIComponent(path);
        const downloadHref = '/api/file?path=' + encodeURIComponent(path) + '&download=1';
        const actionLink = preferDownload
          ? `<a href="${downloadHref}" target="_blank">下载</a><span class="muted small"> | </span><a href="${openHref}" target="_blank">打开</a>`
          : `<a href="${openHref}" target="_blank">打开</a>`;
        card.innerHTML = `<div><strong>${escapeHtml(label)}</strong></div><div class="muted small" style="margin:6px 0 10px;">${escapeHtml(path)}</div>${actionLink}`;
        container.appendChild(card);
      });
    }

    function renderReport(markdown) {
      const panel = document.getElementById('reportPreview');
      panel.innerHTML = markdown ? markdownToHtml(markdown) : '生成 report.md 后会在这里直接展示。';
    }

    async function uploadVideo(file) {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/api/upload', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Upload failed');
      return data.path;
    }

    async function pollTask() {
      if (!currentTaskId) return;
      const res = await fetch(`/api/task?id=${encodeURIComponent(currentTaskId)}`);
      const data = await res.json();
      updateStatus(data.status);
      const logsNode = document.getElementById('logs');
      logsNode.textContent = (data.logs || []).join('\n') || '暂无日志';
      logsNode.scrollTop = logsNode.scrollHeight;
      renderResults(data.result);
      renderReport(data.result && data.result.report_markdown);

      if (data.status === 'completed' || data.status === 'failed') {
        clearInterval(timer);
        timer = null;
        setBusy(false);
      }
    }

    async function startTask() {
      try {
        let videoPath = selectedExistingVideo;
        if (!selectedFile && !videoPath) {
          alert('请先选择本地视频，或从 uploads 中选择已有视频。');
          return;
        }

        setBusy(true);
        updateStatus('uploading');

        if (selectedFile && !uploadedVideoPath) {
          uploadedVideoPath = await uploadVideo(selectedFile);
          videoPath = uploadedVideoPath;
        } else if (!selectedFile && videoPath) {
          uploadedVideoPath = videoPath;
        }

        const payload = {
          video_path: videoPath,
          threshold: Number(document.getElementById('threshold').value),
          skip_frames: Number(document.getElementById('skipFrames').value),
          max_frames: Number(document.getElementById('maxFrames').value),
          device: Number(document.getElementById('device').value),
          save_video: document.getElementById('saveVideo').checked,
          no_labels: document.getElementById('noLabels').checked,
          generate_report: document.getElementById('generateReport').checked,
          output: document.getElementById('outputVideo').value.trim(),
          json_output: document.getElementById('outputJson').value.trim(),
          report_output: document.getElementById('reportOutput').value.trim(),
          max_keyframes: Number(document.getElementById('maxKeyframes').value),
          temperature: Number(document.getElementById('temperature').value),
          api_key_file: 'qwen_api_key.txt'
        };

        const res = await fetch('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || '启动任务失败');
        }

        currentTaskId = data.task_id;
  updateStatus(data.status || 'queued');
        document.getElementById('logs').textContent = '任务已提交，等待执行…';
        document.getElementById('results').innerHTML = '';
        renderReport('');
        if (timer) clearInterval(timer);
        timer = setInterval(pollTask, 2000);
        pollTask();
      } catch (err) {
        setBusy(false);
        updateStatus('failed', err.message || String(err));
        alert(err.message || String(err));
      }
    }

    document.getElementById('videoUpload').addEventListener('change', e => {
      const file = e.target.files && e.target.files[0] ? e.target.files[0] : null;
      setSelectedFile(file);
    });
    document.getElementById('existingVideoSelect').addEventListener('change', e => useExistingVideo(e.target.value));
    document.getElementById('refreshExistingBtn').addEventListener('click', () => loadExistingVideos());
    document.getElementById('clearFileBtn').addEventListener('click', clearSelection);
    document.getElementById('startBtn').addEventListener('click', startTask);
    updateStatus('idle');
    loadExistingVideos();
  </script>
</body>
</html>
"""


class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/api/videos":
            self._send_json({"videos": list_video_files()})
            return
        if parsed.path == "/api/task":
            query = urllib.parse.parse_qs(parsed.query)
            task_id = query.get("id", [""])[0]
            task = get_task(task_id)
            if not task:
                self._send_json({"error": "Task not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({k: v for k, v in task.items() if k != "payload"})
            return
        if parsed.path == "/api/file":
            query = urllib.parse.parse_qs(parsed.query)
            raw_path = query.get("path", [""])[0]
            download = query.get("download", ["0"])[0].lower() in {"1", "true", "yes", "on"}
            if not raw_path:
                self._send_json({"error": "Missing path"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                path = ensure_workspace_path(raw_path)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if not path.exists():
                self._send_json({"error": "File not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_file(path, as_attachment=download)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/run":
            try:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._send_json({"error": "Invalid Content-Length"}, status=HTTPStatus.BAD_REQUEST)
                    return
                if length <= 0:
                    self._send_json({"error": "Missing request body"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    raw_payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except json.JSONDecodeError:
                    self._send_json({"error": "Request body must be valid JSON"}, status=HTTPStatus.BAD_REQUEST)
                    return

                payload = validate_payload(raw_payload)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json(
                    {"error": f"Server failed to validate payload: {exc}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return

            task_id = spawn_task(payload)
            self._send_json({"task_id": task_id, "status": "queued"})
            return
        if parsed.path == "/api/upload":
            self._handle_upload()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _handle_upload(self):
        content_type = self.headers.get("Content-Type", "")
        match = re.search(r"boundary=(.+)", content_type)
        if "multipart/form-data" not in content_type or not match:
            self._send_json({"error": "Expected multipart/form-data upload"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length"}, status=HTTPStatus.BAD_REQUEST)
            return

        if length <= 0:
            self._send_json({"error": "Upload body is empty"}, status=HTTPStatus.BAD_REQUEST)
            return
        if length > MAX_UPLOAD_SIZE:
            self._send_json(
                {"error": f"Upload too large. Max size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."},
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        boundary = match.group(1).encode("utf-8")
        body = self.rfile.read(length)
        parts = body.split(b"--" + boundary)
        uploaded = None

        for part in parts:
            if b"Content-Disposition" not in part:
                continue
            header_blob, _, file_blob = part.partition(b"\r\n\r\n")
            header_text = header_blob.decode("utf-8", errors="ignore")
            filename_match = re.search(r'filename="([^\"]+)"', header_text)
            if not filename_match:
                continue
            filename = Path(filename_match.group(1)).name
            if not filename:
                continue
            ext = Path(filename).suffix.lower()
            if ext not in VIDEO_EXTENSIONS:
                self._send_json({"error": "Only video files are allowed."}, status=HTTPStatus.BAD_REQUEST)
                return
            file_data = file_blob.rstrip(b"\r\n-")
            if not file_data:
                continue
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
            target = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
            target.write_bytes(file_data)
            uploaded = target
            break

        if not uploaded:
            self._send_json({"error": "No file received"}, status=HTTPStatus.BAD_REQUEST)
            return

        rel = str(uploaded.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
        self._send_json({"path": rel, "name": uploaded.name})

    def log_message(self, format, *args):
        return

    @staticmethod
    def _is_client_disconnect(exc):
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return True
        if isinstance(exc, socket.error):
            return True
        return False

    def _send_html(self, body, status=HTTPStatus.OK):
        data = body.encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError as exc:
            if not self._is_client_disconnect(exc):
                raise

    def _send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError as exc:
            if not self._is_client_disconnect(exc):
                raise

    def _send_directory_as_zip(self, directory_path):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp_zip_path = Path(tmp.name)

        try:
            with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file_path in sorted(directory_path.rglob("*")):
                    if not file_path.is_file():
                        continue
                    arcname = str(file_path.relative_to(directory_path)).replace("\\", "/")
                    archive.write(file_path, arcname=arcname)
            self._send_file(tmp_zip_path, as_attachment=True, download_name=f"{directory_path.name}.zip")
        finally:
            try:
                if tmp_zip_path.exists():
                    tmp_zip_path.unlink()
            except OSError:
                pass

    def _send_file(self, path, as_attachment=False, download_name=None):
        if path.is_dir():
            if as_attachment:
                self._send_directory_as_zip(path)
                return
            listing = sorted(item.name for item in path.iterdir())
            self._send_json({"directory": str(path), "entries": listing})
            return

        mime_type, _ = mimetypes.guess_type(path.name)
        mime_type = mime_type or "application/octet-stream"
        if mime_type.startswith("text/") or mime_type in {"application/json", "application/javascript"}:
            mime_type = f"{mime_type}; charset=utf-8"

        file_size = path.stat().st_size
        start = 0
        end = max(file_size - 1, 0)
        status = HTTPStatus.OK

        range_header = self.headers.get("Range", "").strip()
        if range_header.startswith("bytes=") and file_size > 0:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                start_text, end_text = match.groups()
                try:
                    if start_text == "" and end_text:
                        suffix = int(end_text)
                        if suffix > 0:
                            start = max(file_size - suffix, 0)
                    elif start_text:
                        start = int(start_text)

                    if end_text:
                        end = min(int(end_text), file_size - 1)

                    if start >= file_size or start > end:
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{file_size}")
                        self.send_header("Accept-Ranges", "bytes")
                        self.end_headers()
                        return

                    status = HTTPStatus.PARTIAL_CONTENT
                except ValueError:
                    start = 0
                    end = file_size - 1
                    status = HTTPStatus.OK

        content_length = max(end - start + 1, 0)
        try:
            self.send_response(status)
            self.send_header("Content-Type", mime_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(content_length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            filename = download_name or path.name
            disposition = "attachment" if as_attachment else "inline"
            self.send_header("Content-Disposition", f'{disposition}; filename="{filename}"')
            self.end_headers()

            with path.open("rb") as src:
                src.seek(start)
                remaining = content_length
                chunk_size = 64 * 1024
                while remaining > 0:
                    data = src.read(min(chunk_size, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        except OSError as exc:
            if not self._is_client_disconnect(exc):
                raise


def main(host="127.0.0.1", port=6006):
    server = ThreadingHTTPServer((host, port), WebHandler)
    print(f"Web UI running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
