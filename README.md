# 课堂专注度分析系统

基于 YOLO 姿态估计和多目标跟踪的课堂行为分析项目。系统可对视频中的学生进行逐帧检测、跨帧关联、时序行为判定，并输出结构化结果与可视化结果。

本仓库已完成模块化重构，当前推荐主链路为：

- 命令行检测入口：ca_gpu.py
- Web 工作台入口：webapp.py
- LLM 报告入口：llm_reporter.py

## 项目分析

### 1) 结构与内容

项目由三层组成：

- 入口层
   - ca_gpu.py：兼容入口，实际调用 classroom_attention.cli.main
   - webapp.py：本地 Web 工作台，支持异步任务、日志和文件下载
   - llm_reporter.py：兼容入口，实际调用 classroom_attention.reporting.run_llm_reporter
- 核心能力层（classroom_attention 包）
   - cli.py：参数解析与配置装配
   - pipeline.py：检测流程与 LLM 报告流程编排
   - detector.py：YOLO 跟踪推理、行为判定触发、视频标注输出
   - scoring.py：单帧评分与短期行为规则
   - tracker.py：跨帧状态管理与长期行为规则
   - reporting_summary.py：CSV 汇总、事件聚合、JSON 结构化摘要
   - reporting_llm.py：关键帧提取、提示词装配、Qwen API 调用、Markdown 报告落盘
   - config.py：统一配置默认值
- 资产与产物层
   - 模型与配置：yolov8m-pose.pt、trackers/bytetrack.yaml
   - 输入输出样例：attention_report.csv、attention_summary.json、raise_hand_events.csv、report.md、report_keyframes/

### 2) 当前工作流

#### A. 检测主工作流（CLI 或 Web 共用）

1. 读取视频与参数
2. 加载 YOLO 姿态模型，调用 track 模式获得检测框与 track_id
3. 对每个目标执行可靠性过滤（边缘框、最小高度、可见关键点数量）
4. 执行评分规则与时序规则
5. 判定是否不专注，绘制框和标签
6. 累积记录并导出文件

输出包括：

- attention_report.csv：逐条不专注记录
- attention_summary.json：结构化摘要（需要启用导出）
- output_annotated.mp4：标注视频（需要启用保存）
- raise_hand_events.csv：举手事件（存在时输出）

#### B. LLM 报告工作流（可选）

1. 读取 attention_summary.json 与 attention_report.csv
2. 从原视频提取关键帧
3. 使用 prompt_template.txt 组装提示词
4. 调用 Qwen 多模态接口
5. 生成 report.md

Web 工作台会在检测后按配置自动触发该流程。

## 架构图

```text
输入视频
   -> YOLO Pose + ByteTrack
   -> 关键点可靠性过滤
   -> 短期评分规则(scoring)
   -> 长期时序规则(tracker)
   -> 不专注记录聚合(reporting_summary)
   -> CSV/JSON/标注视频
   -> (可选) 关键帧抽取 + Qwen 生成 Markdown 报告
```

## 环境要求

- Python 3.9+
- Windows / Linux / macOS
- 推荐 NVIDIA GPU（也支持 CPU）

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1) 命令行检测

```bash
python ca_gpu.py classroom.mp4 --save-video --export-json --json-output attention_summary.json
```

### 2) 仅生成 LLM 报告

```bash
python llm_reporter.py --summary attention_summary.json --csv attention_report.csv --video classroom.mp4 --output report.md
```

### 3) 启动 Web 工作台

```bash
python webapp.py
```

浏览器打开：

```text
http://127.0.0.1:8765
```

## 命令行参数（核心）

以下默认值来自当前 cli.py：

| 参数 | 说明 | 默认值 |
|---|---|---|
| video_path | 输入视频路径 | 必填 |
| --threshold | 不专注阈值，分数低于该值可能触发不专注 | 85 |
| --skip-frames | 跳帧，实际为每 N+1 帧处理一次 | 2 |
| --save-video | 保存标注视频 | 关闭 |
| --output | 标注视频路径 | output_annotated.mp4 |
| --export-json | 导出结构化 JSON | 关闭 |
| --json-output | JSON 输出路径 | attention_summary.json |
| --max-frames | 最多处理帧数，0 表示全部 | 0 |
| --model | 姿态模型权重 | yolov8m-pose.pt |
| --device | CUDA 设备索引，-1 为 CPU | 0 |
| --tracker | 跟踪配置 | bytetrack.yaml |

行为相关参数：

- --kpt-vis：关键点可见性阈值（默认 0.35）
- --short-head-down-duration：短时低头持续时间阈值（默认 0.45 秒）
- --head-down-threshold：低头判定阈值（默认 0.025）
- --enable-stillness / --disable-stillness：启用或禁用发呆检测（默认关闭）
- --raise-delta / --raise-duration / --raise-vis：举手检测参数

## 常用运行示例

```bash
# 快速测试前 1000 帧
python ca_gpu.py classroom.mp4 --save-video --max-frames 1000

# 提高召回（远距离学生更友好）
python ca_gpu.py classroom.mp4 --save-video --conf 0.20 --imgsz 1536 --skip-frames 1

# CPU 模式
python ca_gpu.py classroom.mp4 --save-video --device -1

# 只做检测，不显示文字标签
python ca_gpu.py classroom.mp4 --save-video --no-labels
```

## 输出文件说明

### attention_report.csv

逐条不专注记录，字段包括：

- student_id
- time_sec / time_str
- frame
- score
- reason
- bbox

### attention_summary.json

结构化摘要，包含：

- video：视频元信息
- overview：全局统计（学生数量、事件数、举手事件数）
- students：按学生聚合后的时段、主因、累计时长
- keyframe_candidates：候选关键帧

### raise_hand_events.csv

当检测到持续举手行为时输出，用于参与度辅助分析。

### report.md

由 LLM 自动生成的课堂行为文字报告，末尾附关键帧文件索引。

## 推荐调参策略

- 画面远、目标小：降低 --conf，增大 --imgsz
- 跟踪不稳定：减小 --skip-frames，优先尝试 0 或 1
- 误报较多：提高 --threshold，或提高 --kpt-vis
- 需要“发呆”规则：显式加 --enable-stillness

## 注意事项

- 当前仍保留 ca.py（旧版 DeepSORT + MediaPipe 流程），与主链路不同，不建议新流程继续基于该文件开发。
- Web 工作台会限制路径在项目工作区内，避免越权访问。
- 生成 LLM 报告前需提供有效 API Key（环境变量或 qwen_api_key.txt）。

## 目录速览

```text
classroom_attention/
   cli.py
   config.py
   detector.py
   pipeline.py
   reporting.py
   reporting_llm.py
   reporting_summary.py
   scoring.py
   tracker.py

ca_gpu.py
webapp.py
llm_reporter.py
prompt_template.txt
requirements.txt
```

