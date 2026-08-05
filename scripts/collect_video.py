"""
视频采集脚本 — 参数化控制,支持批量采集
用法:
  python scripts/collect_video.py --source 0 --scene walking --duration 30 --output data/raw
  python scripts/collect_video.py --batch collection_plan.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.data.video_capture import VideoCapture
from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def collect_single(
    source: str,
    scene: str,
    duration: int,
    output_dir: str,
    behavior: str = "",
    subject_id: str = "unknown",
    lighting: str = "normal",
    fps: float = 15,
    resolution: tuple[int, int] = (1280, 720),
) -> str:
    """
    采集单段视频
    Returns: 输出文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{scene}_{behavior}_{timestamp}.mp4"
    filepath = output_dir / filename

    log.info(f"开始采集: scene={scene}, duration={duration}s, source={source}")

    cap = VideoCapture(source=source, sample_fps=fps)
    if not cap.open():
        raise RuntimeError(f"无法打开视频源: {source}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(filepath),
        fourcc,
        fps,
        resolution,
    )

    frame_count = 0
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            video_frame = cap.read_frame()
            if video_frame is None:
                break
            writer.write(video_frame.frame)
            frame_count += 1
    finally:
        writer.release()
        cap.close()

    # 生成元数据JSON
    metadata = {
        "filename": filename,
        "scene": scene,
        "behavior": behavior,
        "subject_id": subject_id,
        "lighting": lighting,
        "duration_sec": duration,
        "fps": fps,
        "resolution": list(resolution),
        "frame_count": frame_count,
        "timestamp": timestamp,
        "source": source,
    }
    metadata_path = filepath.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info(f"采集完成: {filepath} ({frame_count}帧)")
    return str(filepath)


def collect_batch(plan_csv: str, output_dir: str, source: str = "0"):
    """
    批量采集: 读取CSV计划,自动循环执行
    CSV格式: scene,behavior,duration_sec,lighting,repeat
    """
    plan_path = Path(plan_csv)
    if not plan_path.exists():
        log.error(f"采集计划文件不存在: {plan_csv}")
        return

    with plan_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tasks = list(reader)

    log.info(f"批量采集: 共 {len(tasks)} 条任务")

    for i, task in enumerate(tasks, 1):
        repeat = int(task.get("repeat", 1))
        for r in range(repeat):
            log.info(f"--- 任务 {i}/{len(tasks)} 第 {r+1}/{repeat} 次 ---")
            try:
                collect_single(
                    source=task.get("source", source),
                    scene=task["scene"],
                    behavior=task.get("behavior", ""),
                    duration=int(task["duration_sec"]),
                    output_dir=output_dir,
                    lighting=task.get("lighting", "normal"),
                    subject_id=task.get("subject_id", "unknown"),
                )
            except Exception as e:
                log.error(f"采集失败: {e}")
                continue

    log.info("批量采集全部完成")


def main():
    parser = argparse.ArgumentParser(description="视频采集脚本")
    parser.add_argument("--source", default="0", help="视频源 (RTSP/文件/摄像头编号)")
    parser.add_argument("--scene", default="test", help="场景名称")
    parser.add_argument("--behavior", default="", help="行为标签")
    parser.add_argument("--duration", type=int, default=30, help="采集时长(秒)")
    parser.add_argument("--output", default="data/raw", help="输出目录")
    parser.add_argument("--batch", help="批量采集计划CSV文件路径")
    args = parser.parse_args()

    setup_logging()

    if args.batch:
        collect_batch(args.batch, args.output, args.source)
    else:
        collect_single(
            source=args.source,
            scene=args.scene,
            behavior=args.behavior,
            duration=args.duration,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
