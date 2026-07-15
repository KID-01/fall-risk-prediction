"""
离线批处理脚本 — 对原始视频目录进行预处理(帧提取→人体检测→ROI裁剪)
用法: python scripts/preprocess_videos.py --input data/raw --output data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data.preprocess import PreprocessPipeline
from src.data.video_capture import VideoCapture
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def preprocess_video(
    video_path: str,
    output_dir: str,
    target_fps: float = 10,
    target_size: tuple[int, int] = (256, 256),
) -> int:
    """
    预处理单个视频: 提取帧 → 人体检测 → ROI裁剪 → 保存帧序列
    Returns: 成功处理的帧数
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    video_name = video_path.stem
    frame_dir = output_dir / video_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PreprocessPipeline(target_size=target_size)
    cap = VideoCapture(source=str(video_path), sample_fps=target_fps)

    if not cap.open():
        log.error(f"无法打开视频: {video_path}")
        return 0

    frame_count = 0
    try:
        for video_frame in cap.frames():
            roi, box = pipeline.process(video_frame.frame)
            if roi is not None:
                frame_path = frame_dir / f"{video_name}_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), roi)
                frame_count += 1
    finally:
        cap.close()

    log.info(f"预处理完成: {video_path.name} → {frame_count}帧 → {frame_dir}")
    return frame_count


def main():
    parser = argparse.ArgumentParser(description="视频批量预处理脚本")
    parser.add_argument("--input", default="data/raw", help="原始视频目录")
    parser.add_argument("--output", default="data/processed", help="输出目录")
    parser.add_argument("--fps", type=float, default=10, help="采样帧率")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], help="ROI尺寸")
    args = parser.parse_args()

    setup_logging()

    input_dir = Path(args.input)
    if not input_dir.exists():
        log.error(f"输入目录不存在: {args.input}")
        return

    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi"))
    log.info(f"发现 {len(video_files)} 个视频文件")

    total_frames = 0
    for i, vf in enumerate(video_files, 1):
        log.info(f"--- 处理 {i}/{len(video_files)}: {vf.name} ---")
        count = preprocess_video(
            str(vf),
            args.output,
            target_fps=args.fps,
            target_size=tuple(args.size),
        )
        total_frames += count

    log.info(f"全部完成: {len(video_files)} 个视频, 共 {total_frames} 帧已处理")


if __name__ == "__main__":
    main()
