"""
批量关键点提取脚本 — 对预处理后的帧序列提取关键点,保存为 .npy
用法: python scripts/extract_keypoints.py --input data/processed --output data/keypoints
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.data.keypoint_extractor import KeypointExtractor
from src.data.keypoint_store import KeypointStore
from src.utils.keypoints import KeypointFrame
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def extract_from_dir(input_dir: str, output_dir: str) -> int:
    """
    对目录中所有帧子目录提取关键点
    每个子目录代表一个视频的帧序列
    Returns: 成功处理的序列数
    """
    input_path = Path(input_dir)
    extractor = KeypointExtractor()
    store = KeypointStore(output_dir=output_dir)

    if not input_path.exists():
        log.error(f"输入目录不存在: {input_dir}")
        return 0

    # 每个子目录是一个视频的帧序列
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    log.info(f"发现 {len(subdirs)} 个帧序列目录")

    count = 0
    for subdir in subdirs:
        frame_files = sorted(subdir.glob("*.jpg")) + sorted(subdir.glob("*.png"))
        if not frame_files:
            continue

        log.info(f"处理: {subdir.name} ({len(frame_files)} 帧)")
        kp_frames: list[KeypointFrame] = []

        for ff in frame_files:
            img = cv2.imread(str(ff))
            if img is None:
                continue

            # 构造VideoFrame兼容对象
            from src.data.video_capture import VideoFrame
            vf = VideoFrame(frame=img, timestamp=len(kp_frames) / 10.0, frame_idx=len(kp_frames))
            kp = extractor.extract(vf)
            if kp is not None:
                kp_frames.append(kp)

        if kp_frames:
            store.save(kp_frames, subdir.name)
            count += 1
            log.info(f"  → 保存 {len(kp_frames)} 帧关键点")

    extractor.close()
    log.info(f"完成: {count}/{len(subdirs)} 个序列")
    return count


def main():
    parser = argparse.ArgumentParser(description="批量关键点提取脚本")
    parser.add_argument("--input", default="data/processed", help="预处理帧目录")
    parser.add_argument("--output", default="data/keypoints", help="关键点输出目录")
    args = parser.parse_args()

    setup_logging()
    extract_from_dir(args.input, args.output)


if __name__ == "__main__":
    main()
