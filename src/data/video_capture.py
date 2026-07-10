"""
视频流采集模块 — 支持RTSP拉流、本地文件、摄像头
使用OpenCV VideoCapture进行实时帧解码,支持帧采样降频
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from src.utils.config import get_config


@dataclass
class VideoFrame:
    """单帧视频数据"""

    frame: np.ndarray               # BGR图像
    timestamp: float                # 时间戳(秒)
    frame_idx: int                  # 帧序号


class VideoCapture:
    """视频流采集器,支持RTSP/文件/摄像头"""

    def __init__(
        self,
        source: str = "0",
        sample_fps: float | None = None,
        buffer_size: int = 30,
    ):
        """
        Args:
            source: 视频源 (RTSP地址/文件路径/摄像头编号)
            sample_fps: 采样帧率(None则取原始帧率)
            buffer_size: 帧缓冲区大小
        """
        config = get_config()
        self.source = source
        self.sample_fps = sample_fps or config.video.sample_fps
        self.buffer_size = buffer_size
        self._cap: cv2.VideoCapture | None = None
        self._buffer: deque[VideoFrame] = deque(maxlen=buffer_size)
        self._frame_idx = 0
        self._start_time: float | None = None
        self._last_sample_time = 0.0

    def open(self) -> bool:
        """打开视频源"""
        # RTSP流需要降低延迟
        if self.source.startswith("rtsp"):
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        else:
            self._cap = cv2.VideoCapture(self.source)
        return self._cap is not None and self._cap.isOpened()

    def _should_sample(self) -> bool:
        """根据采样帧率判断是否处理当前帧"""
        now = time.time()
        interval = 1.0 / self.sample_fps
        if now - self._last_sample_time >= interval:
            self._last_sample_time = now
            return True
        return False

    def read_frame(self) -> VideoFrame | None:
        """读取一帧(按采样率)"""
        if self._cap is None or not self._cap.isOpened():
            return None

        # 跳帧以匹配采样率
        while True:
            ret, frame = self._cap.read()
            if not ret:
                return None
            if self._start_time is None:
                self._start_time = time.time()
            if self._should_sample():
                break

        timestamp = time.time() - (self._start_time or 0)
        video_frame = VideoFrame(
            frame=frame,
            timestamp=timestamp,
            frame_idx=self._frame_idx,
        )
        self._frame_idx += 1
        self._buffer.append(video_frame)
        return video_frame

    def get_buffer_frames(self, before_seconds: float, after_seconds: float = 0) -> list[VideoFrame]:
        """获取缓冲区中指定时间范围的帧(用于视频回溯)"""
        if not self._buffer:
            return []
        latest_ts = self._buffer[-1].timestamp
        return [
            f for f in self._buffer
            if latest_ts - f.timestamp <= before_seconds
            or f.timestamp - latest_ts <= after_seconds
        ]

    def frames(self) -> Iterator[VideoFrame]:
        """迭代器: 持续读取帧直到流结束"""
        if not self.open():
            raise RuntimeError(f"无法打开视频源: {self.source}")
        try:
            while True:
                frame = self.read_frame()
                if frame is None:
                    break
                yield frame
        finally:
            self.close()

    def close(self):
        """释放资源"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
