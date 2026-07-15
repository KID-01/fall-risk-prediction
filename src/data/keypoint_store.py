"""
关键点存储模块 — 时序关键点序列的保存与加载
格式: .npy 文件, shape=(T, 33, 4) [x, y, z, visibility]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.keypoints import KeypointFrame


class KeypointStore:
    """关键点时序序列存储"""

    def __init__(self, output_dir: str = "data/keypoints"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, frames: list[KeypointFrame], name: str) -> str:
        """
        保存关键点序列为 .npy 文件
        Args:
            frames: KeypointFrame 列表
            name: 文件名(不含扩展名)
        Returns: 保存路径
        """
        if not frames:
            raise ValueError("帧列表为空")

        # 组装为 (T, 33, 4) 数组
        data = np.stack([f.keypoints for f in frames], axis=0)
        filepath = self.output_dir / f"{name}.npy"
        np.save(filepath, data)
        return str(filepath)

    def load(self, name: str) -> np.ndarray:
        """
        加载关键点序列
        Returns: (T, 33, 4) numpy数组
        """
        filepath = self.output_dir / f"{name}.npy"
        if not filepath.exists():
            raise FileNotFoundError(f"关键点文件不存在: {filepath}")
        return np.load(filepath)

    def list_all(self) -> list[str]:
        """列出所有已存储的关键点文件名"""
        return [f.stem for f in self.output_dir.glob("*.npy")]

    def save_with_metadata(
        self,
        frames: list[KeypointFrame],
        name: str,
        metadata: dict | None = None,
    ) -> str:
        """保存关键点序列 + 元数据"""
        import json

        path = self.save(frames, name)
        if metadata:
            meta_path = self.output_dir / f"{name}.json"
            meta_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return path
