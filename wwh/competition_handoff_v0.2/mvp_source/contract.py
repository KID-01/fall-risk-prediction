# -*- coding: utf-8 -*-
"""fall_mvp 共享 JSON 契约与配置载入（Day-1）。

工程用途：4 天 MVP v0.1。
边界：非科研证据；motion_risk_score 为工程启发，非校准概率。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PersonBox:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    keypoints_present: bool


@dataclass
class VisualFrame:
    """visual 模块输出（每帧；motion_heuristic_score 为工程启发，非校准概率，非冻结 v1.0 LR）。"""
    timestamp: float
    source: str = "motion_heuristic_v0"  # 与冻结 predictor 严格区分
    motion_heuristic_score: float | None = None
    persons: list[dict] = field(default_factory=list)
    alarm: bool = False


@dataclass
class ObjectDet:
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class YoloFrame:
    """环境 YOLO 检测模块输出（每帧）。"""
    timestamp: float
    source: str
    objects: list[dict] = field(default_factory=list)  # ObjectDet dicts
    persons: list[dict] = field(default_factory=list)


@dataclass
class EnvRiskFrame:
    """Day-2 Environment Risk Engine 输出（每帧）。"""
    timestamp: float
    source: str
    person: dict | None
    objects: list[dict] = field(default_factory=list)
    environment_risk_score: float = 0.0
    top_hazards: list[dict] = field(default_factory=list)  # [{class, confidence, normalized_distance, risk_contribution}]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_config() -> dict[str, Any]:
    return {
        "visual": {
            "model": "weights/pose/yolo26n-pose.pt",
            "conf": 0.25,
            "imgsz": 640,
            "motion_window_s": 0.3,
            "source_tag": "motion_heuristic_v0",   # 工程启发；非冻结 v1.0 LR
        },
        "yolo": {
            "model": "yolo26n.pt",  # 预训练 COCO；4 天策略：最小已支持预训练
            "conf": 0.25,
            "imgsz": 640,
            "source_tag": "yolo_enviro_coco",
        },
        "environment_risk": {
            # 工程启发权重（MVP），非科研优化；类别须来自实际模型 COCO 清单
            "object_risk_table": {
                "chair": 0.6,
                "couch": 0.4,
                "bed": 0.3,
                "dining table": 0.5,
                "backpack": 0.5,
                "suitcase": 0.7,
                "sports ball": 0.7,
                "laptop": 0.3,
            },
            "neutral_classes": ["tv", "remote", "dog"],  # Day-2：v0.1 中性/忽略
            "person_conf_min": 0.25,
            "object_conf_min": 0.25,
            "distance_scale": "person_bbox_height",
            "near_thr": 0.4,    # 收紧：仅非常接近脚部视为近
            "far_thr": 1.5,     # 超过 1.5 倍身高基本不贡献
            "foot_band_gate": True,   # 仅与 person 下 1/2 区域有垂直重叠的物体显著贡献
            "risk_clip": 1.0,
            "top_k_hazards": 3,
        },
        "fusion": {
            "mode": "rule_table",        # 确定性规则表（非加权和主路径）
            "motion_thr": [0.3, 0.6],    # 工程边界，非科学最优
            "env_thr": [0.4, 0.7],       # 工程边界，非科学最优
            "person_match_min_iou": 0.15,  # 跨模型(pose/COCO)框差异容忍（工程确定性，非科研）
            "person_match_center_ratio": 0.7,
            "sync_tol_sec": 0.05,
            "persistence_frames": 3,     # 因果稳定化（<=1 关闭）
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """每行一个 JSON 对象。"""
    import json

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def frame_dict(dc: Any) -> dict:
    return asdict(dc)