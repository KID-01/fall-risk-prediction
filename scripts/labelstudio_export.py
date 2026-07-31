"""
LabelStudio 标注导出脚本 — 将 LabelStudio 标注导出 JSON 转换为训练标注格式

将 LabelStudio 导出的任务标注 (keypointlabels + fall_risk 选择) 转换为模型训练
所需的标注格式:

    [{"frame_id": 0, "timestamp": 0.0, "source": "a.mp4",
      "keypoints": [[x, y, z, v] x33], "fall_risk_label": 0}, ...]

其中 fall_risk_label 取值 0-3, 对应 src/alerts/engine.py 的
RiskLevel: 0=低风险, 1=关注级, 2=预警级, 3=高危级。

用法:
    python scripts/labelstudio_export.py --input data/labelstudio/export.json --output data/labelstudio/annotations.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.utils.keypoints import KEYPOINT_NAMES
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

# ==== 常量 ====

# 33 个 MediaPipe 关键点标签 (与 PoseKeypoint 枚举顺序一致)
KEYPOINT_LABEL_NAMES: list[str] = [KEYPOINT_NAMES[i] for i in range(len(KEYPOINT_NAMES))]

# 风险等级标签 → 0-3 整数 (与 RiskLevel.priority 一致)
LABEL_TO_LEVEL: dict[str, int] = {
    "低风险": 0,
    "关注级": 1,
    "预警级": 2,
    "高危级": 3,
    # 兼容别名
    "无风险": 0,
    "中风险": 2,
    "高风险": 3,
    "low": 0,
    "attention": 1,
    "warning": 2,
    "critical": 3,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
}


# ==== 数据加载 ====

def _load_json(data: dict | str | Path) -> list | dict:
    """加载 JSON: 支持 dict / 文件路径 / JSON 字符串"""
    if isinstance(data, dict):
        return data
    path = Path(data)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败: {path} ({exc})") from exc
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败: {exc}") from exc
    raise ValueError(f"不支持的数据类型: {type(data)}")


def load_annotation_export(data: dict | str | Path) -> list[dict]:
    """
    加载 LabelStudio 标注导出 JSON

    支持: 任务列表 [task, ...], 或 {"tasks": [task, ...]}, 或单个任务字典。
    每个任务至少包含 data 字段, 可包含 annotations (result 列表)。
    """
    obj = _load_json(data)
    if isinstance(obj, list):
        tasks = obj
    elif isinstance(obj, dict):
        tasks = obj.get("tasks", [obj])
    else:
        raise ValueError("标注导出 JSON 必须为任务列表或包含 tasks 的字典")
    return [t for t in tasks if isinstance(t, dict)]


# ==== 标注提取 ====

def _annotations_of(task: dict) -> list[dict]:
    """返回任务中已完成标注的 annotation 列表 (跳过 LabelStudio skipped 的标注)"""
    return [
        annotation
        for annotation in (task.get("annotations", []) or [])
        if not annotation.get("skipped")
    ]


def extract_fall_risk_label(task: dict) -> int | None:
    """
    提取任务的风险等级标签

    从 annotations 的 choices 结果 (from_name="fall_risk") 中读取,
    返回 0-3 整数; 无标注或标签无法识别时返回 None。
    """
    for annotation in _annotations_of(task):
        for result in annotation.get("result", []):
            if result.get("from_name") == "fall_risk" and result.get("type") == "choices":
                choices = (result.get("value") or {}).get("choices", [])
                if choices:
                    return LABEL_TO_LEVEL.get(str(choices[0]))
    return None


def extract_keypoint_results(task: dict) -> dict[str, dict]:
    """
    提取 keypointlabels 标注结果

    Returns:
        {关键点标签名: {"x": 百分比, "y": 百分比}} 映射
    """
    results: dict[str, dict] = {}
    for annotation in _annotations_of(task):
        for result in annotation.get("result", []):
            if result.get("from_name") == "pose" and result.get("type") == "keypointlabels":
                value = result.get("value") or {}
                for label in value.get("keypointlabels", []):
                    results[str(label)] = {
                        "x": float(value.get("x", 0.0)),
                        "y": float(value.get("y", 0.0)),
                    }
    return results


def _coerce_keypoints(raw: list | np.ndarray) -> list[list[float]]:
    """校验关键点数组并转换为 (33, 4) 列表, 非法时抛出 ValueError"""
    arr = np.asarray(raw, dtype=float)
    if arr.shape != (33, 4):
        raise ValueError(f"关键点数组必须为 (33, 4), 实际形状: {arr.shape}")
    return arr.tolist()


def keypoints_from_task(task: dict) -> list[list[float]] | None:
    """
    组装单帧 (33, 4) 关键点数组

    基础坐标为任务 data.keypoints (导入时预填充);
    若标注存在 keypointlabels 结果, 用标注坐标覆盖 x/y 并将可见性置 1.0,
    未被标注的关键点可见性置 0.0。
    """
    raw = (task.get("data") or {}).get("keypoints")
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float)
    if arr.shape != (33, 4):
        raise ValueError(f"data.keypoints 必须为 (33, 4), 实际形状: {arr.shape}")
    kp = arr.copy()
    annotated = extract_keypoint_results(task)
    if annotated:
        for i, name in enumerate(KEYPOINT_LABEL_NAMES):
            if name in annotated:
                kp[i, 0] = annotated[name]["x"] / 100.0
                kp[i, 1] = annotated[name]["y"] / 100.0
                kp[i, 3] = 1.0
            else:
                kp[i, 3] = 0.0
    return kp.tolist()


# ==== 训练样本转换 ====

def task_to_samples(task: dict) -> list[dict]:
    """
    单个任务 → 训练样本列表

    frames 模式任务 (data.frame_id): 返回 1 个样本;
    clips 模式任务 (data.frame_ids): 每帧展开为 1 个样本, 共享片段风险标签。
    """
    data = task.get("data") or {}
    label = extract_fall_risk_label(task)
    source = str(data.get("source", ""))

    if "frame_ids" in data and isinstance(data["frame_ids"], list):
        raw = data.get("keypoints")
        if raw is None:
            return []
        frame_ids = [int(x) for x in data["frame_ids"]]
        if not isinstance(raw, list) or len(raw) != len(frame_ids):
            raise ValueError(f"clips 任务 frame_ids 与 keypoints 数量不一致 (task id={task.get('id')})")
        timestamps = [float(t) for t in data.get("timestamps", [0.0] * len(frame_ids))]
        return [
            {
                "frame_id": frame_id,
                "timestamp": timestamps[i] if i < len(timestamps) else 0.0,
                "source": source,
                "keypoints": _coerce_keypoints(raw[i]),
                "fall_risk_label": label,
            }
            for i, frame_id in enumerate(frame_ids)
        ]

    kp = keypoints_from_task(task)
    if kp is None:
        return []
    return [
        {
            "frame_id": int(data.get("frame_id", 0)),
            "timestamp": float(data.get("timestamp", 0.0)),
            "source": source,
            "keypoints": kp,
            "fall_risk_label": label,
        }
    ]


def convert_annotations(export: list[dict]) -> list[dict]:
    """标注导出任务列表 → 训练标注样本列表"""
    samples: list[dict] = []
    for task in export:
        samples.extend(task_to_samples(task))
    return samples


# ==== 主入口 ====

def main() -> int:
    """主入口: 解析命令行参数并执行转换"""
    parser = argparse.ArgumentParser(
        description="LabelStudio 标注导出脚本: 将标注导出 JSON 转换为训练标注格式"
    )
    parser.add_argument("--input", required=True, help="LabelStudio 标注导出 JSON 文件路径")
    parser.add_argument("--output", required=True, help="输出训练标注 JSON 文件路径")
    args = parser.parse_args()

    setup_logging()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"输入文件不存在: {args.input}")
        return 1

    export = load_annotation_export(str(input_path))
    samples = convert_annotations(export)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"共转换 {len(export)} 个任务为 {len(samples)} 条训练样本, 已写入: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
