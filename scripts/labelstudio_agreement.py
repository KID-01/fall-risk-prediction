"""
标注一致性评估脚本 — 计算两名标注者间的标注一致性

对两份训练标注文件 (labelstudio_export.py 输出格式) 计算:
  - fall_risk_label 的原始一致率与 Cohen's kappa
  - 33 个关键点的逐关键点可见性一致率与 kappa

用法:
    python scripts/labelstudio_agreement.py --a data/labelstudio/annotator_a.json --b data/labelstudio/annotator_b.json --output data/labelstudio/agreement.json
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

try:
    from sklearn.metrics import cohen_kappa_score as _sklearn_kappa
except ImportError:  # sklearn 未安装时使用纯 Python 实现
    _sklearn_kappa = None

# 33 个 MediaPipe 关键点标签 (与 PoseKeypoint 枚举顺序一致)
KEYPOINT_LABEL_NAMES: list[str] = [KEYPOINT_NAMES[i] for i in range(len(KEYPOINT_NAMES))]

DEFAULT_VISIBILITY_THRESHOLD = 0.5   # 关键点可见性判定阈值


# ==== Cohen's kappa ====

def _cohen_kappa_pure(labels_a: list, labels_b: list) -> float:
    """纯 Python 实现的 Cohen's kappa (与 sklearn 公式一致)"""
    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))
    observed = sum(1 for x, y in zip(labels_a, labels_b, strict=True) if x == y) / n
    expected = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories
    )
    if expected >= 1.0:
        # 双方均只使用同一类别: 完全一致视为 kappa=1
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's kappa 系数: 优先使用 sklearn, 否则纯 Python 实现"""
    if len(labels_a) != len(labels_b):
        raise ValueError(f"两份标签长度不一致: {len(labels_a)} vs {len(labels_b)}")
    if len(labels_a) == 0:
        raise ValueError("标签列表为空")
    # 双方均只使用同一类别: 完全一致视为 kappa=1 (sklearn 在此情形返回 nan)
    if len(set(labels_a) | set(labels_b)) == 1:
        return 1.0
    if _sklearn_kappa is not None:
        return float(_sklearn_kappa(labels_a, labels_b))
    return _cohen_kappa_pure(labels_a, labels_b)


# ==== 数据加载与对齐 ====

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


def load_annotation_file(path: str) -> list[dict]:
    """加载训练标注文件 (labelstudio_export.py 输出格式)"""
    obj = _load_json(path)
    if isinstance(obj, dict):
        obj = obj.get("samples", [])
    if not isinstance(obj, list):
        raise ValueError("标注文件必须为样本列表")
    samples: list[dict] = []
    for i, sample in enumerate(obj):
        if not isinstance(sample, dict) or "frame_id" not in sample or "keypoints" not in sample:
            raise ValueError(f"样本[{i}] 缺少 frame_id 或 keypoints 字段")
        kp = np.asarray(sample["keypoints"], dtype=float)
        if kp.shape != (33, 4):
            raise ValueError(f"样本[{i}] keypoints 必须为 (33, 4), 实际形状: {kp.shape}")
        samples.append(sample)
    return samples


def match_by_frame_id(
    samples_a: list[dict],
    samples_b: list[dict],
) -> tuple[list[tuple[dict, dict]], list[int]]:
    """
    按 frame_id 对齐两份标注

    Returns:
        (配对列表 [(样本A, 样本B), ...], 未匹配的 frame_id 列表)
    """
    index_b: dict[int, list[dict]] = {}
    for sample in samples_b:
        index_b.setdefault(int(sample["frame_id"]), []).append(sample)

    pairs: list[tuple[dict, dict]] = []
    matched: set[int] = set()
    for sample_a in samples_a:
        frame_id = int(sample_a["frame_id"])
        if frame_id in index_b:
            pairs.append((sample_a, index_b[frame_id][0]))
            matched.add(frame_id)

    unmatched = sorted({
        int(s["frame_id"])
        for s in samples_a + samples_b
        if int(s["frame_id"]) not in matched
    })
    return pairs, unmatched


# ==== 一致性指标 ====

def fall_risk_agreement(pairs: list[tuple[dict, dict]]) -> dict:
    """
    fall_risk_label 的原始一致率、Cohen's kappa 与混淆矩阵

    任一方标签为 None 的配对会被跳过 (计入 skipped)。
    """
    a_vals: list[int] = []
    b_vals: list[int] = []
    skipped = 0
    for sample_a, sample_b in pairs:
        label_a = sample_a.get("fall_risk_label")
        label_b = sample_b.get("fall_risk_label")
        if label_a is None or label_b is None:
            skipped += 1
            continue
        a_vals.append(int(label_a))
        b_vals.append(int(label_b))

    if not a_vals:
        return {
            "n": 0,
            "n_agree": 0,
            "raw_agreement": 0.0,
            "kappa": None,
            "confusion_matrix": {},
            "skipped": skipped,
        }

    n = len(a_vals)
    n_agree = sum(1 for x, y in zip(a_vals, b_vals, strict=True) if x == y)
    categories = sorted(set(a_vals) | set(b_vals))
    confusion = {c: {d: 0 for d in categories} for c in categories}
    for x, y in zip(a_vals, b_vals, strict=True):
        confusion[x][y] += 1

    return {
        "n": n,
        "n_agree": n_agree,
        "raw_agreement": round(n_agree / n, 4),
        "kappa": round(cohen_kappa(a_vals, b_vals), 4),
        "confusion_matrix": confusion,
        "skipped": skipped,
    }


def _visibility_of(sample: dict, threshold: float) -> list[bool]:
    """提取样本的关键点可见性 (visibility > threshold)"""
    return [bool(kp[3] > threshold) for kp in sample["keypoints"]]


def visibility_agreement(
    pairs: list[tuple[dict, dict]],
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> list[dict]:
    """
    逐关键点可见性一致率与 kappa

    Returns:
        33 项 [{"index", "name", "n", "n_agree", "agreement", "kappa"}] 列表
    """
    results: list[dict] = []
    for i, name in enumerate(KEYPOINT_LABEL_NAMES):
        va: list[bool] = []
        vb: list[bool] = []
        for sample_a, sample_b in pairs:
            va.append(bool(sample_a["keypoints"][i][3] > threshold))
            vb.append(bool(sample_b["keypoints"][i][3] > threshold))
        n = len(va)
        n_agree = sum(1 for x, y in zip(va, vb, strict=True) if x == y)
        kappa = cohen_kappa(va, vb) if n else None
        results.append({
            "index": i,
            "name": name,
            "n": n,
            "n_agree": n_agree,
            "agreement": round(n_agree / n, 4) if n else 0.0,
            "kappa": round(kappa, 4) if kappa is not None else None,
        })
    return results


def compute_agreement(
    a_path: str,
    b_path: str,
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> dict:
    """加载两份标注文件并计算完整一致性报告"""
    samples_a = load_annotation_file(a_path)
    samples_b = load_annotation_file(b_path)
    pairs, unmatched = match_by_frame_id(samples_a, samples_b)
    return {
        "annotator_a": a_path,
        "annotator_b": b_path,
        "n_a": len(samples_a),
        "n_b": len(samples_b),
        "n_matched": len(pairs),
        "unmatched_frames": unmatched,
        "visibility_threshold": threshold,
        "fall_risk": fall_risk_agreement(pairs),
        "visibility": visibility_agreement(pairs, threshold),
    }


# ==== 主入口 ====

def main() -> int:
    """主入口: 解析命令行参数并输出一致性报告"""
    parser = argparse.ArgumentParser(
        description="标注一致性评估脚本: 计算两名标注者间的 Cohen's kappa 与关键点可见性一致率"
    )
    parser.add_argument("--a", required=True, help="标注者 A 的训练标注文件 (labelstudio_export.py 输出)")
    parser.add_argument("--b", required=True, help="标注者 B 的训练标注文件")
    parser.add_argument("--output", default="", help="输出一致性报告 JSON 文件路径 (不指定则仅打印)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_VISIBILITY_THRESHOLD,
        help="关键点可见性判定阈值 (默认 0.5)",
    )
    args = parser.parse_args()

    setup_logging()
    report = compute_agreement(args.a, args.b, args.threshold)

    fall_risk = report["fall_risk"]
    print("=" * 70)
    print("标注一致性评估结果")
    print("=" * 70)
    print(f"  标注 A: {args.a} ({report['n_a']} 条)")
    print(f"  标注 B: {args.b} ({report['n_b']} 条)")
    print(f"  按 frame_id 配对: {report['n_matched']} 条, 未匹配帧: {report['unmatched_frames']}")
    print(
        f"  风险等级 原始一致率: {fall_risk['raw_agreement']}  "
        f"(n={fall_risk['n']}, 一致 {fall_risk['n_agree']}, 跳过 {fall_risk['skipped']})"
    )
    print(f"  风险等级 Cohen's kappa: {fall_risk['kappa']}")
    print("  逐关键点可见性一致率:")
    for item in report["visibility"]:
        print(
            f"    [{item['index']:>2}] {item['name']:<18} "
            f"一致率={item['agreement']:<8} kappa={item['kappa']}  (n={item['n']})"
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"一致性报告已写入: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
