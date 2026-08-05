"""
LabelStudio 标注工具脚本的单元测试

覆盖 labelstudio_import / labelstudio_export / labelstudio_agreement 的纯逻辑:
关键点任务构建、标注导出转换、Cohen's kappa 计算、异常输入处理。
所有测试均为纯逻辑测试, 无需 LabelStudio 服务器、网络或外部服务。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# 将项目根目录添加到 sys.path 以导入 scripts 包
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.labelstudio_agreement import (  # noqa: E402  -- sys.path 调整后导入 scripts/
    _cohen_kappa_pure,
    cohen_kappa,
    compute_agreement,
    fall_risk_agreement,
    load_annotation_file,
    match_by_frame_id,
    visibility_agreement,
)
from scripts.labelstudio_export import (  # noqa: E402
    LABEL_TO_LEVEL,
    convert_annotations,
    extract_fall_risk_label,
    load_annotation_export,
    task_to_samples,
)
from scripts.labelstudio_import import (  # noqa: E402
    KEYPOINT_LABEL_NAMES,
    _keypoint_map_from_json,
    build_label_config,
    frame_to_task,
    frames_to_tasks,
    parse_keypoint_json,
)


# ============================================================
# 测试辅助: 合成关键点数据
# ============================================================
def _make_keypoint_frame(seed: int, visibility: float = 0.9) -> list[list[float]]:
    """构造随机 (33, 4) 关键点数组 (坐标 0-1, 可见性统一)"""
    rng = np.random.default_rng(seed)
    kps = np.zeros((33, 4))
    kps[:, :2] = rng.random((33, 2))
    kps[:, 3] = visibility
    return kps.tolist()


@pytest.fixture
def keypoint_json() -> dict:
    """两帧关键点 JSON (frames 格式)"""
    return {
        "source": "video_001.mp4",
        "fps": 10.0,
        "frames": [
            {"timestamp": 0.0, "is_valid": True, "keypoints": _make_keypoint_frame(1)},
            {"timestamp": 0.1, "is_valid": True, "keypoints": _make_keypoint_frame(2)},
        ],
    }


def _annotation_for_task(task: dict, choice: str) -> dict:
    """将任务预标注转换为标注结果 (模拟标注者确认预标注并选择风险等级)"""
    result = []
    for pred in task.get("predictions", [{}])[0].get("result", []):
        result.append({
            "from_name": "pose",
            "to_name": "image",
            "type": "keypointlabels",
            "value": dict(pred["value"]),
        })
    result.append({
        "from_name": "fall_risk",
        "to_name": "image",
        "type": "choices",
        "value": {"choices": [choice]},
    })
    return {"result": result}


def _sample(frame_id: int, label: int | None, visibility: float = 0.8) -> dict:
    """构造一条训练标注样本 (export 输出格式)"""
    kps = np.zeros((33, 4))
    kps[:, 3] = visibility
    return {
        "frame_id": frame_id,
        "timestamp": float(frame_id) / 10.0,
        "source": "video_001.mp4",
        "keypoints": kps.tolist(),
        "fall_risk_label": label,
    }


# ============================================================
# Cohen's kappa
# ============================================================
class TestCohenKappa:
    def test_perfect_agreement(self):
        assert cohen_kappa([1, 2, 3], [1, 2, 3]) == 1.0

    def test_chance_agreement(self):
        # po=0.5, pe=0.5 => kappa=0.0
        assert cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)

    def test_expected_disagreement(self):
        # 完全反向标注: po=0, pe=0.25 => kappa=-1/3
        assert cohen_kappa([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-1 / 3)

    def test_pure_implementation_matches(self):
        a = [0, 0, 1, 1, 2, 1]
        b = [0, 1, 0, 1, 1, 2]
        assert _cohen_kappa_pure(a, b) == pytest.approx(cohen_kappa(a, b))

    def test_single_category_perfect(self):
        assert cohen_kappa([0, 0, 0], [0, 0, 0]) == 1.0

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError):
            cohen_kappa([1, 2], [1])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cohen_kappa([], [])


# ============================================================
# 导入: 关键点 JSON 解析
# ============================================================
class TestParseKeypointJson:
    def test_parse_frames_format(self, keypoint_json):
        source, fps, frames = parse_keypoint_json(keypoint_json)
        assert source == "video_001.mp4"
        assert fps == 10.0
        assert len(frames) == 2
        assert len(frames[0]["keypoints"]) == 33
        assert len(frames[0]["keypoints"][0]) == 4

    def test_parse_array_format(self):
        data = {
            "source": "arr.mp4",
            "fps": 5.0,
            "keypoints": np.zeros((3, 33, 4)).tolist(),
            "timestamps": [0.0, 0.2, 0.4],
        }
        source, fps, frames = parse_keypoint_json(data)
        assert source == "arr.mp4"
        assert fps == 5.0
        assert len(frames) == 3
        assert frames[1]["timestamp"] == pytest.approx(0.2)

    def test_parse_from_file(self, tmp_path, keypoint_json):
        path = tmp_path / "kp.json"
        path.write_text(json.dumps(keypoint_json), encoding="utf-8")
        source, _fps, frames = parse_keypoint_json(str(path))
        assert source == "video_001.mp4"
        assert len(frames) == 2

    def test_malformed_json_string_raises(self):
        with pytest.raises(ValueError):
            parse_keypoint_json("{not valid json")

    def test_missing_keypoints_field_raises(self):
        with pytest.raises(ValueError):
            parse_keypoint_json({"source": "x.mp4", "frames": [{"timestamp": 0.0}]})

    def test_bad_keypoint_shape_raises(self):
        with pytest.raises(ValueError):
            parse_keypoint_json({"frames": [{"keypoints": [[0.0, 0.0, 0.0, 0.9]] * 10}]})

    def test_unrecognized_structure_raises(self):
        with pytest.raises(ValueError):
            parse_keypoint_json({"source": "x.mp4", "other": 1})


# ============================================================
# 导入: LabelStudio 任务构建
# ============================================================
class TestBuildTasks:
    def test_frames_mode_structure(self, keypoint_json):
        _s, _f, frames = parse_keypoint_json(keypoint_json)
        tasks = frames_to_tasks(
            frames,
            mode="frames",
            source="video_001.mp4",
            image_url_base="http://localhost:8080/images",
        )
        assert len(tasks) == 2
        task = tasks[0]
        assert task["data"]["frame_id"] == 0
        assert task["data"]["image"] == "http://localhost:8080/images/frame_000000.jpg"
        assert task["data"]["source"] == "video_001.mp4"
        kp = task["data"]["keypoints"]
        assert len(kp) == 33
        assert len(kp[0]) == 4

        # 可见性 0.9 > 0.5: 全部 33 个关键点均生成预标注
        predictions = task["predictions"][0]["result"]
        assert len(predictions) == 33
        for pred in predictions:
            assert pred["value"]["keypointlabels"][0] in KEYPOINT_LABEL_NAMES
            assert pred["from_name"] == "pose"
            assert pred["type"] == "keypointlabels"

    def test_visibility_threshold_filters_predictions(self, keypoint_json):
        _s, _f, frames = parse_keypoint_json(keypoint_json)
        tasks = frames_to_tasks(frames, mode="frames", visibility_threshold=0.99)
        assert "predictions" not in tasks[0]

    def test_frame_to_task_requires_33x4(self):
        with pytest.raises(ValueError):
            frame_to_task({"keypoints": [[0.0] * 4] * 5}, 0)

    def test_clips_mode(self, keypoint_json):
        _s, _f, frames = parse_keypoint_json(keypoint_json)
        tasks = frames_to_tasks(
            frames,
            mode="clips",
            clip_len=2,
            image_url_base="http://localhost:8080/clips",
        )
        assert len(tasks) == 1
        data = tasks[0]["data"]
        assert data["frame_ids"] == [0, 1]
        assert data["video"] == "http://localhost:8080/clips/clip_000000.mp4"
        assert len(data["keypoints"]) == 2

    def test_clips_mode_splits_long_sequence(self):
        frames = [
            {"timestamp": float(i), "keypoints": _make_keypoint_frame(10 + i)}
            for i in range(5)
        ]
        tasks = frames_to_tasks(frames, mode="clips", clip_len=2)
        assert len(tasks) == 3
        assert tasks[0]["data"]["frame_ids"] == [0, 1]
        assert tasks[1]["data"]["frame_ids"] == [2, 3]
        assert tasks[2]["data"]["frame_ids"] == [4]

    def test_empty_frames(self):
        assert frames_to_tasks([]) == []


# ============================================================
# 导入: 标注配置 XML
# ============================================================
class TestLabelConfig:
    def test_frames_config_contains_all_keypoints_and_choices(self):
        xml = build_label_config(scope="frames")
        for name in KEYPOINT_LABEL_NAMES:
            assert f'value="{name}"' in xml
        for choice in ["低风险", "关注级", "预警级", "高危级"]:
            assert f'value="{choice}"' in xml
        assert 'name="pose"' in xml
        assert 'name="fall_risk"' in xml
        assert "<Image" in xml

    def test_clips_config(self):
        xml = build_label_config(scope="clips")
        assert "<Video" in xml
        assert "<Image" not in xml

    def test_invalid_scope_uses_frames(self):
        xml = build_label_config(scope="unknown")
        assert "<Image" in xml


# ============================================================
# 导出: 标注 → 训练格式
# ============================================================
class TestExport:
    def test_roundtrip_import_export(self, keypoint_json):
        """导入任务 → 模拟标注 → 导出, 关键点坐标与标签应完整还原"""
        _s, _f, frames = parse_keypoint_json(keypoint_json)
        tasks = frames_to_tasks(frames, mode="frames", source="video_001.mp4")
        export_tasks = [
            {"id": i, "data": task["data"], "annotations": [_annotation_for_task(task, "关注级")]}
            for i, task in enumerate(tasks)
        ]
        samples = convert_annotations(export_tasks)
        assert len(samples) == 2
        for i, sample in enumerate(samples):
            assert sample["frame_id"] == i
            assert sample["fall_risk_label"] == 1
            assert sample["source"] == "video_001.mp4"
            kp = np.asarray(sample["keypoints"])
            assert kp.shape == (33, 4)
            # 所有关键点均被标注: 可见性 1.0, 坐标与原始一致
            original = np.asarray(frames[i]["keypoints"])
            assert np.allclose(kp[:, :2], original[:, :2], atol=0.01)
            assert np.all(kp[:, 3] == 1.0)

    def test_partial_annotation_visibility(self, keypoint_json):
        """只标注部分关键点时, 未标注关键点可见性置 0, 坐标保持预填充"""
        _s, _f, frames = parse_keypoint_json(keypoint_json)
        tasks = frames_to_tasks(frames, mode="frames", source="video_001.mp4")
        task = tasks[0]
        first_pred = dict(task["predictions"][0]["result"][0]["value"])
        export_task = {
            "id": 0,
            "data": task["data"],
            "annotations": [{
                "result": [{
                    "from_name": "pose",
                    "to_name": "image",
                    "type": "keypointlabels",
                    "value": first_pred,
                }]
            }],
        }
        sample = convert_annotations([export_task])[0]
        kp = np.asarray(sample["keypoints"])
        assert kp[0, 3] == 1.0
        assert np.all(kp[1:, 3] == 0.0)
        original = np.asarray(frames[0]["keypoints"])
        assert np.allclose(kp[1:, :2], original[1:, :2])

    def test_extract_label_direct(self, keypoint_json):
        _s, _f, frames = parse_keypoint_json(keypoint_json)
        tasks = frames_to_tasks(frames, mode="frames")
        export_task = {
            "data": tasks[0]["data"],
            "annotations": [{
                "result": [{
                    "from_name": "fall_risk",
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": ["高危级"]},
                }]
            }],
        }
        assert extract_fall_risk_label(export_task) == 3

    def test_missing_label_is_none(self, keypoint_json):
        _s, _f, frames = parse_keypoint_json(keypoint_json)
        tasks = frames_to_tasks(frames, mode="frames", source="video_001.mp4")
        export_task = {"id": 0, "data": tasks[0]["data"], "annotations": [{"result": []}]}
        sample = convert_annotations([export_task])[0]
        assert sample["fall_risk_label"] is None

    def test_label_mapping(self):
        assert LABEL_TO_LEVEL["低风险"] == 0
        assert LABEL_TO_LEVEL["关注级"] == 1
        assert LABEL_TO_LEVEL["预警级"] == 2
        assert LABEL_TO_LEVEL["高危级"] == 3
        assert LABEL_TO_LEVEL["critical"] == 3
        assert LABEL_TO_LEVEL["2"] == 2

    def test_clip_expansion(self, keypoint_json):
        """clip 任务导出为逐帧样本, 共享片段风险标签"""
        _s, _f, frames = parse_keypoint_json(keypoint_json)
        tasks = frames_to_tasks(frames, mode="clips", clip_len=2, source="video_001.mp4")
        export_task = {
            "id": 0,
            "data": tasks[0]["data"],
            "annotations": [{
                "result": [{
                    "from_name": "fall_risk",
                    "to_name": "video",
                    "type": "choices",
                    "value": {"choices": ["预警级"]},
                }]
            }],
        }
        samples = convert_annotations([export_task])
        assert len(samples) == 2
        assert [s["frame_id"] for s in samples] == [0, 1]
        assert all(s["fall_risk_label"] == 2 for s in samples)
        assert np.asarray(samples[0]["keypoints"]).shape == (33, 4)

    def test_malformed_keypoints_raises(self):
        task = {"data": {"frame_id": 0, "keypoints": [[0.0] * 4] * 5}}
        with pytest.raises(ValueError):
            task_to_samples(task)

    def test_task_without_keypoints_yields_empty(self):
        task = {"data": {"frame_id": 0}}
        assert task_to_samples(task) == []

    def test_load_export_malformed(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_annotation_export(str(path))


# ============================================================
# 一致性: fall_risk_label
# ============================================================
class TestFallRiskAgreement:
    def test_known_kappa_value(self):
        pairs = [
            (_sample(0, 0), _sample(0, 0)),
            (_sample(1, 1), _sample(1, 1)),
            (_sample(2, 2), _sample(2, 3)),
        ]
        report = fall_risk_agreement(pairs)
        assert report["n"] == 3
        assert report["n_agree"] == 2
        assert report["raw_agreement"] == pytest.approx(2 / 3, abs=1e-4)
        # po=2/3, pe=2/9 => kappa=4/7
        assert report["kappa"] == pytest.approx(4 / 7, abs=1e-4)

    def test_none_labels_skipped(self):
        pairs = [
            (_sample(0, 0), _sample(0, 0)),
            (_sample(1, None), _sample(1, 1)),
        ]
        report = fall_risk_agreement(pairs)
        assert report["n"] == 1
        assert report["skipped"] == 1
        assert report["kappa"] == 1.0

    def test_empty_pairs(self):
        report = fall_risk_agreement([])
        assert report["n"] == 0
        assert report["kappa"] is None


# ============================================================
# 一致性: 可见性与对齐
# ============================================================
class TestVisibilityAgreement:
    def test_per_keypoint_agreement(self):
        pairs = [
            (_sample(0, 0, 0.9), _sample(0, 0, 0.9)),  # 双方都可见
            (_sample(1, 0, 0.1), _sample(1, 0, 0.1)),  # 双方都不可见
            (_sample(2, 0, 0.9), _sample(2, 0, 0.1)),  # 不一致
        ]
        report = visibility_agreement(pairs, threshold=0.5)
        assert len(report) == 33
        first = report[0]
        assert first["name"] == "nose"
        assert first["n"] == 3
        assert first["n_agree"] == 2
        assert first["agreement"] == pytest.approx(2 / 3, abs=1e-4)
        assert all(item["agreement"] == pytest.approx(2 / 3, abs=1e-4) for item in report)


class TestMatchByFrameId:
    def test_matching_and_unmatched(self):
        a = [_sample(0, 0), _sample(1, 1), _sample(5, 0)]
        b = [_sample(0, 0), _sample(1, 0), _sample(2, 2)]
        pairs, unmatched, duplicates = match_by_frame_id(a, b)
        assert len(pairs) == 2
        assert unmatched == [2, 5]
        assert duplicates == []

    def test_duplicate_frames_reported(self):
        # 同一 frame_id 在一份标注中出现多次: B 侧仅首次参与配对, 重复帧单独报告
        a = [_sample(0, 0), _sample(0, 1), _sample(1, 0)]
        b = [_sample(0, 0), _sample(1, 0), _sample(1, 1)]
        pairs, unmatched, duplicates = match_by_frame_id(a, b)
        assert len(pairs) == 3
        assert unmatched == []
        assert duplicates == [0, 1]


class TestLoadAnnotationFile:
    def test_malformed_sample_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"frame_id": 0}]), encoding="utf-8")
        with pytest.raises(ValueError):
            load_annotation_file(str(path))

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_annotation_file(str(path))


# ============================================================
# 一致性: 端到端报告
# ============================================================
class TestComputeAgreement:
    def test_end_to_end_perfect(self, tmp_path):
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        samples_a = [_sample(0, 0, 0.9), _sample(1, 1, 0.9)]
        samples_b = [_sample(0, 0, 0.9), _sample(1, 1, 0.9)]
        path_a.write_text(json.dumps(samples_a, ensure_ascii=False), encoding="utf-8")
        path_b.write_text(json.dumps(samples_b, ensure_ascii=False), encoding="utf-8")
        report = compute_agreement(str(path_a), str(path_b))
        assert report["n_a"] == 2
        assert report["n_b"] == 2
        assert report["n_matched"] == 2
        assert report["fall_risk"]["kappa"] == 1.0
        assert report["fall_risk"]["raw_agreement"] == 1.0
        assert all(item["agreement"] == 1.0 for item in report["visibility"])

    def test_end_to_end_partial(self, tmp_path):
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        samples_a = [_sample(0, 0, 0.9), _sample(1, 1, 0.9), _sample(2, 0, 0.9)]
        samples_b = [_sample(0, 0, 0.9), _sample(1, 2, 0.9)]
        path_a.write_text(json.dumps(samples_a, ensure_ascii=False), encoding="utf-8")
        path_b.write_text(json.dumps(samples_b, ensure_ascii=False), encoding="utf-8")
        report = compute_agreement(str(path_a), str(path_b))
        assert report["n_matched"] == 2
        assert report["unmatched_frames"] == [2]
        assert report["fall_risk"]["n_agree"] == 1


# ============================================================
# 导入: 关键点预填充对齐 (H1 回归)
# ============================================================
class TestKeypointMapFromJson:
    def test_keys_align_with_sample_interval(self, tmp_path):
        """JSON 第 i 帧应对齐视频第 i*sample_interval 帧, 而非按序 0,1,2,..."""
        path = tmp_path / "kps.json"
        frames = [
            {"timestamp": i * 0.1, "is_valid": True, "keypoints": _make_keypoint_frame(i)}
            for i in range(3)
        ]
        path.write_text(
            json.dumps({"source": "video_001.mp4", "fps": 10.0, "frames": frames}),
            encoding="utf-8",
        )
        kp_map = _keypoint_map_from_json(str(path), sample_interval=5)
        assert sorted(kp_map) == [0, 5, 10]

    def test_interval_one_keeps_order(self, tmp_path):
        """sample_interval=1 时退化为按序映射"""
        path = tmp_path / "kps.json"
        frames = [
            {"timestamp": i * 0.1, "is_valid": True, "keypoints": _make_keypoint_frame(i)}
            for i in range(2)
        ]
        path.write_text(
            json.dumps({"source": "video_001.mp4", "fps": 10.0, "frames": frames}),
            encoding="utf-8",
        )
        kp_map = _keypoint_map_from_json(str(path), sample_interval=1)
        assert sorted(kp_map) == [0, 1]


# ============================================================
# 导出: skipped 标注处理
# ============================================================
class TestSkippedAnnotations:
    def test_skipped_annotation_ignored(self):
        task = {
            "annotations": [
                {
                    "skipped": True,
                    "result": [
                        {
                            "from_name": "fall_risk",
                            "type": "choices",
                            "value": {"choices": ["高危级"]},
                        }
                    ],
                },
                {
                    "skipped": False,
                    "result": [
                        {
                            "from_name": "fall_risk",
                            "type": "choices",
                            "value": {"choices": ["关注级"]},
                        }
                    ],
                },
            ]
        }
        assert extract_fall_risk_label(task) == LABEL_TO_LEVEL["关注级"]

    def test_all_skipped_returns_none(self):
        task = {
            "annotations": [
                {
                    "skipped": True,
                    "result": [
                        {
                            "from_name": "fall_risk",
                            "type": "choices",
                            "value": {"choices": ["高危级"]},
                        }
                    ],
                }
            ]
        }
        assert extract_fall_risk_label(task) is None
