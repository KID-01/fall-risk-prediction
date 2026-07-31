"""
LabelStudio 标注任务导入脚本 — 将关键点 JSON / 视频文件转换为 LabelStudio 任务

将已提取的关键点序列 (KeypointFrame JSON) 或视频文件转换为 LabelStudio 可导入的
任务 JSON。每个任务包含图片/视频引用、预填充的 33 个 MediaPipe 关键点标注
(与 src/utils/keypoints.py 的 PoseKeypoint 枚举顺序一致), 以及跌倒风险等级选择
(低风险/关注级/预警级/高危级, 与 src/alerts/engine.py 的 RiskLevel 对齐)。

用法:
    python scripts/labelstudio_import.py --input data/keypoints/video_001.json --output data/labelstudio/tasks_video_001.json --mode frames --image-url-base http://localhost:8080/images
    python scripts/labelstudio_import.py --input data/keypoints/video_001.json --output data/labelstudio/tasks_clips.json --mode clips --clip-len 30
    python scripts/labelstudio_import.py --input data/videos --output data/labelstudio/tasks_video.json --mode video --image-dir data/labelstudio/images
    python scripts/labelstudio_import.py --emit-label-config configs/labelstudio_config.xml
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

# 跌倒风险等级选项 (与 src/alerts/engine.py RiskLevel 对齐)
FALL_RISK_CHOICES: list[str] = ["低风险", "关注级", "预警级", "高危级"]

# 33 个 MediaPipe 关键点标签 (与 PoseKeypoint 枚举顺序一致)
KEYPOINT_LABEL_NAMES: list[str] = [KEYPOINT_NAMES[i] for i in range(len(KEYPOINT_NAMES))]

DEFAULT_VISIBILITY_THRESHOLD = 0.5   # 关键点可见性阈值
DEFAULT_CLIP_LEN = 30                # clips 模式片段长度(帧数)
DEFAULT_SAMPLE_INTERVAL = 5          # video 模式帧采样间隔
DEFAULT_FPS = 10.0                   # 关键点序列默认帧率

# 关键点标注配色 (循环使用)
_KP_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#808080", "#ffffff", "#000000",
]


# ==== 标注配置 ====

def build_label_config(scope: str = "frames") -> str:
    """
    生成 LabelStudio 标注配置 XML

    Args:
        scope: "frames" 单帧关键点标注 (Image + KeyPointLabels + Choices)
               "clips"  片段级风险标注 (Video + Choices)
    Returns:
        LabelStudio 项目标注配置 XML 字符串
    """
    keypoint_labels = "\n".join(
        f'    <Label value="{name}" background="{_KP_COLORS[i % len(_KP_COLORS)]}"/>'
        for i, name in enumerate(KEYPOINT_LABEL_NAMES)
    )
    choice_labels = "\n".join(
        f'    <Choice value="{choice}"/>' for choice in FALL_RISK_CHOICES
    )
    if scope == "clips":
        return (
            "<View>\n"
            '  <Header value="跌倒风险片段标注"/>\n'
            '  <Video name="video" value="$video"/>\n'
            '  <Choices name="fall_risk" toName="video" choice="single" '
            'showInLine="true" required="true">\n'
            f"{choice_labels}\n"
            "  </Choices>\n"
            "</View>\n"
        )
    return (
        "<View>\n"
        '  <Header value="跌倒风险关键点标注"/>\n'
        '  <Image name="image" value="$image" zoom="true" zoomControl="true"/>\n'
        '  <KeyPointLabels name="pose" toName="image" strokeWidth="3" '
        'pointSize="small" opacity="0.9">\n'
        f"{keypoint_labels}\n"
        "  </KeyPointLabels>\n"
        '  <Choices name="fall_risk" toName="image" choice="single" '
        'showInLine="true" required="true">\n'
        f"{choice_labels}\n"
        "  </Choices>\n"
        "</View>\n"
    )


# ==== 数据解析 ====

def _load_json(data: dict | str | Path) -> dict:
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


def _coerce_keypoints(raw: list | np.ndarray) -> list[list[float]]:
    """校验关键点数组并转换为 (33, 4) 列表, 非法时抛出 ValueError"""
    arr = np.asarray(raw, dtype=float)
    if arr.shape != (33, 4):
        raise ValueError(f"关键点数组必须为 (33, 4), 实际形状: {arr.shape}")
    return arr.tolist()


def parse_keypoint_json(
    data: dict | str | Path,
    default_fps: float = DEFAULT_FPS,
) -> tuple[str, float, list[dict]]:
    """
    解析关键点 JSON 数据

    支持两种结构:
      1. 帧列表: {"source": "a.mp4", "fps": 10.0, "frames": [
             {"timestamp": 0.0, "is_valid": true, "keypoints": [[x, y, z, v] x33]}, ...]}
      2. 数组: {"source": "a.mp4", "fps": 10.0, "timestamps": [0.0, ...],
               "keypoints": [[[x, y, z, v] x33] xT]}

    Returns:
        (source, fps, frames) — frames 为 [{timestamp, keypoints}] 列表
    """
    obj = _load_json(data)
    source = str(obj.get("source", ""))
    fps = float(obj.get("fps", default_fps))

    if "frames" in obj:
        frames: list[dict] = []
        for i, frame in enumerate(obj["frames"]):
            if not isinstance(frame, dict) or "keypoints" not in frame:
                raise ValueError(f"frames[{i}] 缺少 keypoints 字段")
            frames.append({
                "timestamp": float(frame.get("timestamp", i / fps)),
                "keypoints": _coerce_keypoints(frame["keypoints"]),
            })
        return source, fps, frames

    if "keypoints" in obj:
        arr = np.asarray(obj["keypoints"], dtype=float)
        if arr.ndim != 3 or arr.shape[1:] != (33, 4):
            raise ValueError(f"关键点数组必须为 (T, 33, 4), 实际形状: {arr.shape}")
        timestamps = obj.get("timestamps", [])
        frames = [
            {
                "timestamp": float(timestamps[i]) if i < len(timestamps) else i / fps,
                "keypoints": arr[i].tolist(),
            }
            for i in range(arr.shape[0])
        ]
        return source, fps, frames

    raise ValueError("无法识别的关键点 JSON: 需要 frames 或 keypoints 字段")


# ==== 任务构建 ====

def keypoint_predictions(
    keypoints: list[list[float]],
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> list[dict]:
    """
    根据关键点数组生成 LabelStudio keypointlabels 预标注

    仅对可见性超过阈值的可见关键点生成预标注, 坐标按图片百分比 (0-100) 输出。
    """
    results: list[dict] = []
    for i, (x, y, _z, v) in enumerate(keypoints):
        if v > visibility_threshold:
            results.append({
                "from_name": "pose",
                "to_name": "image",
                "type": "keypointlabels",
                "value": {
                    "keypointlabels": [KEYPOINT_LABEL_NAMES[i]],
                    "x": round(float(x) * 100, 2),
                    "y": round(float(y) * 100, 2),
                    "width": 100.0,
                    "height": 100.0,
                },
            })
    return results


def frame_to_task(
    frame: dict,
    frame_id: int,
    *,
    image: str = "",
    source: str = "",
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> dict:
    """单帧关键点 → 单个 LabelStudio 任务 (frames 模式)"""
    task: dict = {
        "data": {
            "image": image,
            "source": source,
            "frame_id": frame_id,
            "timestamp": float(frame.get("timestamp", frame_id)),
        }
    }
    if "keypoints" in frame:
        keypoints = _coerce_keypoints(frame["keypoints"])
        task["data"]["keypoints"] = keypoints
        predictions = keypoint_predictions(keypoints, visibility_threshold)
        if predictions:
            task["predictions"] = [{"result": predictions}]
    return task


def clip_to_task(
    frames: list[dict],
    frame_ids: list[int],
    clip_id: int,
    *,
    video: str = "",
    source: str = "",
) -> dict:
    """关键点片段 → 单个 LabelStudio 任务 (clips 模式, 片段级风险标注)"""
    if not frames:
        raise ValueError("clip 帧列表为空")
    timestamps = [float(f.get("timestamp", i)) for i, f in enumerate(frames)]
    keypoints = [_coerce_keypoints(f["keypoints"]) for f in frames]
    return {
        "data": {
            "video": video,
            "source": source,
            "clip_id": clip_id,
            "frame_ids": frame_ids,
            "timestamps": timestamps,
            "keypoints": keypoints,
        }
    }


def _image_ref(
    frame_id: int,
    image_dir: str = "",
    image_url_base: str = "",
) -> str:
    """构造帧图片引用 (优先 URL, 其次本地路径, 都没有则返回空字符串)"""
    if image_url_base:
        return f"{image_url_base.rstrip('/')}/frame_{frame_id:06d}.jpg"
    if image_dir:
        return f"{image_dir.rstrip('/')}/frame_{frame_id:06d}.jpg"
    return ""


def _video_ref(
    clip_id: int,
    image_dir: str = "",
    image_url_base: str = "",
) -> str:
    """构造片段视频引用 (优先 URL, 其次本地路径, 都没有则返回空字符串)"""
    if image_url_base:
        return f"{image_url_base.rstrip('/')}/clip_{clip_id:06d}.mp4"
    if image_dir:
        return f"{image_dir.rstrip('/')}/clip_{clip_id:06d}.mp4"
    return ""


def frames_to_tasks(
    frames: list[dict],
    *,
    mode: str = "frames",
    clip_len: int = DEFAULT_CLIP_LEN,
    image_dir: str = "",
    image_url_base: str = "",
    source: str = "",
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> list[dict]:
    """
    关键点帧列表 → LabelStudio 任务列表

    Args:
        frames: parse_keypoint_json 返回的帧列表
        mode: "frames" 每帧一个任务 (预填关键点标注), "clips" 每片段一个任务
        clip_len: clips 模式片段长度 (帧数)
        image_dir / image_url_base: 帧图片的本地目录 / URL 前缀
        source: 数据来源标识 (如视频文件名)
        visibility_threshold: 预标注可见性阈值

    Returns:
        LabelStudio 任务列表 (JSON 数组可直接导入)
    """
    if not frames:
        return []
    tasks: list[dict] = []
    if mode == "frames":
        for frame_id, frame in enumerate(frames):
            image = _image_ref(frame_id, image_dir, image_url_base)
            tasks.append(frame_to_task(
                frame, frame_id,
                image=image,
                source=source,
                visibility_threshold=visibility_threshold,
            ))
        return tasks
    if mode == "clips":
        for clip_id, start in enumerate(range(0, len(frames), clip_len)):
            chunk = frames[start:start + clip_len]
            frame_ids = list(range(start, start + len(chunk)))
            video = _video_ref(clip_id, image_dir, image_url_base)
            tasks.append(clip_to_task(
                chunk, frame_ids, clip_id,
                video=video,
                source=source,
            ))
        return tasks
    raise ValueError(f"不支持的导入模式: {mode}")


# ==== 视频帧抽取 ====

def extract_video_frames(
    video_path: str,
    image_dir: str,
    sample_interval: int = DEFAULT_SAMPLE_INTERVAL,
) -> list[dict]:
    """
    从视频文件按间隔抽样帧并保存为 JPEG 图片

    懒加载 opencv (cv2), 仅 video 模式使用。
    Returns:
        [{"frame_id": int, "timestamp": float, "image": str}] 列表
    """
    import cv2

    out_dir = Path(image_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        frames: list[dict] = []
        frame_id = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            if frame_id % sample_interval == 0:
                image = str(out_dir / f"frame_{frame_id:06d}.jpg")
                cv2.imwrite(image, img)
                frames.append({
                    "frame_id": frame_id,
                    "timestamp": round(frame_id / fps, 3),
                    "image": image,
                })
            frame_id += 1
        return frames
    finally:
        cap.release()


def tasks_from_video(
    video_path: str,
    image_dir: str,
    image_url_base: str = "",
    sample_interval: int = DEFAULT_SAMPLE_INTERVAL,
    keypoint_map: dict[int, list[list[float]]] | None = None,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> list[dict]:
    """视频 → LabelStudio 任务列表 (每张抽样帧一个任务, 可选预填充关键点)"""
    source = Path(video_path).name
    keypoint_map = keypoint_map or {}
    tasks: list[dict] = []
    for frame in extract_video_frames(video_path, image_dir, sample_interval):
        frame_id = frame["frame_id"]
        image = frame["image"] if not image_url_base else _image_ref(frame_id, "", image_url_base)
        task_frame: dict = {"timestamp": frame["timestamp"]}
        if frame_id in keypoint_map:
            task_frame["keypoints"] = keypoint_map[frame_id]
        tasks.append(frame_to_task(
            task_frame, frame_id,
            image=image,
            source=source,
            visibility_threshold=visibility_threshold,
        ))
    return tasks


# ==== 主入口 ====

def _positive_int(value: str) -> int:
    """argparse 类型校验: 必须为正整数"""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"必须是整数: {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"必须是正整数: {value}")
    return parsed


def _keypoint_map_from_json(
    path: str, sample_interval: int = DEFAULT_SAMPLE_INTERVAL
) -> dict[int, list[list[float]]]:
    """从关键点 JSON 构建 {frame_id: keypoints} 映射 (video 模式预填充)

    约定: JSON 第 i 帧对应视频第 i * sample_interval 帧 (需与抽样间隔一致),
    因此映射键为 i * sample_interval, 与 extract_video_frames 产出的 frame_id 对齐。
    """
    _source, _fps, frames = parse_keypoint_json(path)
    return {i * sample_interval: f["keypoints"] for i, f in enumerate(frames)}


def main() -> int:
    """主入口: 解析命令行参数并执行转换"""
    parser = argparse.ArgumentParser(
        description="LabelStudio 标注任务导入脚本: 将关键点 JSON 或视频文件转换为标注任务"
    )
    parser.add_argument("--input", help="输入: 关键点 JSON 文件/目录, 或视频文件/目录 (--mode video)")
    parser.add_argument("--output", help="输出 LabelStudio 任务 JSON 文件路径")
    parser.add_argument(
        "--mode",
        choices=["frames", "clips", "video"],
        default="frames",
        help="导入模式: frames=每帧一个任务, clips=每短片段一个任务, video=从视频抽样生成帧任务",
    )
    parser.add_argument(
        "--clip-len",
        type=_positive_int,
        default=DEFAULT_CLIP_LEN,
        help=f"clips 模式片段长度 (帧数, 默认 {DEFAULT_CLIP_LEN})",
    )
    parser.add_argument("--image-dir", default="", help="帧图片目录: 作为图片引用路径或保存抽样帧")
    parser.add_argument("--image-url-base", default="", help="图片 URL 前缀 (LabelStudio 需可访问该地址)")
    parser.add_argument(
        "--sample-interval",
        type=_positive_int,
        default=DEFAULT_SAMPLE_INTERVAL,
        help=f"video 模式帧采样间隔 (默认每 {DEFAULT_SAMPLE_INTERVAL} 帧取 1 帧)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="关键点序列默认帧率, 仅当 JSON 未提供 fps 时生效 (默认 10.0)",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=DEFAULT_VISIBILITY_THRESHOLD,
        help="关键点可见性阈值 (默认 0.5)",
    )
    parser.add_argument(
        "--keypoints",
        default="",
        help="(video 模式可选) 关键点 JSON, 按帧序预填充到任务 "
        "(JSON 第 i 帧对应视频第 i×sample-interval 帧)",
    )
    parser.add_argument(
        "--emit-label-config",
        default="",
        help="将标注配置 XML (帧级) 写入该文件后退出",
    )
    args = parser.parse_args()

    setup_logging()

    if args.emit_label_config:
        output = Path(args.emit_label_config)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(build_label_config(scope="frames"), encoding="utf-8")
        log.info(f"标注配置已写入: {output}")
        return 0

    if not args.input or not args.output:
        parser.error("必须指定 --input 与 --output (或仅使用 --emit-label-config)")

    input_path = Path(args.input)
    tasks: list[dict] = []

    if args.mode == "video":
        video_paths = [input_path] if input_path.is_file() else sorted(
            input_path.glob("*.mp4")
        ) + sorted(input_path.glob("*.avi")) + sorted(input_path.glob("*.mov"))
        if not video_paths:
            log.error(f"未找到视频文件: {input_path}")
            return 1
        keypoint_map: dict[int, list[list[float]]] = {}
        if args.keypoints:
            keypoint_map = _keypoint_map_from_json(args.keypoints, args.sample_interval)
        for video_path in video_paths:
            video_tasks = tasks_from_video(
                str(video_path),
                args.image_dir,
                args.image_url_base,
                sample_interval=args.sample_interval,
                keypoint_map=keypoint_map,
                visibility_threshold=args.visibility_threshold,
            )
            tasks.extend(video_tasks)
            log.info(f"视频 {video_path.name}: 生成 {len(video_tasks)} 个任务")
    else:
        if input_path.is_dir():
            json_paths = sorted(input_path.glob("*.json"))
        elif input_path.is_file():
            json_paths = [input_path]
        else:
            log.error(f"输入不存在: {args.input}")
            return 1
        for json_path in json_paths:
            source, _fps, frames = parse_keypoint_json(str(json_path), default_fps=args.fps)
            mode_tasks = frames_to_tasks(
                frames,
                mode=args.mode,
                clip_len=args.clip_len,
                image_dir=args.image_dir,
                image_url_base=args.image_url_base,
                source=source or json_path.name,
                visibility_threshold=args.visibility_threshold,
            )
            tasks.extend(mode_tasks)
            log.info(f"文件 {json_path.name}: 生成 {len(mode_tasks)} 个任务")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"共生成 {len(tasks)} 个 LabelStudio 任务, 已写入: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
