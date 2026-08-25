# -*- coding: utf-8 -*-
"""visual 预测器适配器（Day-1 模块 A）。

复用冻结 visual 系统（weights/pose/yolo26n-pose.pt + YOLO-Pose）作为工程集成。
输出每帧 JSON：timestamp, source, persons(含 track/bbox/关键点存在), motion_risk_score(工程启发)。
边界：非新模型；不重训/不调参；motion_risk_score 为工程启发（bbox 体心垂直速度启发），
不是校准概率，名称不用 fall_probability。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .contract import ObjectDet  # noqa: F401  (统一引入，供调用方)

REPO = Path(__file__).resolve().parents[1]

KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _iter_frames(source: Path):
    """目录(正确 RGB PNG 序列) 或 mp4；产出 (i, ts_ms, BGR)。ts 用 i*1000/fps 估算（工程）。"""
    if source.is_dir():
        files = sorted(source.rglob("*.png"))
        for i, fp in enumerate(files):
            img = cv2.imread(str(fp))
            if img is not None:
                yield i, i * (1000.0 / 30.0), img
    else:
        cap = cv2.VideoCapture(str(source))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield i, i * (1000.0 / fps), fr
            i += 1
        cap.release()


class VisualPreimpactAdapter:
    """冻结视觉预测器的工程适配：逐帧 person+pose+工程 motion 启发。"""

    def __init__(self, model_path: Path, conf: float = 0.25, imgsz: int = 640, motion_window_s: float = 0.3):
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.imgsz = imgsz
        self.motion_window_s = motion_window_s
        self._prev_centers: dict[int, tuple[float, float, float]] = {}  # track -> (ts_ms, cy, bbox_h)

    def _motion_risk(self, track_id: int, ts_ms: float, cy: float, bbox_h: float) -> float:
        """工程启发：体心垂直速度（归一化于 bbox 高）的下降速率 → 正=下坠。非校准。"""
        prev = self._prev_centers.get(track_id)
        risk = 0.0
        if prev is not None and ts_ms - prev[0] > 1:
            dt = (ts_ms - prev[0]) / 1000.0
            vy = (cy - prev[1]) / dt / max(prev[2], 1e-3)  # 向下为正（图像坐标）
            risk = max(0.0, min(1.0, vy / 5.0))            # 启发式饱和，任意阈值
        self._prev_centers[track_id] = (ts_ms, cy, bbox_h)
        return risk

    def run(self, source: Path) -> list[dict]:
        rows = []
        for idx, ts_ms, frame in _iter_frames(source):
            res = self.model.predict(source=frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            boxes = res.boxes
            kpts = res.keypoints
            persons = []
            risk = 0.0
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                ids = boxes.id.int().cpu().numpy() if boxes.id is not None else np.arange(len(xyxy))
                kd = kpts.data.cpu().numpy() if kpts is not None and len(kpts) > 0 else None
                for i, b in enumerate(xyxy):
                    tid = int(ids[i])
                    x1, y1, x2, y2 = map(float, b)
                    cy = (y1 + y2) / 2
                    bh = y2 - y1
                    ok_k = kd is not None and i < len(kd)
                    persons.append({
                        "track_id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": float(confs[i]), "keypoints_present": bool(ok_k),
                        "bbox_center_y": cy, "bbox_h": bh,
                    })
                    risk = max(risk, self._motion_risk(tid, ts_ms, cy, bh))
            rows.append({
                "timestamp": ts_ms / 1000.0,
                "source": "motion_heuristic_v0",   # 工程启发，非冻结 v1.0 科研 predictor
                "motion_heuristic_score": round(risk, 4),
                "alarm": bool(risk >= 0.5),        # 工程启发阈值（配置化后移）
                "persons": persons,
            })
        return rows


if __name__ == "__main__":  # 冒烟（Day-1）
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"E:\ur_fall_rgb\fall-01")
    out = REPO / "artifacts/fall/mvp_day1/visual_frames.jsonl"
    adapter = VisualPreimpactAdapter(REPO / "weights/pose/yolo26n-pose.pt")
    rows = adapter.run(src)
    import json as _json
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows[:30]:
            f.write(_json.dumps(r, ensure_ascii=False) + "\n")
    framed = [r for r in rows if r["persons"]]
    print(f"visual smoke: frames={len(rows)} with_person={len(framed)} "
          f"max_motion={max(r['motion_heuristic_score'] for r in rows):.3f}")
    print("sample:", _json.dumps(rows[1] if rows else {}, ensure_ascii=False)[:220])