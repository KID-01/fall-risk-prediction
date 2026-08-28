# -*- coding: utf-8 -*-
"""环境 YOLO 检测器适配器（Day-1 模块 B）。

用预训练 Ultralytics/YOLO COCO 模型（最小、已支持、无训练改动）。
输出每帧 JSON：timestamp, source, objects(label,conf,bbox), persons。
边界：非跌倒预测；类别 = 实际模型 COCO 清单；mAP 不作目标。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]


def _iter_frames(source: Path):
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


class YoloEnvDetector:
    def __init__(self, model_path: Path, conf: float = 0.25, imgsz: int = 640):
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.imgsz = imgsz

    @property
    def names(self) -> dict[int, str]:
        return self.model.names

    def run(self, source: Path) -> list[dict]:
        rows = []
        for idx, ts_ms, frame in _iter_frames(source):
            res = self.model.predict(source=frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            objs, persons = [], []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.int().cpu().numpy()
                for i, b in enumerate(xyxy):
                    label = self.names.get(int(cls[i]), "?")
                    x1, y1, x2, y2 = map(float, b)
                    item = {"label": label, "conf": float(confs[i]), "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    (persons if label == "person" else objs).append(item)
            rows.append({"timestamp": ts_ms / 1000.0, "source": "yolo_enviro_coco",
                         "objects": objs, "persons": persons})
        return rows


if __name__ == "__main__":  # 冒烟（Day-1）
    src = Path(r"E:\ur_fall_rgb\fall-01")
    out = REPO / "artifacts/fall/mvp_day1/yolo_frames.jsonl"
    det = YoloEnvDetector(REPO / "yolo26n.pt")
    rows = det.run(src)
    import json as _json
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows[:30]:
            f.write(_json.dumps(r, ensure_ascii=False) + "\n")
    labels = {}
    for r in rows:
        for o in r["objects"]:
            labels[o["label"]] = labels.get(o["label"], 0) + 1
    print(f"yolo smoke: frames={len(rows)}  objects_by_label={labels}")
    print("sample:", _json.dumps(rows[1] if rows else {}, ensure_ascii=False)[:220])