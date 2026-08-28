# -*- coding: utf-8 -*-
"""Day-1 integration foundation 冒烟：visual 与 yolo 两分支独立可跑 + 机器可读输出。

RUN（用本地 yolo conda 解释器，含 ultralytics 8.4.96 + torch CPU）:
  E:\\anaconda\\conda_envs\\yolo\\python.exe fall_mvp/run_day1_smoke.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fall_mvp.visual_adapter import VisualPreimpactAdapter  # noqa: E402
from fall_mvp.yolo_detector import YoloEnvDetector  # noqa: E402

SAMPLE = Path(r"E:\ur_fall_rgb\fall-01")   # 正确 RGB 帧序列（工程样本）
OUT = REPO / "artifacts/fall/mvp_day1"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # 分支 B：预训练 YOLO COCO 环境检测
    det = YoloEnvDetector(REPO / "yolo26n.pt")
    yolo_rows = det.run(SAMPLE)
    (OUT / "yolo_frames.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in yolo_rows), encoding="utf-8")
    labels = {}
    for r in yolo_rows:
        for o in r["objects"]:
            labels[o["label"]] = labels.get(o["label"], 0) + 1
    print(f"[YOLO] frames={len(yolo_rows)}  detected_labels={labels}")

    # 分支 A：visual 预冲击适配（复用冻结 pose 模型，motion_risk_score 为工程启发）
    vis = VisualPreimpactAdapter(REPO / "weights/pose/yolo26n-pose.pt")
    vis_rows = vis.run(SAMPLE)
    (OUT / "visual_frames.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in vis_rows), encoding="utf-8")
    with_p = [r for r in vis_rows if r["persons"]]
    print(f"[VISUAL] frames={len(vis_rows)} with_person={len(with_p)} "
          f"max_motion_risk={max(r['motion_risk_score'] for r in vis_rows):.3f}")

    # Day-1 骨架：共享 timestamp 契约并存（E2E 最小骨架：两分支各自 JSONL）
    contract = {
        "day": 1, "gate": "visual_and_yolo_independent_machine_readable",
        "sample": str(SAMPLE),
        "outputs": {
            "visual": str(OUT / "visual_frames.jsonl"),
            "yolo": str(OUT / "yolo_frames.jsonl"),
        },
    }
    (OUT / "day1_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DAY1] contract:", json.dumps(contract, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())