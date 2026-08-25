# -*- coding: utf-8 -*-
"""Day-3 单一入口：python -m fall_mvp.run --input <video_or_frames> --output <dir>

端到端：visual(motion) + YOLO(env) -> env risk -> fusion(LOW/MED/HIGH/UNKNOWN) -> JSONL + demo mp4 + 摘要 + config 快照。
- 无训练/无调参；工程启发；非科研证据。
- 同步：同源帧（同一帧依次喂两分支）→ delta=0，超容差→UNKNOWN。
- 缺失语义：无人→UNKNOWN(person_missing)；motion 缺失但 YOLO 有人→motion=UNKNOWN（环境仍报）；无相关物→env LOW（"nothing hazardous"）
- person 一致性：visual 人框 vs YOLO 人框 IoU/中心。
- 稳定化：3 帧多数（因果，仅过去+当前）。
运行时：
  E:\\anaconda\\conda_envs\\yolo\\python.exe -m fall_mvp.run --input E:\\ur_fall_rgb\\fall-01 --output E:\\yolo26\\ultralytics-main\\artifacts\\fall\\mvp_day3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fall_mvp.contract import default_config  # noqa: E402
from fall_mvp.env_risk import choose_primary, compute_env_risk  # noqa: E402
from fall_mvp.fusion import fuse, causal_persist, FusionResult  # noqa: E402


def _iou(a, b) -> float:
    x1 = max(a["x1"], b["x1"]); y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"]); y2 = min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ar = max(1e-6, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    br = max(1e-6, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    return inter / (ar + br - inter)


def person_match_pair(vis_person, yolo_person, min_iou: float, center_ratio: float) -> bool:
    if vis_person is None or yolo_person is None:
        return False
    i = _iou(vis_person, yolo_person)
    cx1 = (vis_person["x1"] + vis_person["x2"]) / 2
    cy1 = (vis_person["y1"] + vis_person["y2"]) / 2
    cx2 = (yolo_person["x1"] + yolo_person["x2"]) / 2
    cy2 = (yolo_person["y1"] + yolo_person["y2"]) / 2
    h = max(vis_person["y2"] - vis_person["y1"], 1e-3)
    dc = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    return i >= min_iou or dc <= center_ratio * h


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


def _parse_boxes(res, names, obj_conf_min, track_seed=0):
    persons, objs = [], []
    if res.boxes is not None and len(res.boxes) > 0:
        xy = res.boxes.xyxy.cpu().numpy()
        cf = res.boxes.conf.cpu().numpy()
        cl = res.boxes.cls.int().cpu().numpy()
        for i, b in enumerate(xy):
            x1, y1, x2, y2 = map(float, b)
            lab = names.get(int(cl[i]), "?")
            item = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(cf[i]),
                    "track_id": track_seed + i, "label": lab}
            (persons if lab == "person" else objs).append(item)
    return persons, objs


def _draw(img, vis_p, env_p, objs, mscore, mstate, escore, estate, overall, top_hazards, reasons, t):
    if vis_p is not None:
        cv2.rectangle(img, (int(vis_p["x1"]), int(vis_p["y1"])), (int(vis_p["x2"]), int(vis_p["y2"])), (0, 255, 0), 2)
        foot = (int((vis_p["x1"] + vis_p["x2"]) / 2), int(vis_p["y2"]))
        cv2.circle(img, foot, 5, (0, 255, 255), -1)
    for o in objs:
        cv2.rectangle(img, (int(o["x1"]), int(o["y1"])), (int(o["x2"]), int(o["y2"])), (255, 0, 0), 2)
        cv2.putText(img, f"{o['label']} {o.get('conf',0):.2f}", (int(o["x1"]), max(14, int(o["y1"]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    col = {"LOW": (0, 255, 0), "MEDIUM": (0, 200, 255), "HIGH": (0, 0, 255), "UNKNOWN": (200, 200, 200)}.get(overall, (255, 255, 255))
    lines = [
        f"t={t:.2f}s  Motion={mscore if mscore is None else round(mscore,2)}/{mstate}",
        f"Env={escore if escore is None else round(escore,2)}/{estate}",
        f"OVERALL={overall}  ({'+'.join(reasons) if reasons else 'none'})",
    ]
    for hz in top_hazards[:1]:
        lines.append(f"  top hazard: {hz['class']} d={hz['normalized_distance']} c={hz['risk_contribution']}")
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    cfg = default_config()
    v = cfg["visual"]; y = cfg["yolo"]; er = cfg["environment_risk"]; fs = cfg["fusion"]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pose_model = YOLO(str(REPO / v["model"]))
    env_model = YOLO(str(REPO / y["model"]))

    # config 快照
    (out_dir / "config_snapshot.yaml").write_text(
        yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    rows = []
    states_hist: list[str] = []
    out_mp4 = out_dir / "demo.mp4"
    writer = None
    summary = {"frames": 0, "valid_person_frames": 0,
               "low_frames": 0, "medium_frames": 0, "high_frames": 0, "unknown_frames": 0,
               "max_environment_risk": 0.0, "top_detected_hazards": {}}
    prev_vis_primary = None
    prev_motion_state = None  # (ts_ms, cy, bbox_h)

    for idx, ts_ms, frame in _iter_frames(Path(args.input)):
        if args.max_frames and idx >= args.max_frames:
            break
        t = ts_ms / 1000.0
        # visual（motion）分支
        vp = pose_model.predict(source=frame, conf=v["conf"], imgsz=v["imgsz"], verbose=False)[0]
        vis_persons, _ = _parse_boxes(vp, pose_model.names, er["person_conf_min"])
        vis_primary = choose_primary(vis_persons, prev_vis_primary, er["person_conf_min"])
        prev_vis_primary = vis_primary
        # motion_heuristic_v0：bbox 体心垂直速度（归一化于 bbox 高，向下为正）；工程启发，非校准
        mscore = None
        if vis_primary is not None:
            cy = (vis_primary["y1"] + vis_primary["y2"]) / 2
            bh = vis_primary["y2"] - vis_primary["y1"]
            if prev_motion_state is not None and ts_ms - prev_motion_state[0] > 1:
                vy = (cy - prev_motion_state[1]) / ((ts_ms - prev_motion_state[0]) / 1000.0) / max(prev_motion_state[2], 1e-3)
                mscore = min(1.0, max(0.0, vy / 5.0))
            else:
                mscore = 0.0
            prev_motion_state = (ts_ms, cy, bh)
        else:
            prev_motion_state = None

        # env（YOLO）分支
        ep = env_model.predict(source=frame, conf=er["object_conf_min"], imgsz=y["imgsz"], verbose=False)[0]
        yolo_persons, objs = _parse_boxes(ep, env_model.names, er["object_conf_min"], track_seed=100)
        env_person = choose_primary(yolo_persons, None, er["person_conf_min"])
        envf = compute_env_risk(env_person, objs, cfg)
        envf.timestamp = t
        escore = envf.environment_risk_score

        # person 一致性
        person_match = None
        if vis_primary is not None and env_person is not None:
            person_match = person_match_pair(vis_primary, env_person,
                                             fs["person_match_min_iou"], fs["person_match_center_ratio"])
        elif vis_primary is None and env_person is None:
            person_match = None  # 无人
        else:
            person_match = False

        person_present = vis_primary is not None or env_person is not None

        # 融合
        fr = fuse(mscore, escore if person_present else None,
                  envf.top_hazards, person_present, person_match,
                  tuple(fs["motion_thr"]), tuple(fs["env_thr"]))
        states_hist.append(fr.overall_state)
        overall = causal_persist(states_hist, fs["persistence_frames"])

        row = {
            "timestamp": round(t, 3),
            "person": {"present": person_present, "match": person_match},
            "motion": {"source": v["source_tag"], "score": mscore, "state": fr.motion_state},
            "environment": {"source": "env_risk_v0", "score": escore if person_present else None,
                            "state": fr.environment_state, "top_hazards": envf.top_hazards},
            "fusion": {"source": "risk_fusion_v0", "overall_state": overall,
                       "raw_state": fr.overall_state, "reason": fr.reason_codes,
                       "context_elevated": fr.context_elevated},
            "quality": {"sync_delta_sec": 0.0, "status": "OK"},
        }
        rows.append(row)
        summary["frames"] += 1
        if person_present:
            summary["valid_person_frames"] += 1
        summary[f"{overall.lower()}_frames"] += 1
        summary["max_environment_risk"] = max(summary["max_environment_risk"], escore or 0.0)
        for hz in envf.top_hazards:
            summary["top_detected_hazards"][hz["class"]] = summary["top_detected_hazards"].get(hz["class"], 0) + 1

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
        annotated = _draw(frame.copy(), vis_primary, env_person, objs,
                          mscore, fr.motion_state, escore if person_present else None,
                          fr.environment_state, overall, envf.top_hazards, fr.reason_codes, t)
        writer.write(annotated)
    if writer is not None:
        writer.release()

    (out_dir / "frames.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DAY3] summary:", json.dumps(summary, ensure_ascii=False))
    print(f"[DAY3] outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())