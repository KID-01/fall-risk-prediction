# -*- coding: utf-8 -*-
"""Day-2 runner：Environment Risk Engine on real sample + debug overlay + sanity cases。

RUN（yolo conda env 含 ultralytics）:
  E:\\anaconda\\conda_envs\\yolo\\python.exe fall_mvp/run_day2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fall_mvp.env_risk import choose_primary, compute_env_risk, sanity_cases  # noqa: E402
from fall_mvp.yolo_detector import YoloEnvDetector  # noqa: E402
from fall_mvp.contract import default_config  # noqa: E402

SAMPLE = Path(r"E:\ur_fall_rgb\fall-01")
OUT = REPO / "artifacts/fall/mvp_day2"


def frames(source: Path):
    if source.is_dir():
        for i, fp in enumerate(sorted(source.rglob("*.png"))):
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


def draw_overlay(img, person, objs, f, out_path: Path):
    if person:
        cv2.rectangle(img, (int(person["x1"]), int(person["y1"])), (int(person["x2"]), int(person["y2"])), (0, 255, 0), 2)
        foot = (int((person["x1"] + person["x2"]) / 2), int(person["y2"]))
        cv2.circle(img, foot, 5, (0, 255, 255), -1)
        cv2.putText(img, "foot", (foot[0] - 20, foot[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for o in objs:
        cv2.rectangle(img, (int(o["x1"]), int(o["y1"])), (int(o["x2"]), int(o["y2"])), (255, 0, 0), 2)
        cv2.putText(img, f"{o['label']} {o.get('conf',0):.2f}", (int(o["x1"]), max(14, int(o["y1"]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if f.top_hazards:
        hz = f.top_hazards[0]
        cv2.putText(img, f"Environment Risk: {f.environment_risk_score}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        lines = [f"Environment Risk: {f.environment_risk_score}"]
        for h in f.top_hazards:
            lines.append(f"{h['class']}: d_norm={h['normalized_distance']} contrib={h['risk_contribution']}")
        y = 52
        for ln in lines[1:]:
            cv2.putText(img, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            y += 22
    cv2.imwrite(str(out_path), img)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = default_config()
    er = cfg["environment_risk"]

    # sanity cases（确定性几何，非科研）
    print("== sanity ==")
    sanity_cases(cfg)

    det = YoloEnvDetector(REPO / "yolo26n.pt")
    prev_primary = None
    rows = []
    representative = None
    for idx, ts_ms, frame in frames(SAMPLE):
        res = det.model.predict(source=frame, conf=er["object_conf_min"], imgsz=det.imgsz, verbose=False)[0]
        persons, objs = [], []
        if res.boxes is not None and len(res.boxes) > 0:
            xy = res.boxes.xyxy.cpu().numpy(); cf = res.boxes.conf.cpu().numpy()
            cl = res.boxes.cls.int().cpu().numpy()
            for i, b in enumerate(xy):
                x1, y1, x2, y2 = map(float, b)
                lab = det.names.get(int(cl[i]), "?")
                item = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(cf[i]), "track_id": i, "label": lab}
                (persons if lab == "person" else objs).append(item)
        primary = choose_primary(persons, prev_primary, er["person_conf_min"])
        prev_primary = primary
        f = compute_env_risk(primary, objs, cfg)
        f.timestamp = ts_ms / 1000.0
        rec = {
            "timestamp": f.timestamp, "source": f.source,
            "person": f.person, "objects": f.objects,
            "environment_risk_score": f.environment_risk_score,
            "top_hazards": f.top_hazards,
        }
        rows.append(rec)
        # 代表帧（有人 + 有 chair 等），用于可视化
        if representative is None and primary and any(o["label"] in ("chair", "suitcase", "couch", "backpack") for o in objs):
            representative = (idx, frame, primary, objs, f)

    (OUT / "env_frames.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    scores = [r["environment_risk_score"] for r in rows if r["person"]]
    print(f"[DAY2] frames={len(rows)} with_person={len(scores)} "
          f"env_risk min={min(scores):.3f} max={max(scores):.3f} mean={sum(scores)/len(scores):.3f}" if scores
          else "[DAY2] no person frames")
    hi = max(range(len(rows)), key=lambda i: rows[i]["environment_risk_score"]) if rows else -1
    print(f"[DAY2] representative(real) max-risk frame idx={hi}: {rows[hi] if hi>=0 else None}")

    if representative:
        idx, frame, primary, objs, f = representative
        vis_path = OUT / f"debug_frame_{idx:03d}.png"
        draw_overlay(frame.copy(), primary, objs, f, vis_path)
        print(f"[DAY2] overlay saved: {vis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())