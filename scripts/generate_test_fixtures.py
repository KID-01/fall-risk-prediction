"""Generate reproducible test data and clearly labelled model fixtures.

This script is for plumbing/integration tests only. The generated project
checkpoint is randomly initialized and must not be used for accuracy claims.
YOLO files, when requested, are downloaded from Ultralytics as official
pretrained models; they are not trained for this project.

Examples (PowerShell):
    .\\.venv\\Scripts\\python.exe scripts\\generate_test_fixtures.py
    .\\.venv\\Scripts\\python.exe scripts\\generate_test_fixtures.py --download-yolo
    .\\.venv\\Scripts\\python.exe scripts\\generate_test_fixtures.py --skip-model
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import struct
from pathlib import Path

import numpy as np


def _write_npy(path: Path, array: np.ndarray) -> None:
    """Write a NumPy v1.0 file without relying on np.save implementation details."""
    array = np.asarray(array, dtype=np.float32, order="C")
    header = repr(
        {"descr": "<f4", "fortran_order": False, "shape": tuple(array.shape)}
    ).replace("'", "'")
    header_bytes = (header + " " * (64 - ((10 + len(header) + 1) % 64)) + "\n").encode(
        "latin1"
    )
    if len(header_bytes) >= 65536:
        raise ValueError("NPY header is too large")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"\x93NUMPY\x01\x00")
        handle.write(struct.pack("<H", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(array.tobytes(order="C"))


def _synthetic_sequence(seq_len: int, sample_index: int, rng: np.random.Generator) -> np.ndarray:
    """Create a plausible normalized (T, 33, 4) keypoint sequence."""
    # Approximate MediaPipe body layout. Unmapped facial points remain near the head.
    anchors = np.zeros((33, 2), dtype=np.float32)
    anchors[0] = (0.50, 0.18)
    anchors[11:17] = [
        (0.43, 0.30), (0.57, 0.30), (0.39, 0.42),
        (0.61, 0.42), (0.35, 0.54), (0.65, 0.54),
    ]
    anchors[23:29] = [
        (0.45, 0.55), (0.55, 0.55), (0.43, 0.72),
        (0.57, 0.72), (0.40, 0.90), (0.60, 0.90),
    ]
    anchors[29:33] = [(0.39, 0.93), (0.61, 0.93), (0.37, 0.95), (0.63, 0.95)]
    anchors[1:11] = anchors[0]
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    phase = np.float32(sample_index * 0.37)
    sway = 0.012 * np.sin(2 * np.pi * (1.5 * t + phase))
    step = 0.018 * np.sin(2 * np.pi * (2.0 * t + phase))
    sequence = np.repeat(anchors[None, :, :], seq_len, axis=0)
    sequence[:, :, 0] += sway[:, None]
    sequence[:, 27, 0] += step
    sequence[:, 28, 0] -= step
    sequence += rng.normal(0.0, 0.0025, size=sequence.shape).astype(np.float32)
    sequence = np.clip(sequence, 0.0, 1.0)
    confidence = np.full((seq_len, 33, 1), 0.95, dtype=np.float32)
    return np.concatenate([sequence, confidence], axis=-1)


def generate_keypoints(output_dir: Path, labels_path: Path, samples: int, seq_len: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    labels: dict[str, dict[str, float | int | str]] = {}
    actions = [("standing", 8.0, 0), ("walking", 22.0, 1), ("sitting", 38.0, 2), ("fall_like", 86.0, 3)]
    for index in range(samples):
        name = f"fixture_{index:03d}"
        path = output_dir / f"{name}.npy"
        _write_npy(path, _synthetic_sequence(seq_len, index, rng))
        action, score, level = actions[index % len(actions)]
        labels[name] = {"risk_score": score, "risk_level": level, "action": action}
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_project_checkpoint(checkpoint_path: Path, seed: int) -> bool:
    try:
        import torch
        import sys

        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.models.fall_risk_predictor import FallRiskPredictor
    except ImportError as exc:
        print(f"Skip project checkpoint: missing dependency ({exc})")
        return False

    torch.manual_seed(seed)
    model = FallRiskPredictor()
    checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "best_mae": None,
        "metadata": {
            "checkpoint_type": "synthetic_random_init",
            "production_ready": False,
            "purpose": "load and forward-pass integration testing only",
            "seed": seed,
        },
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Wrote project test checkpoint: {checkpoint_path}")
    return True


def download_yolo_models(checkpoint_dir: Path) -> None:
    from ultralytics import YOLO

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for model_name in ("yolov8n.pt", "yolov8n-pose.pt"):
        target = checkpoint_dir / model_name
        if target.exists():
            print(f"YOLO file already exists: {target}")
            continue
        model = YOLO(model_name)
        source = Path(getattr(model, "ckpt_path", model_name))
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        print(f"Downloaded official pretrained YOLO model: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--skip-model", action="store_true", help="Do not create test_best.pt")
    parser.add_argument("--download-yolo", action="store_true", help="Download official YOLO weights")
    args = parser.parse_args()
    if args.samples < 1 or args.seq_len < 1:
        parser.error("--samples and --seq-len must be positive")

    random.seed(args.seed)
    root = args.output_root.resolve()
    generate_keypoints(
        root / "data" / "keypoints",
        root / "data" / "labels" / "fixture_labels.json",
        args.samples,
        args.seq_len,
        args.seed,
    )
    print(f"Wrote {args.samples} keypoint fixtures under {root / 'data' / 'keypoints'}")
    if not args.skip_model:
        generate_project_checkpoint(root / "checkpoints" / "test_best.pt", args.seed)
    if args.download_yolo:
        download_yolo_models(root / "checkpoints")
    print("All generated artifacts are for integration tests, not accuracy evaluation.")


if __name__ == "__main__":
    main()
