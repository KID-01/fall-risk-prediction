# -*- coding: utf-8 -*-
"""Regression guard for Fall MVP v0.2 output/config compatibility."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from fall_mvp.contract import default_config  # noqa: E402


class TestV02Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            (REPO / "experiments/competition_sprint/manifests/fall_mvp_v02_contract.json").read_text(encoding="utf-8")
        )
        cls.artifact = REPO / "artifacts/fall/mvp_final"

    def test_frozen_frame_schema_has_all_v02_keys(self):
        with (self.artifact / "frames.jsonl").open(encoding="utf-8") as f:
            frame = json.loads(next(line for line in f if line.strip()))
        self.assertTrue(set(self.manifest["required_frame_top_level_keys"]).issubset(frame))
        for parent, keys in self.manifest["required_frame_nested_keys"].items():
            self.assertTrue(set(keys).issubset(frame[parent]), parent)

    def test_frozen_summary_has_all_v02_keys(self):
        summary = json.loads((self.artifact / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(set(self.manifest["required_summary_keys"]).issubset(summary))

    def test_default_config_keeps_v02_blocks_and_disables_extensions(self):
        cfg = default_config()
        self.assertTrue(set(self.manifest["required_config_top_level_keys"]).issubset(cfg))
        ext = cfg.get("risk_extensions", {"enabled": False})
        self.assertFalse(ext.get("enabled", False))

    def test_frozen_config_snapshot_does_not_require_extensions(self):
        cfg = yaml.safe_load((self.artifact / "config_snapshot.yaml").read_text(encoding="utf-8"))
        self.assertTrue(set(self.manifest["required_config_top_level_keys"]).issubset(cfg))
        self.assertNotIn("risk_extensions", cfg)


if __name__ == "__main__":
    unittest.main()
