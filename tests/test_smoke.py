"""
冒烟测试 — 验证配置加载、关键模块导入、模型实例化
"""
from __future__ import annotations

import torch

from src.utils.config import get_config


class TestConfig:
    """配置加载与完整性"""

    def test_get_config(self):
        config = get_config()
        assert config is not None
        # 基础字段
        assert config.project.name == "fall-risk-prediction"
        assert config.project.version == "0.1.0"
        # 新增的 model / training 字段
        assert config.model.d_model == 256
        assert config.model.nhead == 8
        assert config.training.weight_decay == 0.01
        assert config.training.patience == 10
        # 关键路径
        assert config.paths.checkpoints == "checkpoints"


class TestModel:
    """模型实例化冒烟"""

    def test_fall_risk_predictor_instantiation(self):
        from src.models.fall_risk_predictor import FallRiskPredictor

        model = FallRiskPredictor()
        assert model is not None

        batch_size, seq_len = 2, 30
        keypoints = torch.randn(batch_size, seq_len, 33, 3)
        gait_feat = torch.randn(batch_size, 4)
        env_feat = torch.randn(batch_size, 5)

        output = model(keypoints, gait_feat, env_feat)
        assert "risk_score" in output
        assert "risk_probs" in output
        assert output["risk_score"].shape == (batch_size, 1)
        assert output["risk_probs"].shape == (batch_size, 4)

    def test_compute_loss(self):
        from src.models.fall_risk_predictor import FallRiskPredictor

        model = FallRiskPredictor()
        batch_size = 2
        keypoints = torch.randn(batch_size, 30, 33, 3)
        gait_feat = torch.randn(batch_size, 4)
        env_feat = torch.randn(batch_size, 5)
        preds = model(keypoints, gait_feat, env_feat)
        loss_dict = model.compute_loss(
            preds,
            risk_score_target=torch.rand(batch_size, 1),
            risk_level_target=torch.randint(0, 4, (batch_size,)),
        )
        assert "loss" in loss_dict
        assert loss_dict["loss"].item() > 0


class TestAPI:
    """API 模块导入冒烟"""

    def test_main_app_import(self):
        from src.api.main import app

        assert app.title is not None
        assert app.version is not None


class TestInference:
    """推理模块导入冒烟"""

    def test_feature_module(self):
        from src.inference.features import FeatureCalculator

        assert callable(FeatureCalculator)
