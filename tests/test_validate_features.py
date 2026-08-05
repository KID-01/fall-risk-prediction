"""
步态特征有效性验证脚本的单元测试

测试合成数据生成 / t-SNE / SHAP / 相关性 / 文件保存。
所有测试均不依赖 KeypointFrame 或视频数据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 将项目根目录添加到 sys.path 以导入 scripts 包
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import scripts.validate_features  # noqa: E402
from scripts.validate_features import (  # noqa: E402  -- sys.path 调整后导入 scripts/
    FEATURE_NAMES,
    _feature_params,
    correlation_analysis,
    generate_synthetic_features,
    shap_analysis,
    tsne_analysis,
)


@pytest.fixture(autouse=True)
def _isolate_figure_output(tmp_path, monkeypatch):
    """将图保存重定向到临时目录, 避免覆盖/删除 docs/figures 下的提交产物"""
    monkeypatch.setattr("scripts.validate_features.OUTPUT_DIR", tmp_path)


# ============================================================
# 合成数据生成
# ============================================================
class TestGenerateSyntheticFeatures:
    def test_output_shapes(self):
        """验证返回的 x 和 y 维度正确."""
        x, y = generate_synthetic_features(n_per_level=50, seed=42)
        assert x.shape == (200, 4), f"Expected (200, 4), got {x.shape}"
        assert y.shape == (200,), f"Expected (200,), got {y.shape}"

    def test_risk_level_distribution(self):
        """验证每个风险等级样本数相等."""
        x, y = generate_synthetic_features(n_per_level=60, seed=42)
        unique, counts = np.unique(y, return_counts=True)
        assert len(unique) == 4
        assert np.all(counts == 60), f"Expected all 60, got {counts}"

    def test_feature_value_ranges(self):
        """验证特征值在合理范围内."""
        x, y = generate_synthetic_features(n_per_level=100, seed=42)
        # activity_density 应在 [0, 1] 内
        assert np.all(x[:, 3] >= 0.0), "activity_density 有负值"
        assert np.all(x[:, 3] <= 1.0), "activity_density 超过 1.0"
        # trunk_stability 不应为负
        assert np.all(x[:, 2] >= 0.0), "trunk_stability 有负值"

    def test_monotonic_feature_trend(self):
        """验证特征均值随风险等级单调变化."""
        x, y = generate_synthetic_features(n_per_level=150, seed=42)
        # walking_rhythm 应随风险等级降低
        means = [np.mean(x[y == level, 0]) for level in range(4)]
        for i in range(3):
            assert means[i] > means[i + 1], (
                f"walking_rhythm 均值应递减: {means}"
            )
        # trunk_stability 应随风险等级升高
        means_stab = [np.mean(x[y == level, 2]) for level in range(4)]
        for i in range(3):
            assert means_stab[i] < means_stab[i + 1], (
                f"trunk_stability 均值应递增: {means_stab}"
            )


# ============================================================
# _feature_params
# ============================================================
class TestFeatureParams:
    def test_returns_valid_params(self):
        """验证每个风险等级都有配置."""
        for level in range(4):
            params = _feature_params(level)
            assert "walking_rhythm" in params
            assert "step_amplitude" in params
            assert "trunk_stability" in params
            assert "activity_density" in params
            # 每个配置包含 (mean, std)
            for name in FEATURE_NAMES:
                mean, std = params[name]
                assert isinstance(mean, float)
                assert isinstance(std, float)
                assert std > 0

    def test_monotonic_params(self):
        """验证配置参数随风险等级单调变化."""
        means_rhythm = [_feature_params(level)["walking_rhythm"][0] for level in range(4)]
        means_trunk = [_feature_params(level)["trunk_stability"][0] for level in range(4)]
        # walking_rhythm 应递减
        for i in range(3):
            assert means_rhythm[i] > means_rhythm[i + 1]
        # trunk_stability 应递增
        for i in range(3):
            assert means_trunk[i] < means_trunk[i + 1]


# ============================================================
# t-SNE 分析
# ============================================================
class TestTSNEAnalysis:
    def test_output_shape(self):
        """验证 t-SNE 返回 (N, 2) 矩阵."""
        x, y = generate_synthetic_features(n_per_level=50, seed=42)
        embedded = tsne_analysis(x, y)
        assert embedded.shape == (200, 2), f"Expected (200, 2), got {embedded.shape}"

    def test_returns_float_array(self):
        """验证 t-SNE 返回浮点数组."""
        x, y = generate_synthetic_features(n_per_level=50, seed=43)
        embedded = tsne_analysis(x, y)
        assert embedded.dtype.kind == "f", f"Expected float dtype, got {embedded.dtype}"


# ============================================================
# SHAP 分析
# ============================================================
class TestSHAPAnalysis:
    def test_shap_values_shape(self):
        """验证 SHAP 值形状正确."""
        x, y = generate_synthetic_features(n_per_level=50, seed=44)
        result = shap_analysis(x, y)
        if result["shap_values"] is not None:
            shap_vals = result["shap_values"]
            # shap 0.52+ 返回 3D 数组 (n_samples, n_features, n_classes)
            if isinstance(shap_vals, list):
                assert len(shap_vals) == 4
                for sv in shap_vals:
                    assert sv.shape == (200, 4), f"Expected (200, 4), got {sv.shape}"
            elif shap_vals.ndim == 3:
                n_samples, n_features, n_classes = shap_vals.shape
                assert n_samples == 200
                assert n_features == 4
                assert n_classes == 4
            else:
                assert shap_vals.shape[1] == 4

    def test_feature_importance_keys(self):
        """验证特征重要性字典包含所有特征名."""
        x, y = generate_synthetic_features(n_per_level=50, seed=45)
        result = shap_analysis(x, y)
        importance = result["feature_importance"]
        for name in FEATURE_NAMES:
            assert name in importance, f"缺少特征: {name}"

    def test_ranking_length(self):
        """验证特征排名包含全部 4 个特征."""
        x, y = generate_synthetic_features(n_per_level=50, seed=46)
        result = shap_analysis(x, y)
        assert len(result["ranking"]) == len(FEATURE_NAMES)


# ============================================================
# 相关性分析
# ============================================================
class TestCorrelationAnalysis:
    def test_pearson_shape(self):
        """验证 Pearson 系数形状."""
        x, y = generate_synthetic_features(n_per_level=100, seed=42)
        result = correlation_analysis(x, y)
        pearson_r, pearson_p = result["pearson"]
        assert pearson_r.shape == (4,), f"Expected (4,), got {pearson_r.shape}"
        assert pearson_p.shape == (4,), f"Expected (4,), got {pearson_p.shape}"

    def test_spearman_shape(self):
        """验证 Spearman 系数形状."""
        x, y = generate_synthetic_features(n_per_level=100, seed=43)
        result = correlation_analysis(x, y)
        spearman_r, spearman_p = result["spearman"]
        assert spearman_r.shape == (4,), f"Expected (4,), got {spearman_r.shape}"
        assert spearman_p.shape == (4,), f"Expected (4,), got {spearman_p.shape}"

    def test_significant_features_not_empty(self):
        """验证至少有一个显著相关特征."""
        x, y = generate_synthetic_features(n_per_level=100, seed=44)
        result = correlation_analysis(x, y)
        assert len(result["significant"]) > 0, "应至少有一个显著相关特征"

    def test_pearson_direction(self):
        """验证相关系数方向."""
        x, y = generate_synthetic_features(n_per_level=150, seed=42)
        result = correlation_analysis(x, y)
        pearson_r, pearson_p = result["pearson"]
        # walking_rhythm 与风险等级应负相关
        assert pearson_r[0] < 0, (
            f"walking_rhythm 应与风险负相关, 实际 r={pearson_r[0]:.4f}"
        )
        # trunk_stability 与风险等级应正相关
        assert pearson_r[2] > 0, (
            f"trunk_stability 应与风险正相关, 实际 r={pearson_r[2]:.4f}"
        )
        # 所有 p 值应显著
        assert np.all(pearson_p < 0.05), "所有特征 Pearson 应显著"


# ============================================================
# 文件保存
# ============================================================
class TestFigureSaving:
    def test_tsne_figure_saved(self):
        """验证 t-SNE 图保存到正确路径 (输出重定向到临时目录)."""
        x, y = generate_synthetic_features(n_per_level=50, seed=42)
        tsne_analysis(x, y)
        path = scripts.validate_features.OUTPUT_DIR / "tsne_feature_space.png"
        assert path.exists(), f"文件不存在: {path}"
        assert path.stat().st_size > 0, f"文件为空: {path}"

    def test_shap_figure_saved(self):
        """验证 SHAP 图保存到正确路径 (输出重定向到临时目录)."""
        x, y = generate_synthetic_features(n_per_level=50, seed=43)
        shap_analysis(x, y)
        path = scripts.validate_features.OUTPUT_DIR / "shap_importance.png"
        assert path.exists(), f"文件不存在: {path}"
        assert path.stat().st_size > 0, f"文件为空: {path}"

    def test_correlation_figure_saved(self):
        """验证相关性热力图保存到正确路径 (输出重定向到临时目录)."""
        x, y = generate_synthetic_features(n_per_level=50, seed=44)
        correlation_analysis(x, y)
        path = scripts.validate_features.OUTPUT_DIR / "correlation_matrix.png"
        assert path.exists(), f"文件不存在: {path}"
        assert path.stat().st_size > 0, f"文件为空: {path}"
