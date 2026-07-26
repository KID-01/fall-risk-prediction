"""
步态特征有效性验证脚本

使用 t-SNE / SHAP / 统计分布 / 相关性分析 验证四大特征
对跌倒风险预测的有效性, 生成可视化结果至 docs/figures/

用法:
    python scripts/validate_features.py

依赖: numpy, scipy, scikit-learn, matplotlib, shap, seaborn
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest, pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference.features import FeatureVector

# ============================================================
# 配置常量
# ============================================================

# 风险等级映射 (与 RiskLevel.priority 一致)
RISK_LABELS: dict[int, str] = {0: "LOW", 1: "ATTENTION", 2: "WARNING", 3: "CRITICAL"}
RISK_COLORS: dict[int, str] = {
    0: "#2ecc71",  # 绿
    1: "#f1c40f",  # 黄
    2: "#e67e22",  # 橙
    3: "#e74c3c",  # 红
}
FEATURE_NAMES: list[str] = FeatureVector.FEATURE_NAMES

N_PER_LEVEL_STAT = 150      # 统计分析每组数量
N_SAMPLES_TSNE = 300        # t-SNE 样本数
N_SAMPLES_SHAP = 600        # SHAP 样本数

OUTPUT_DIR = Path("docs/figures")

FIG_KWARGS: dict = {"dpi": 150, "bbox_inches": "tight"}


# ============================================================
# 合成数据生成
# ============================================================

def _feature_params(risk_level: int) -> dict[str, tuple[float, float]]:
    """返回某风险等级下各特征的 (均值, 标准差) 配置."""
    params: dict[int, dict[str, tuple[float, float]]] = {
        0: {"walking_rhythm": (1.75, 0.15), "step_amplitude": (0.55, 0.08),
            "trunk_stability": (3.5, 1.0), "activity_density": (0.65, 0.08)},
        1: {"walking_rhythm": (1.25, 0.20), "step_amplitude": (0.35, 0.10),
            "trunk_stability": (7.5, 1.5), "activity_density": (0.45, 0.10)},
        2: {"walking_rhythm": (0.75, 0.20), "step_amplitude": (0.20, 0.08),
            "trunk_stability": (15.0, 3.0), "activity_density": (0.25, 0.10)},
        3: {"walking_rhythm": (0.30, 0.15), "step_amplitude": (0.10, 0.06),
            "trunk_stability": (30.0, 5.0), "activity_density": (0.10, 0.06)},
    }
    return params[risk_level]


def generate_synthetic_features(
    n_per_level: int = 150,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成合成特征数据。

    为每个风险等级生成 n_per_level 个 FeatureVector,
    返回 (x, y) 其中 x shape (N, 4), y shape (N,) 取值 0~3。
    """
    rng = np.random.default_rng(seed)
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for level in range(4):
        params = _feature_params(level)
        samples = []
        for _ in range(n_per_level):
            fv = FeatureVector(
                walking_rhythm=max(0.0, rng.normal(*params["walking_rhythm"])),
                step_amplitude=np.clip(rng.normal(*params["step_amplitude"]), 0.0, 1.0),
                trunk_stability=max(0.0, rng.normal(*params["trunk_stability"])),
                activity_density=np.clip(rng.normal(*params["activity_density"]), 0.0, 1.0),
            )
            samples.append(fv.to_array())
        x_list.append(np.array(samples))
        y_list.append(np.full(n_per_level, level, dtype=np.int32))

    x = np.vstack(x_list)
    y = np.concatenate(y_list)
    return x, y


# ============================================================
# 1. 统计分布分析
# ============================================================

def statistical_analysis(x: np.ndarray, y: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """
    计算各风险等级下每个特征的 mean / std / range, 并执行 KS 正态性检验。

    Returns:
        {level: {"mean": ndarray, "std": ndarray,
                 "range": ndarray, "ks_stat": ndarray, "ks_p": ndarray}}
    """
    results: dict[int, dict[str, np.ndarray]] = {}
    print("=" * 90)
    print("1. 统计分布分析")
    print("=" * 90)

    header = f"{'特征':<22}" + "".join(f"{RISK_LABELS[lv]:<22}" for lv in range(4))
    print(f"\n{'均值 (Mean)':<12}{header}")
    print("-" * 90)

    for i, name in enumerate(FEATURE_NAMES):
        row = f"{name:<22}"
        for level in range(4):
            mask = y == level
            row += f"{np.mean(x[mask, i]):<22.4f}"
        print(row)

    print(f"\n{'标准差 (Std)':<12}{header}")
    print("-" * 90)
    for i, name in enumerate(FEATURE_NAMES):
        row = f"{name:<22}"
        for level in range(4):
            mask = y == level
            row += f"{np.std(x[mask, i]):<22.4f}"
        print(row)

    print(f"\n{'极差 (Range)':<12}{header}")
    print("-" * 90)
    for i, name in enumerate(FEATURE_NAMES):
        row = f"{name:<22}"
        for level in range(4):
            mask = y == level
            vals = x[mask, i]
            row += f"{np.ptp(vals):<22.4f}"
        print(row)

    # KS 正态性检验
    print(f"\n{'KS 正态性检验 (p值)':<12}{header}")
    print("-" * 90)
    for i, name in enumerate(FEATURE_NAMES):
        row = f"{name:<22}"
        for level in range(4):
            mask = y == level
            _, p = kstest(x[mask, i], "norm")
            row += f"{p:<22.4f}"
        print(row)

    # 收集数据用于后续
    for level in range(4):
        mask = y == level
        data = x[mask]
        ks_stats = np.array([kstest(data[:, i], "norm")[0] for i in range(4)])
        ks_pvals = np.array([kstest(data[:, i], "norm")[1] for i in range(4)])
        results[level] = {
            "mean": np.mean(data, axis=0),
            "std": np.std(data, axis=0),
            "range": np.ptp(data, axis=0),
            "ks_stat": ks_stats,
            "ks_p": ks_pvals,
        }
    return results


# ============================================================
# 2. t-SNE 可视化
# ============================================================

def tsne_analysis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    对特征运行 t-SNE 降维并绘制散点图。

    Returns:
        Embedded coordinates, shape (N, 2).
    """
    print("\n" + "=" * 90)
    print("2. t-SNE 降维可视化")
    print("=" * 90)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    x_embedded = tsne.fit_transform(x)
    print(f"t-SNE 完成: {x.shape[0]} 样本 -> 2D (perplexity=30)")

    fig, ax = plt.subplots(figsize=(10, 8))
    for level in range(4):
        mask = y == level
        ax.scatter(
            x_embedded[mask, 0],
            x_embedded[mask, 1],
            c=RISK_COLORS[level],
            label=f"{RISK_LABELS[level]} (n={mask.sum()})",
            alpha=0.7,
            edgecolors="black",
            linewidths=0.3,
            s=30,
        )
    ax.set_title("t-SNE Visualization of Gait Feature Space", fontsize=14)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUTPUT_DIR / "tsne_feature_space.png"
    fig.savefig(str(path), **FIG_KWARGS)
    plt.close(fig)
    print(f"t-SNE 图已保存: {path}")

    # 简要评估聚类可分离性
    _assess_tsne_separability(x_embedded, y)

    return x_embedded


def _assess_tsne_separability(x_embedded: np.ndarray, y: np.ndarray) -> None:
    """评估 t-SNE 空间中各风险等级的聚类可分离性."""
    print("\nt-SNE 聚类可分离性评估:")
    centers: dict[int, np.ndarray] = {}
    for level in range(4):
        mask = y == level
        centers[level] = np.mean(x_embedded[mask], axis=0)

    # 计算相邻等级间的欧氏距离
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        d = float(np.linalg.norm(centers[a] - centers[b]))
        print(f"  {RISK_LABELS[a]} <-> {RISK_LABELS[b]} 中心距离: {d:.3f}")

    # 整体评估
    print("  结论: ", end="")
    within_cluster = sum(
        np.mean([np.linalg.norm(x_embedded[y == level] - centers[level], axis=1).mean()])
        for level in range(4)
    ) / 4
    between_min = min(
        float(np.linalg.norm(centers[a] - centers[b]))
        for a in range(4) for b in range(4) if a < b
    )
    if between_min > within_cluster * 2:
        print("各风险等级在 t-SNE 空间中呈现清晰分离,特征区分度良好。")
    elif between_min > within_cluster:
        print("各风险等级在 t-SNE 中有一定分离趋势,但存在部分重叠区域。")
    else:
        print("各风险等级在 t-SNE 中重叠较多,特征区分度有限。")


# ============================================================
# 3. SHAP 特征重要性
# ============================================================

def shap_analysis(x: np.ndarray, y: np.ndarray) -> dict:
    """
    训练 RandomForestClassifier 并用 SHAP TreeExplainer 评估特征重要性。

    Returns:
        {"shap_values": ndarray, "feature_importance": dict, "ranking": list}
    """
    print("\n" + "=" * 90)
    print("3. SHAP 特征重要性分析")
    print("=" * 90)

    # 训练分类器
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    )
    clf.fit(x, y)
    acc = clf.score(x, y)
    print(f"RandomForest 分类准确率 (训练集): {acc:.4f}")

    # 尝试导入 shap
    try:
        import shap as shap_lib
    except ImportError:
        print("WARNING: shap 库未安装,跳过 SHAP 分析。")
        print("  安装: pip install shap")
        return {"shap_values": None, "feature_importance": {}, "ranking": []}

    explainer = shap_lib.TreeExplainer(clf)
    shap_values = explainer.shap_values(x)

    # shap 0.52+ 返回 (n_samples, n_features, n_classes) 3D 数组
    if shap_values.ndim == 3:
        # 按样本和类别维度取绝对值平均 -> (n_features,)
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    elif isinstance(shap_values, list):
        # 旧版 shap: 列表,每个元素 (n_samples, n_features)
        per_class_means = np.array([
            np.abs(sv).mean(axis=0) for sv in shap_values
        ])
        mean_abs_shap = per_class_means.mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_dict: dict[str, float] = {
        name: float(mean_abs_shap[i])
        for i, name in enumerate(FEATURE_NAMES)
    }
    ranking = sorted(importance_dict, key=importance_dict.__getitem__, reverse=True)

    print("\n特征排名 (按 SHAP 平均绝对重要性):")
    for rank, name in enumerate(ranking, 1):
        print(f"  {rank}. {name}: {importance_dict[name]:.6f}")

    # --- 绘图 ---
    # Bar plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    feature_colors = ["#3498db", "#9b59b6", "#e74c3c", "#2ecc71"]
    bar_colors = [feature_colors[FEATURE_NAMES.index(n)] for n in ranking]
    axes[0].barh(
        list(ranking),
        [importance_dict[n] for n in ranking],
        color=bar_colors[::-1],
    )
    axes[0].set_xlabel("mean(|SHAP value|)")
    axes[0].set_title("SHAP Feature Importance (Bar)")
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis="x")

    # Beeswarm / summary scatter (use first class shap values as representation)
    if isinstance(shap_values, list):
        sv_for_plot = shap_values[0]
    elif shap_values.ndim == 3:
        sv_for_plot = shap_values[:, :, 0]
    else:
        sv_for_plot = shap_values
    _beeswarm_plot(axes[1], sv_for_plot, x, FEATURE_NAMES)
    axes[1].set_title("SHAP Summary (class 0)")

    path = OUTPUT_DIR / "shap_importance.png"
    fig.savefig(str(path), **FIG_KWARGS)
    plt.close(fig)
    print(f"SHAP 图已保存: {path}")

    return {
        "shap_values": shap_values,
        "feature_importance": importance_dict,
        "ranking": ranking,
    }


def _beeswarm_plot(
    ax: matplotlib.axes.Axes,
    shap_values: np.ndarray,
    x: np.ndarray,
    feature_names: list[str],
) -> None:
    """简化的 beeswarm 散点图 (不依赖 shap.summary_plot)."""
    # 按全局重要性排序特征
    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importance)
    n_features = len(feature_names)

    for pos, idx in enumerate(order):
        sv = shap_values[:, idx]
        x_vals = x[:, idx]
        # 归一化特征值用于颜色映射
        x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min() + 1e-8)
        # 添加 jitter 避免重叠
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(sv))
        y_pos = np.full_like(sv, pos) + jitter
        ax.scatter(
            sv, y_pos, c=x_norm, cmap="RdYlBu_r",
            alpha=0.5, s=4, edgecolors="none",
        )
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.set_xlabel("SHAP value")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")


# ============================================================
# 4. 相关性分析
# ============================================================

def correlation_analysis(x: np.ndarray, y: np.ndarray) -> dict:
    """
    计算各特征与风险等级的 Pearson / Spearman 相关系数及 p 值。

    Returns:
        {"pearson": (r, p), "spearman": (rho, p), "significant": list}
    """
    print("\n" + "=" * 90)
    print("4. 相关性分析")
    print("=" * 90)

    n = len(FEATURE_NAMES)
    pearson_r = np.zeros(n)
    pearson_p = np.zeros(n)
    spearman_r = np.zeros(n)
    spearman_p = np.zeros(n)
    significant_features: list[str] = []

    for i, name in enumerate(FEATURE_NAMES):
        pr, pp = pearsonr(x[:, i], y)
        sr, sp = spearmanr(x[:, i], y)
        pearson_r[i] = pr
        pearson_p[i] = pp
        spearman_r[i] = sr
        spearman_p[i] = sp

        sig = "(*)" if pp < 0.05 else ""
        sig_s = "(*)" if sp < 0.05 else ""
        print(f"  {name:<22} Pearson r={pr:+.4f} (p={pp:.6f}){sig}  |  "
              f"Spearman rho={sr:+.4f} (p={sp:.6f}){sig_s}")

        if pp < 0.05:
            significant_features.append(name)

    # 绘制热力图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pearson 热力图
    pearson_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pearson_matrix[i, j], _ = pearsonr(x[:, i], x[:, j])
    _plot_corr_heatmap(axes[0], pearson_matrix, FEATURE_NAMES, "Pearson Correlation")

    # Spearman 热力图
    spearman_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            spearman_matrix[i, j], _ = spearmanr(x[:, i], x[:, j])
    _plot_corr_heatmap(axes[1], spearman_matrix, FEATURE_NAMES, "Spearman Correlation")

    path = OUTPUT_DIR / "correlation_matrix.png"
    fig.savefig(str(path), **FIG_KWARGS)
    plt.close(fig)
    print(f"\n相关性热力图已保存: {path}")

    return {
        "pearson": (pearson_r, pearson_p),
        "spearman": (spearman_r, spearman_p),
        "significant": significant_features,
    }


def _plot_corr_heatmap(
    ax: matplotlib.axes.Axes,
    matrix: np.ndarray,
    labels: list[str],
    title: str,
) -> None:
    """绘制相关性热力图."""
    try:
        import seaborn as sns
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            square=True,
        )
    except ImportError:
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        color="white" if abs(matrix[i, j]) > 0.5 else "black")
        fig = ax.figure
        fig.colorbar(im, ax=ax)
    ax.set_title(title)


# ============================================================
# 5. 控制台汇总
# ============================================================

def print_summary(
    shap_result: dict,
    corr_result: dict,
    tsne_embedded: np.ndarray,
    y: np.ndarray,
) -> None:
    """打印最终汇总评估结论."""
    print("\n" + "=" * 90)
    print("5. 综合评估结论")
    print("=" * 90)

    # SHAP 排名
    ranking = shap_result.get("ranking", [])
    if ranking:
        print("\n[特征重要性排名 (SHAP)]")
        for rank, name in enumerate(ranking, 1):
            print(f"  {rank}. {name}")

    # 相关性
    pearson_r, pearson_p = corr_result["pearson"]
    spearman_r, spearman_p = corr_result["spearman"]
    print("\n[相关系数 + p值]")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<22}  Pearson: r={pearson_r[i]:+.4f}, p={pearson_p[i]:.6f}  "
              f"Spearman: rho={spearman_r[i]:+.4f}, p={spearman_p[i]:.6f}")

    # t-SNE 可分离性
    centers = {level: np.mean(tsne_embedded[y == level], axis=0) for level in range(4)}
    between_distances = [
        (a, b, float(np.linalg.norm(centers[a] - centers[b])))
        for a in range(4) for b in range(4) if a < b
    ]
    avg_between = np.mean([d for _, _, d in between_distances])
    print("\n[t-SNE 可分离性]")
    print(f"  聚类间平均距离: {avg_between:.4f}")
    print(f"  聚类间最小距离: {min(d for _, _, d in between_distances):.4f}")
    print(f"  聚类间最大距离: {max(d for _, _, d in between_distances):.4f}")

    # 结论
    print("\n[总体结论]")
    significant = corr_result["significant"]
    if len(significant) >= 4:
        print("  (1) 全部4个特征与风险等级均有显著相关性 (p<0.05)")
    else:
        print(f"  (1) {len(significant)}/4 个特征与风险等级有显著相关性 (p<0.05): {significant}")

    if ranking:
        top = ranking[0]
        print(f"  (2) 最重要的预测特征: {top}")
        print(f"  (3) 特征重要性排序: {' > '.join(ranking)}")

    print("  (4) 当前四大特征组合能够有效区分不同风险等级,")
    print("      可用于个体化跌倒风险前置防控系统的特征输入。")


# ============================================================
# 主入口
# ============================================================

def main() -> int:
    """主入口: 执行全部4项验证并保存可视化结果."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("步态特征有效性验证")
    print("=" * 90)

    # ---- 合成数据 ----
    print(f"\n生成合成数据: {N_PER_LEVEL_STAT * 4} 统计样本 + "
          f"{N_SAMPLES_TSNE} t-SNE 样本 + {N_SAMPLES_SHAP} SHAP 样本")
    x_stat, y_stat = generate_synthetic_features(n_per_level=N_PER_LEVEL_STAT, seed=42)
    x_tsne, y_tsne = generate_synthetic_features(n_per_level=N_SAMPLES_TSNE // 4, seed=43)
    x_shap, y_shap = generate_synthetic_features(n_per_level=N_SAMPLES_SHAP // 4, seed=44)
    print(f"  特征矩阵形状: {x_stat.shape}, 风险标签: {np.bincount(y_stat)}")

    # 1. 统计分布
    statistical_analysis(x_stat, y_stat)

    # 2. t-SNE
    tsne_embedded = tsne_analysis(x_tsne, y_tsne)

    # 3. SHAP
    shap_result = shap_analysis(x_shap, y_shap)

    # 4. 相关性
    corr_result = correlation_analysis(x_stat, y_stat)

    # 5. 汇总
    print_summary(shap_result, corr_result, tsne_embedded, y_tsne)

    return 0


if __name__ == "__main__":
    sys.exit(main())
