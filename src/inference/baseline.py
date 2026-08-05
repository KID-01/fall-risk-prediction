"""
个体化基线管理模块
系统部署后前7天为基线采集期,计算各特征的均值、标准差和协方差矩阵
作为后续所有偏离判断的参照标准
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.inference.features import FeatureVector
from src.utils.config import get_config


@dataclass
class IndividualBaseline:
    """个体化活动模式基线"""

    person_id: str
    mean: np.ndarray                    # 各特征均值 (4,)
    std: np.ndarray                     # 各特征标准差 (4,)
    cov_inv: np.ndarray                 # 协方差矩阵的逆 (4,4),用于马氏距离
    sample_count: int = 0
    collection_days: int = 0
    is_ready: bool = False              # 是否完成基线采集

    FEATURE_NAMES = FeatureVector.FEATURE_NAMES

    def mahalanobis_distance(self, feature_vec: FeatureVector | np.ndarray) -> float:
        """计算特征向量到基线的马氏距离"""
        if not self.is_ready:
            return 0.0
        x = feature_vec.to_array() if isinstance(feature_vec, FeatureVector) else np.asarray(feature_vec)
        diff = x - self.mean
        return float(np.sqrt(diff @ self.cov_inv @ diff))

    def z_scores(self, feature_vec: FeatureVector | np.ndarray) -> np.ndarray:
        """计算各特征的Z-Score"""
        if not self.is_ready:
            return np.zeros(4)
        x = feature_vec.to_array() if isinstance(feature_vec, FeatureVector) else np.asarray(feature_vec)
        return (x - self.mean) / (self.std + 1e-8)

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "cov_inv": self.cov_inv.tolist(),
            "sample_count": self.sample_count,
            "collection_days": self.collection_days,
            "is_ready": self.is_ready,
        }

    @classmethod
    def from_dict(cls, d: dict) -> IndividualBaseline:
        return cls(
            person_id=d["person_id"],
            mean=np.array(d["mean"]),
            std=np.array(d["std"]),
            cov_inv=np.array(d["cov_inv"]),
            sample_count=d["sample_count"],
            collection_days=d["collection_days"],
            is_ready=d["is_ready"],
        )


class BaselineManager:
    """基线管理器: 采集、计算、存储、加载"""

    def __init__(self, db_path: str | None = None):
        config = get_config()
        self.collection_days = config.baseline.collection_days
        self.min_samples = config.baseline.min_samples
        db = db_path or str(Path(config.paths.baseline_db))
        Path(db).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db
        self._init_db()

    def _init_db(self):
        """初始化SQLite数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    walking_rhythm REAL,
                    step_amplitude REAL,
                    trunk_stability REAL,
                    activity_density REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    person_id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

    def add_sample(self, person_id: str, feature: FeatureVector):
        """添加一个特征样本到基线采集池"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO feature_samples
                   (person_id, timestamp, walking_rhythm, step_amplitude,
                    trunk_stability, activity_density)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (person_id, feature.timestamp, feature.walking_rhythm,
                 feature.step_amplitude, feature.trunk_stability, feature.activity_density),
            )

    def get_samples(self, person_id: str) -> np.ndarray:
        """获取指定人员的所有样本,返回 (N, 4) 数组"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT walking_rhythm, step_amplitude, trunk_stability, activity_density
                   FROM feature_samples WHERE person_id = ? ORDER BY timestamp""",
                (person_id,),
            ).fetchall()
        return np.array(rows, dtype=np.float64) if rows else np.empty((0, 4))

    def compute_baseline(self, person_id: str) -> IndividualBaseline:
        """根据已采集样本计算基线"""
        samples = self.get_samples(person_id)
        n = len(samples)

        if n == 0:
            return IndividualBaseline(
                person_id=person_id,
                mean=np.zeros(4),
                std=np.ones(4),
                cov_inv=np.eye(4),
                sample_count=0,
                is_ready=False,
            )

        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)

        # 协方差矩阵(正则化以防奇异)
        if n >= 4:
            cov = np.cov(samples.T)
            cov += np.eye(4) * 1e-6  # 正则化
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.eye(4)
        else:
            cov_inv = np.eye(4)

        is_ready = n >= self.min_samples

        baseline = IndividualBaseline(
            person_id=person_id,
            mean=mean,
            std=std,
            cov_inv=cov_inv,
            sample_count=n,
            is_ready=is_ready,
        )

        # 持久化
        with sqlite3.connect(self.db_path) as conn:
            import time
            conn.execute(
                """INSERT OR REPLACE INTO baselines (person_id, data_json, updated_at)
                   VALUES (?, ?, ?)""",
                (person_id, json.dumps(baseline.to_dict()), time.time()),
            )

        return baseline

    def load_baseline(self, person_id: str) -> IndividualBaseline | None:
        """从数据库加载已计算的基线"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data_json FROM baselines WHERE person_id = ?",
                (person_id,),
            ).fetchone()
        if row is None:
            return None
        return IndividualBaseline.from_dict(json.loads(row[0]))

    def reset_baseline(self, person_id: str):
        """重置基线(删除所有样本和基线)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM feature_samples WHERE person_id = ?", (person_id,))
            conn.execute("DELETE FROM baselines WHERE person_id = ?", (person_id,))
