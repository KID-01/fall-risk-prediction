"""
数据库层 — SQLite 持久化
两张核心表:
  ① risk_records — 风险评分历史记录
  ② alert_events — 告警事件日志
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class Database:
    """SQLite 数据库管理器"""

    _instance: Database | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        config = get_config()
        db_path = Path(config.paths.baseline_db).parent / "app.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """初始化数据库表"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    device_id TEXT DEFAULT 'default',
                    person_id TEXT DEFAULT 'default',
                    risk_score REAL,
                    risk_level TEXT,
                    gait_features_json TEXT,
                    env_features_json TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    device_id TEXT DEFAULT 'default',
                    person_id TEXT DEFAULT 'default',
                    alert_level TEXT NOT NULL,
                    risk_score REAL,
                    message TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    video_clip_path TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_risk_timestamp ON risk_records(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON alert_events(timestamp)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    device_id TEXT DEFAULT 'default',
                    person_id TEXT DEFAULT 'default',
                    category TEXT NOT NULL,
                    label TEXT NOT NULL,
                    class_index INTEGER,
                    score REAL NOT NULL,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audio_timestamp ON audio_events(timestamp)
            """)
            self._ensure_columns(
                conn,
                "risk_records",
                {
                    "risk_score_source": "TEXT DEFAULT 'mahalanobis_v0'",
                    "raw_deviation_score": "REAL",
                    "human_risk_score": "REAL",
                    "environment_risk_score": "REAL",
                    "interaction_risk_score": "REAL",
                    "reason_codes_json": "TEXT",
                },
            )
            self._ensure_columns(
                conn,
                "alert_events",
                {
                    "risk_score_source": "TEXT DEFAULT 'mahalanobis_v0'",
                    "reason_codes_json": "TEXT",
                },
            )
        log.info(f"数据库初始化完成: {self.db_path}")

    @staticmethod
    def _ensure_columns(
        conn: sqlite3.Connection,
        table: str,
        columns: dict[str, str],
    ) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, declaration in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    # ── 风险记录 ──

    def insert_risk_record(
        self,
        risk_score: float,
        risk_level: str,
        person_id: str = "default",
        device_id: str = "default",
        gait_features: dict | None = None,
        env_features: dict | None = None,
        risk_score_source: str = "mahalanobis_v0",
        raw_deviation_score: float | None = None,
        human_risk_score: float | None = None,
        environment_risk_score: float | None = None,
        interaction_risk_score: float | None = None,
        reason_codes: list[str] | None = None,
    ) -> int:
        """插入一条风险记录"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO risk_records
                   (timestamp, device_id, person_id, risk_score, risk_level,
                    gait_features_json, env_features_json, risk_score_source,
                    raw_deviation_score, human_risk_score, environment_risk_score,
                    interaction_risk_score, reason_codes_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(), device_id, person_id, risk_score, risk_level,
                    json.dumps(gait_features, ensure_ascii=False) if gait_features else None,
                    json.dumps(env_features, ensure_ascii=False) if env_features else None,
                    risk_score_source,
                    raw_deviation_score,
                    human_risk_score,
                    environment_risk_score,
                    interaction_risk_score,
                    json.dumps(reason_codes or [], ensure_ascii=False),
                ),
            )
            return cursor.lastrowid

    def query_risk_records(
        self,
        person_id: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """查询风险记录(支持时间范围和分页)"""
        query = "SELECT * FROM risk_records WHERE 1=1"
        params: list[Any] = []

        if person_id:
            query += " AND person_id = ?"
            params.append(person_id)
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._normalize_risk_record(dict(row)) for row in rows]

    @staticmethod
    def _normalize_risk_record(record: dict) -> dict:
        raw_score = record.get("risk_score")
        source = record.get("risk_score_source") or "mahalanobis_v0"
        if raw_score is None:
            display_score = 0.0
        elif source == "mahalanobis_v0":
            display_score = min(100.0, max(0.0, float(raw_score) / 6.0 * 100.0))
        else:
            display_score = min(100.0, max(0.0, float(raw_score)))
        record["raw_risk_score"] = raw_score
        record["risk_score"] = round(display_score, 2)
        try:
            record["reason_codes"] = json.loads(record.get("reason_codes_json") or "[]")
        except json.JSONDecodeError:
            record["reason_codes"] = []
        return record

    # ── 告警事件 ──

    def insert_alert_event(
        self,
        alert_level: str,
        message: str,
        risk_score: float | None = None,
        person_id: str = "default",
        device_id: str = "default",
        video_clip_path: str | None = None,
        risk_score_source: str = "mahalanobis_v0",
        reason_codes: list[str] | None = None,
    ) -> int:
        """插入一条告警事件"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO alert_events
                   (timestamp, device_id, person_id, alert_level, risk_score,
                    message, video_clip_path, risk_score_source, reason_codes_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(), device_id, person_id, alert_level, risk_score,
                    message, video_clip_path, risk_score_source,
                    json.dumps(reason_codes or [], ensure_ascii=False),
                ),
            )
            return cursor.lastrowid

    def query_alert_events(
        self,
        alert_level: str | None = None,
        person_id: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        acknowledged: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """查询告警事件"""
        query = "SELECT * FROM alert_events WHERE 1=1"
        params: list[Any] = []

        if alert_level:
            query += " AND alert_level = ?"
            params.append(alert_level)
        if person_id:
            query += " AND person_id = ?"
            params.append(person_id)
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)
        if acknowledged is not None:
            query += " AND acknowledged = ?"
            params.append(acknowledged)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._normalize_alert_record(dict(row)) for row in rows]

    @staticmethod
    def _normalize_alert_record(record: dict) -> dict:
        raw_score = record.get("risk_score")
        source = record.get("risk_score_source") or "mahalanobis_v0"
        if raw_score is not None:
            record["raw_risk_score"] = raw_score
            record["risk_score"] = round(
                min(
                    100.0,
                    max(
                        0.0,
                        float(raw_score) / 6.0 * 100.0
                        if source == "mahalanobis_v0"
                        else float(raw_score),
                    ),
                ),
                2,
            )
        try:
            record["reason_codes"] = json.loads(record.get("reason_codes_json") or "[]")
        except json.JSONDecodeError:
            record["reason_codes"] = []
        return record

    def acknowledge_alert(self, alert_id: int) -> bool:
        """确认告警"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "UPDATE alert_events SET acknowledged = 1 WHERE id = ?",
                (alert_id,),
            )
            return cursor.rowcount > 0

    # ── 音频事件 ──

    def insert_audio_events(
        self,
        events: list,
        person_id: str = "default",
        device_id: str = "default",
    ) -> int:
        """批量插入音频事件"""
        if not events:
            return 0
        rows = [
            (e.timestamp, device_id, person_id, e.category.value, e.label, e.class_index, e.score)
            for e in events
        ]
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT INTO audio_events
                   (timestamp, device_id, person_id, category, label, class_index, score)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)

    def query_audio_events(
        self,
        person_id: str | None = None,
        category: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """查询音频事件历史"""
        query = "SELECT * FROM audio_events WHERE 1=1"
        params: list[Any] = []

        if person_id:
            query += " AND person_id = ?"
            params.append(person_id)
        if category:
            query += " AND category = ?"
            params.append(category)
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ── 统计 ──

    def get_stats(self, hours: int = 24) -> dict:
        """获取统计面板数据"""
        cutoff = time.time() - hours * 3600
        with self._get_conn() as conn:
            total_risk = conn.execute(
                "SELECT COUNT(*) FROM risk_records WHERE timestamp >= ?", (cutoff,)
            ).fetchone()[0]
            total_alerts = conn.execute(
                "SELECT COUNT(*) FROM alert_events WHERE timestamp >= ?", (cutoff,)
            ).fetchone()[0]
            alerts_by_level = conn.execute(
                """SELECT alert_level, COUNT(*) as count FROM alert_events
                   WHERE timestamp >= ? GROUP BY alert_level""",
                (cutoff,),
            ).fetchall()
            risk_rows = conn.execute(
                "SELECT risk_score, risk_score_source FROM risk_records WHERE timestamp >= ?",
                (cutoff,),
            ).fetchall()

        normalized_scores = [
            self._normalize_risk_record(dict(row))["risk_score"] for row in risk_rows
        ]
        avg_risk = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0

        return {
            "hours": hours,
            "total_risk_records": total_risk,
            "total_alerts": total_alerts,
            "alerts_by_level": {r["alert_level"]: r["count"] for r in alerts_by_level},
            "avg_risk_score": round(avg_risk, 2),
        }
