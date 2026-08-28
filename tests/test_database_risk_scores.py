from __future__ import annotations

from src.api.database import Database


def _database(tmp_path):
    database = object.__new__(Database)
    database._initialized = True
    database.db_path = str(tmp_path / "app.db")
    database._init_tables()
    return database


def test_old_and_new_risk_scores_share_standardized_history(tmp_path):
    database = _database(tmp_path)
    database.insert_risk_record(3.0, "attention", person_id="old")
    database.insert_risk_record(
        82.5,
        "critical",
        person_id="new",
        risk_score_source="overall_engineering_v1",
        human_risk_score=80,
        environment_risk_score=72,
        interaction_risk_score=82.5,
        reason_codes=["human_context_compound"],
    )

    old = database.query_risk_records(person_id="old")[0]
    new = database.query_risk_records(person_id="new")[0]

    assert old["raw_risk_score"] == 3.0
    assert old["risk_score"] == 50.0
    assert new["risk_score"] == 82.5
    assert new["reason_codes"] == ["human_context_compound"]
