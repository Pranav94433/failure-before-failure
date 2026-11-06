"""SQLite persistence for machine readings, analyses, and maintenance actions."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from backend.schemas.analysis import AnalysisResponse
from backend.services.investigator import SensorReading

DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "failure_before_failure.db"


class Database:
    """Small SQLite data-access layer using one connection per operation."""

    def __init__(self, path: Path | None = None) -> None:
        configured_path = os.environ.get("DATABASE_PATH")
        if path is not None:
            self.path = path
        elif configured_path:
            self.path = Path(configured_path)
        else:
            self.path = DEFAULT_DATABASE_PATH

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS machines (
                    machine_id TEXT PRIMARY KEY
                        CHECK(length(trim(machine_id)) BETWEEN 1 AND 100),
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    machine_id TEXT NOT NULL REFERENCES machines(machine_id),
                    product_type TEXT NOT NULL CHECK(product_type IN ('L', 'M', 'H')),
                    air_temperature_k REAL NOT NULL CHECK(air_temperature_k > 0),
                    process_temperature_k REAL NOT NULL CHECK(process_temperature_k > 0),
                    rotational_speed_rpm REAL NOT NULL CHECK(rotational_speed_rpm >= 0),
                    torque_nm REAL NOT NULL CHECK(torque_nm >= 0),
                    tool_wear_min REAL NOT NULL CHECK(tool_wear_min >= 0),
                    recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reading_id INTEGER NOT NULL UNIQUE REFERENCES sensor_readings(id),
                    machine_id TEXT NOT NULL REFERENCES machines(machine_id),
                    classifier_predicted_label INTEGER NOT NULL CHECK(classifier_predicted_label IN (0, 1)),
                    classifier_probability REAL NOT NULL
                        CHECK(classifier_probability >= 0 AND classifier_probability <= 1),
                    anomaly_score REAL NOT NULL CHECK(anomaly_score >= 0),
                    anomaly_threshold REAL NOT NULL CHECK(anomaly_threshold >= 0),
                    anomaly_flag INTEGER NOT NULL CHECK(anomaly_flag IN (0, 1)),
                    investigator_summary TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    recommended_checks_json TEXT NOT NULL,
                    limitations_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE TABLE IF NOT EXISTS maintenance_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    machine_id TEXT NOT NULL REFERENCES machines(machine_id),
                    action TEXT NOT NULL CHECK(length(trim(action)) > 0),
                    reviewed_by TEXT NOT NULL CHECK(length(trim(reviewed_by)) > 0),
                    notes TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_sensor_readings_machine_time
                    ON sensor_readings(machine_id, recorded_at);
                CREATE INDEX IF NOT EXISTS idx_analysis_machine_time
                    ON analysis_results(machine_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_maintenance_machine_time
                    ON maintenance_actions(machine_id, created_at);
                """
            )
            analysis_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(analysis_results)").fetchall()
            }
            if "limitations_json" not in analysis_columns:
                connection.execute(
                    "ALTER TABLE analysis_results "
                    "ADD COLUMN limitations_json TEXT NOT NULL DEFAULT '[]'"
                )

    @staticmethod
    def _touch_machine(connection: sqlite3.Connection, machine_id: str) -> None:
        connection.execute(
            """
            INSERT INTO machines (machine_id) VALUES (?)
            ON CONFLICT(machine_id) DO UPDATE
            SET last_seen_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            """,
            (machine_id,),
        )

    @staticmethod
    def _insert_reading(
        connection: sqlite3.Connection,
        machine_id: str,
        reading: SensorReading,
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO sensor_readings (
                machine_id, product_type, air_temperature_k, process_temperature_k,
                rotational_speed_rpm, torque_nm, tool_wear_min
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                machine_id,
                reading.product_type,
                reading.air_temperature_k,
                reading.process_temperature_k,
                reading.rotational_speed_rpm,
                reading.torque_nm,
                reading.tool_wear_min,
            ),
        )
        return int(cursor.lastrowid)

    def record_sensor_reading(self, machine_id: str, reading: SensorReading) -> int:
        with self._connection() as connection:
            self._touch_machine(connection, machine_id)
            return self._insert_reading(connection, machine_id, reading)

    def record_analysis(
        self,
        machine_id: str,
        reading: SensorReading,
        result: AnalysisResponse,
    ) -> tuple[int, int]:
        with self._connection() as connection:
            self._touch_machine(connection, machine_id)
            reading_id = self._insert_reading(connection, machine_id, reading)
            cursor = connection.execute(
                """
                INSERT INTO analysis_results (
                    reading_id, machine_id, classifier_predicted_label,
                    classifier_probability, anomaly_score, anomaly_threshold,
                    anomaly_flag, investigator_summary, evidence_json,
                    recommended_checks_json, limitations_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reading_id,
                    machine_id,
                    result.classifier.predicted_label,
                    result.classifier.model_estimated_failure_probability,
                    result.anomaly.reconstruction_error,
                    result.anomaly.threshold,
                    int(result.anomaly.flagged_as_anomaly),
                    result.investigation.summary,
                    json.dumps(
                        [item.model_dump() for item in result.investigation.evidence],
                        ensure_ascii=False,
                    ),
                    json.dumps(result.investigation.recommended_checks, ensure_ascii=False),
                    json.dumps(result.investigation.limitations, ensure_ascii=False),
                ),
            )
            return reading_id, int(cursor.lastrowid)

    def list_machines(self) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT m.machine_id, m.created_at, m.last_seen_at,
                       COUNT(r.id) AS reading_count
                FROM machines AS m
                LEFT JOIN sensor_readings AS r ON r.machine_id = m.machine_id
                GROUP BY m.machine_id
                ORDER BY m.machine_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def machine_exists(self, machine_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM machines WHERE machine_id = ?",
                (machine_id,),
            ).fetchone()
        return row is not None

    def sensor_history(
        self,
        machine_id: str,
        limit: int = 20,
    ) -> list[dict[str, object]] | None:
        with self._connection() as connection:
            machine = connection.execute(
                "SELECT 1 FROM machines WHERE machine_id = ?",
                (machine_id,),
            ).fetchone()
            if machine is None:
                return None

            rows = connection.execute(
                """
                SELECT id AS reading_id, machine_id, product_type,
                       air_temperature_k, process_temperature_k,
                       rotational_speed_rpm, torque_nm, tool_wear_min, recorded_at
                FROM sensor_readings
                WHERE machine_id = ?
                ORDER BY recorded_at DESC, id DESC
                LIMIT ?
                """,
                (machine_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_risk(self, machine_id: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id AS analysis_id, reading_id, machine_id,
                       classifier_predicted_label AS predicted_label,
                       classifier_probability AS model_estimated_failure_probability,
                       anomaly_score, anomaly_threshold, anomaly_flag, created_at
                FROM analysis_results
                WHERE machine_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (machine_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def latest_analysis(self, machine_id: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id AS analysis_id, reading_id, machine_id,
                       classifier_predicted_label, classifier_probability,
                       anomaly_score, anomaly_threshold, anomaly_flag,
                       investigator_summary, evidence_json, recommended_checks_json,
                       limitations_json
                FROM analysis_results
                WHERE machine_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (machine_id,),
            ).fetchone()
        if row is None:
            return None

        return {
            "machine_id": row["machine_id"],
            "reading_id": int(row["reading_id"]),
            "analysis_id": int(row["analysis_id"]),
            "classifier": {
                "predicted_label": int(row["classifier_predicted_label"]),
                "model_estimated_failure_probability": float(row["classifier_probability"]),
                "decision_threshold": 0.5,
                "note": (
                    "This is a model estimate from synthetic training data, not a "
                    "calibrated real-world failure risk."
                ),
            },
            "anomaly": {
                "reconstruction_error": float(row["anomaly_score"]),
                "threshold": float(row["anomaly_threshold"]),
                "flagged_as_anomaly": bool(row["anomaly_flag"]),
                "note": "An anomaly is an unusual pattern, not proof of failure.",
            },
            "investigation": {
                "summary": row["investigator_summary"],
                "evidence": json.loads(row["evidence_json"]),
                "recommended_checks": json.loads(row["recommended_checks_json"]),
                "limitations": json.loads(row["limitations_json"]),
            },
        }

    def maintenance_history(self, machine_id: str) -> dict[str, object] | None:
        with self._connection() as connection:
            machine = connection.execute(
                "SELECT machine_id FROM machines WHERE machine_id = ?",
                (machine_id,),
            ).fetchone()
            if machine is None:
                return None

            latest = connection.execute(
                """
                SELECT id, recommended_checks_json
                FROM analysis_results
                WHERE machine_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (machine_id,),
            ).fetchone()
            actions = connection.execute(
                """
                SELECT id AS action_id, machine_id, action, reviewed_by, notes, created_at
                FROM maintenance_actions
                WHERE machine_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (machine_id,),
            ).fetchall()

        return {
            "machine_id": machine_id,
            "latest_analysis_id": int(latest["id"]) if latest is not None else None,
            "suggested_checks": (
                json.loads(latest["recommended_checks_json"]) if latest is not None else []
            ),
            "actions": [dict(row) for row in actions],
        }

    def record_maintenance_action(
        self,
        machine_id: str,
        action: str,
        reviewed_by: str,
        notes: str | None,
    ) -> dict[str, object] | None:
        with self._connection() as connection:
            machine = connection.execute(
                "SELECT 1 FROM machines WHERE machine_id = ?",
                (machine_id,),
            ).fetchone()
            if machine is None:
                return None

            cursor = connection.execute(
                """
                INSERT INTO maintenance_actions (machine_id, action, reviewed_by, notes)
                VALUES (?, ?, ?, ?)
                """,
                (machine_id, action, reviewed_by, notes),
            )
            row = connection.execute(
                """
                SELECT id AS action_id, machine_id, action, reviewed_by, notes, created_at
                FROM maintenance_actions
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
        if row is None:
            raise RuntimeError("The maintenance action was saved but could not be retrieved.")
        return dict(row)
