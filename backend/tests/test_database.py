import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.database.storage import DEFAULT_DATABASE_PATH, Database
from backend.schemas.analysis import (
    AnalysisResponse,
    AnomalyResult,
    ClassifierResult,
    InvestigationResponse,
)
from backend.services.investigator import SensorReading


class DatabaseTests(unittest.TestCase):
    def test_database_path_uses_environment_override(self) -> None:
        configured_path = Path(tempfile.gettempdir()) / "deployed.sqlite3"

        with patch.dict("os.environ", {"DATABASE_PATH": str(configured_path)}):
            database = Database()

        self.assertEqual(database.path, configured_path)
        self.assertNotEqual(database.path, DEFAULT_DATABASE_PATH)

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temporary_directory.name) / "test.sqlite3")
        self.database.initialize()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_records_readings_and_lists_registered_machines(self) -> None:
        reading_id = self.database.record_sensor_reading(
            "press-01",
            self._reading(),
        )

        machines = self.database.list_machines()

        self.assertGreater(reading_id, 0)
        self.assertEqual(len(machines), 1)
        self.assertEqual(machines[0]["machine_id"], "press-01")
        self.assertEqual(machines[0]["reading_count"], 1)

    def test_analysis_is_saved_with_reading_and_retrievable_as_latest_risk(self) -> None:
        reading_id, analysis_id = self.database.record_analysis(
            "press-02",
            self._reading(),
            self._analysis(),
        )

        risk = self.database.latest_risk("press-02")
        latest_analysis = self.database.latest_analysis("press-02")
        maintenance = self.database.maintenance_history("press-02")

        self.assertIsNotNone(risk)
        self.assertIsNotNone(latest_analysis)
        self.assertEqual(risk["reading_id"], reading_id)
        self.assertEqual(risk["analysis_id"], analysis_id)
        self.assertEqual(risk["predicted_label"], 1)
        self.assertEqual(latest_analysis["analysis_id"], analysis_id)
        self.assertEqual(latest_analysis["investigation"]["summary"], "Review required.")
        self.assertEqual(maintenance["latest_analysis_id"], analysis_id)
        self.assertEqual(maintenance["suggested_checks"], ["Inspect cooling readings."])

    def test_sensor_history_is_ordered_and_limited(self) -> None:
        self.database.record_sensor_reading("press-04", self._reading())
        self.database.record_sensor_reading("press-04", self._reading())

        history = self.database.sensor_history("press-04", limit=1)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["machine_id"], "press-04")
        self.assertIsNone(self.database.sensor_history("unknown"))

    def test_records_human_reviewed_actions_and_rejects_unknown_machines(self) -> None:
        self.database.record_sensor_reading("press-03", self._reading())

        saved = self.database.record_maintenance_action(
            "press-03",
            "Inspected the cooling system",
            "A. Technician",
            "No visible blockage.",
        )

        self.assertIsNotNone(saved)
        self.assertEqual(saved["reviewed_by"], "A. Technician")
        self.assertEqual(
            self.database.maintenance_history("press-03")["actions"],
            [saved],
        )
        self.assertIsNone(
            self.database.record_maintenance_action("unknown", "Inspect", "Reviewer", None)
        )

    @staticmethod
    def _reading() -> SensorReading:
        return SensorReading(
            product_type="M",
            air_temperature_k=300.0,
            process_temperature_k=310.0,
            rotational_speed_rpm=1500.0,
            torque_nm=40.0,
            tool_wear_min=10.0,
        )

    @staticmethod
    def _analysis() -> AnalysisResponse:
        return AnalysisResponse(
            classifier=ClassifierResult(
                predicted_label=1,
                model_estimated_failure_probability=0.8,
                decision_threshold=0.5,
                note="Test estimate.",
            ),
            anomaly=AnomalyResult(
                reconstruction_error=0.2,
                threshold=0.1,
                flagged_as_anomaly=True,
                note="Test anomaly.",
            ),
            investigation=InvestigationResponse(
                summary="Review required.",
                evidence=[],
                recommended_checks=["Inspect cooling readings."],
                limitations=["Synthetic data."],
            ),
        )


if __name__ == "__main__":
    unittest.main()
