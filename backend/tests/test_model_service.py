import unittest

from backend.services.investigator import SensorReading
from backend.services.model_service import AnalysisService


class AnalysisServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = AnalysisService.from_saved_models()

    def test_saved_models_return_structured_outputs_for_a_sensor_reading(self) -> None:
        reading = SensorReading(
            product_type="M",
            air_temperature_k=300.0,
            process_temperature_k=310.0,
            rotational_speed_rpm=1500.0,
            torque_nm=40.0,
            tool_wear_min=10.0,
        )

        result = self.service.analyze(reading)

        self.assertIn(result.classifier.predicted_label, (0, 1))
        self.assertGreaterEqual(result.classifier.model_estimated_failure_probability, 0)
        self.assertLessEqual(result.classifier.model_estimated_failure_probability, 1)
        self.assertGreaterEqual(result.anomaly.reconstruction_error, 0)
        self.assertGreaterEqual(result.anomaly.threshold, 0)
        self.assertTrue(result.investigation.summary)
        self.assertTrue(result.investigation.limitations)


if __name__ == "__main__":
    unittest.main()
