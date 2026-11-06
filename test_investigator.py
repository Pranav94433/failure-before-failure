import unittest

from backend.services.investigator import (
    ModelOutputs,
    SensorReading,
    investigate,
)


class InvestigatorTests(unittest.TestCase):
    def test_reports_matching_dataset_patterns_without_claiming_failure(self) -> None:
        reading = SensorReading(
            product_type="L",
            air_temperature_k=300.0,
            process_temperature_k=308.0,
            rotational_speed_rpm=1300.0,
            torque_nm=50.0,
            tool_wear_min=230.0,
        )

        report = investigate(reading)
        rule_ids = {item.rule_id for item in report.evidence}

        self.assertEqual(
            rule_ids,
            {"AI4I-HDF-pattern", "AI4I-OSF-pattern", "AI4I-TWF-wear-band"},
        )
        self.assertIn("none confirms a failure", report.summary)
        self.assertEqual(len(report.recommended_checks), 3)

    def test_reports_power_pattern_using_torque_and_speed(self) -> None:
        reading = SensorReading(
            product_type="M",
            air_temperature_k=300.0,
            process_temperature_k=310.0,
            rotational_speed_rpm=1400.0,
            torque_nm=100.0,
            tool_wear_min=10.0,
        )

        report = investigate(reading)

        self.assertEqual([item.rule_id for item in report.evidence], ["AI4I-PWF-pattern"])
        self.assertIn("approximately 14661 W", report.evidence[0].observation)

    def test_anomaly_threshold_is_evidence_but_probability_is_not_a_rule(self) -> None:
        reading = SensorReading(
            product_type="H",
            air_temperature_k=300.0,
            process_temperature_k=310.0,
            rotational_speed_rpm=1500.0,
            torque_nm=40.0,
            tool_wear_min=10.0,
        )
        outputs = ModelOutputs(
            anomaly_score=0.2,
            anomaly_threshold=0.15,
            failure_probability=0.9,
        )

        report = investigate(reading, outputs)

        self.assertEqual([item.rule_id for item in report.evidence], ["autoencoder-threshold"])
        self.assertEqual(report.model_outputs.failure_probability, 0.9)
        self.assertIn("not a calibrated real-world failure risk", report.limitations[2])

    def test_rejects_invalid_model_output_and_product_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided together"):
            ModelOutputs(anomaly_score=0.1)
        with self.assertRaisesRegex(ValueError, "product_type"):
            SensorReading("X", 300.0, 310.0, 1500.0, 40.0, 10.0)


if __name__ == "__main__":
    unittest.main()
