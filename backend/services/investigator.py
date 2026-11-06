"""Transparent, dataset-grounded evidence checks for industrial sensor readings."""

from __future__ import annotations

import math
from dataclasses import dataclass

HDF_TEMPERATURE_DIFFERENCE_K = 8.6
HDF_MAX_ROTATIONAL_SPEED_RPM = 1380
PWF_MIN_POWER_W = 3500
PWF_MAX_POWER_W = 9000
OSF_TORQUE_WEAR_LIMITS = {"L": 11000, "M": 12000, "H": 13000}
TWF_TOOL_WEAR_MIN_MINUTES = 200
TWF_TOOL_WEAR_MAX_MINUTES = 240


@dataclass(frozen=True)
class SensorReading:
    product_type: str
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float

    def __post_init__(self) -> None:
        if self.product_type not in OSF_TORQUE_WEAR_LIMITS:
            raise ValueError("product_type must be one of: L, M, H.")

        values = {
            "air_temperature_k": self.air_temperature_k,
            "process_temperature_k": self.process_temperature_k,
            "rotational_speed_rpm": self.rotational_speed_rpm,
            "torque_nm": self.torque_nm,
            "tool_wear_min": self.tool_wear_min,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.air_temperature_k <= 0 or self.process_temperature_k <= 0:
            raise ValueError("Temperatures in Kelvin must be greater than zero.")
        if self.rotational_speed_rpm < 0:
            raise ValueError("rotational_speed_rpm cannot be negative.")
        if self.torque_nm < 0:
            raise ValueError("torque_nm cannot be negative.")
        if self.tool_wear_min < 0:
            raise ValueError("tool_wear_min cannot be negative.")


@dataclass(frozen=True)
class ModelOutputs:
    anomaly_score: float | None = None
    anomaly_threshold: float | None = None
    failure_probability: float | None = None

    def __post_init__(self) -> None:
        if (self.anomaly_score is None) != (self.anomaly_threshold is None):
            raise ValueError("anomaly_score and anomaly_threshold must be provided together.")

        for name, value in (
            ("anomaly_score", self.anomaly_score),
            ("anomaly_threshold", self.anomaly_threshold),
            ("failure_probability", self.failure_probability),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")

        if self.anomaly_score is not None and self.anomaly_score < 0:
            raise ValueError("anomaly_score cannot be negative.")
        if self.anomaly_threshold is not None and self.anomaly_threshold < 0:
            raise ValueError("anomaly_threshold cannot be negative.")
        if self.failure_probability is not None and not 0 <= self.failure_probability <= 1:
            raise ValueError("failure_probability must be between 0 and 1.")


@dataclass(frozen=True)
class Evidence:
    rule_id: str
    observation: str
    interpretation: str


@dataclass(frozen=True)
class InvestigationReport:
    summary: str
    evidence: list[Evidence]
    recommended_checks: list[str]
    model_outputs: ModelOutputs
    limitations: list[str]


def investigate(
    reading: SensorReading,
    model_outputs: ModelOutputs | None = None,
) -> InvestigationReport:
    """Apply published AI4I patterns and report evidence without asserting causes."""
    outputs = model_outputs or ModelOutputs()
    evidence: list[Evidence] = []
    recommended_checks: list[str] = []

    temperature_difference = reading.process_temperature_k - reading.air_temperature_k
    if (
        temperature_difference < HDF_TEMPERATURE_DIFFERENCE_K
        and reading.rotational_speed_rpm < HDF_MAX_ROTATIONAL_SPEED_RPM
    ):
        evidence.append(
            Evidence(
                rule_id="AI4I-HDF-pattern",
                observation=(
                    f"Process-air temperature difference is {temperature_difference:.2f} K "
                    f"and rotational speed is {reading.rotational_speed_rpm:.0f} rpm."
                ),
                interpretation=(
                    "These values meet the AI4I synthetic dataset's heat-dissipation "
                    "failure rule; they do not prove a real equipment fault."
                ),
            )
        )
        recommended_checks.append(
            "Review cooling and heat-dissipation readings, and verify both temperature sensors."
        )

    power_w = reading.torque_nm * reading.rotational_speed_rpm * (2 * math.pi / 60)
    if power_w < PWF_MIN_POWER_W or power_w > PWF_MAX_POWER_W:
        evidence.append(
            Evidence(
                rule_id="AI4I-PWF-pattern",
                observation=f"Torque and speed imply approximately {power_w:.0f} W of power.",
                interpretation=(
                    "This is outside the AI4I synthetic dataset's 3500-9000 W power range; "
                    "it is a pattern to inspect, not a diagnosis."
                ),
            )
        )
        recommended_checks.append(
            "Check motor load, drive readings, and torque and speed sensor calibration."
        )

    wear_torque = reading.tool_wear_min * reading.torque_nm
    overstrain_limit = OSF_TORQUE_WEAR_LIMITS[reading.product_type]
    if wear_torque > overstrain_limit:
        evidence.append(
            Evidence(
                rule_id="AI4I-OSF-pattern",
                observation=(
                    f"Tool-wear/torque product is {wear_torque:.0f} minNm; "
                    f"the AI4I {reading.product_type} limit is {overstrain_limit} minNm."
                ),
                interpretation=(
                    "This meets the AI4I synthetic dataset's overstrain rule; "
                    "it does not establish that equipment has failed."
                ),
            )
        )
        recommended_checks.append(
            "Inspect tool condition and review the maintenance and replacement history."
        )

    if TWF_TOOL_WEAR_MIN_MINUTES <= reading.tool_wear_min <= TWF_TOOL_WEAR_MAX_MINUTES:
        evidence.append(
            Evidence(
                rule_id="AI4I-TWF-wear-band",
                observation=(
                    f"Tool wear is {reading.tool_wear_min:.1f} minutes, within the "
                    "AI4I dataset's documented 200-240 minute tool-wear range."
                ),
                interpretation=(
                    "The wear value is in a range used by the synthetic dataset's "
                    "tool-wear failure logic; it does not confirm a tool failure."
                ),
            )
        )
        recommended_checks.append(
            "Check tool-wear measurements and the tool replacement schedule."
        )

    if outputs.anomaly_score is not None and outputs.anomaly_threshold is not None:
        if outputs.anomaly_score > outputs.anomaly_threshold:
            evidence.append(
                Evidence(
                    rule_id="autoencoder-threshold",
                    observation=(
                        f"Reconstruction error {outputs.anomaly_score:.6f} exceeds "
                        f"the validation threshold {outputs.anomaly_threshold:.6f}."
                    ),
                    interpretation=(
                        "The Autoencoder found an unusual pattern relative to its "
                        "normal training examples; an anomaly is not proof of failure."
                    ),
                )
            )
            recommended_checks.append(
                "Review the contributing sensor readings and compare them with "
                "machine-specific operating limits."
            )

    if outputs.failure_probability is not None:
        recommended_checks.append(
            "Review the classifier output alongside sensor evidence; it does not identify a cause."
        )

    if evidence:
        summary = (
            f"Found {len(evidence)} dataset-rule or model-threshold pattern(s) "
            "for human review; none confirms a failure."
        )
    else:
        summary = (
            "No configured dataset-rule pattern or above-threshold anomaly was observed. "
            "This does not establish that the machine is failure-free."
        )
        recommended_checks.append(
            "Compare readings with the equipment manufacturer's limits and verify sensor calibration."
        )

    return InvestigationReport(
        summary=summary,
        evidence=evidence,
        recommended_checks=recommended_checks,
        model_outputs=outputs,
        limitations=[
            "AI4I is synthetic; its rules may not transfer to real factory equipment.",
            "An anomaly is not proof of failure, and model outputs require human review.",
            "Classifier output is a model estimate, not a calibrated real-world failure risk.",
        ],
    )
