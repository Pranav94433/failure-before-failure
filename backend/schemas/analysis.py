"""Pydantic request and response schemas for machine analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SensorReadingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machine_id: str = Field(default="demo-machine-001", min_length=1, max_length=100)
    product_type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(gt=0, allow_inf_nan=False)
    process_temperature_k: float = Field(gt=0, allow_inf_nan=False)
    rotational_speed_rpm: float = Field(ge=0, allow_inf_nan=False)
    torque_nm: float = Field(ge=0, allow_inf_nan=False)
    tool_wear_min: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("machine_id")
    @classmethod
    def machine_id_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("machine_id cannot be blank.")
        return normalized


class SensorDataAccepted(BaseModel):
    status: Literal["accepted"]
    machine_id: str
    reading_id: int
    message: str
    reading: SensorReadingRequest


class ClassifierResult(BaseModel):
    predicted_label: Literal[0, 1]
    model_estimated_failure_probability: float = Field(ge=0, le=1)
    decision_threshold: float
    note: str


class AnomalyResult(BaseModel):
    reconstruction_error: float = Field(ge=0)
    threshold: float = Field(ge=0)
    flagged_as_anomaly: bool
    note: str


class EvidenceResponse(BaseModel):
    rule_id: str
    observation: str
    interpretation: str


class InvestigationResponse(BaseModel):
    summary: str
    evidence: list[EvidenceResponse]
    recommended_checks: list[str]
    limitations: list[str]


class AnalysisResponse(BaseModel):
    machine_id: str | None = None
    reading_id: int | None = None
    analysis_id: int | None = None
    classifier: ClassifierResult
    anomaly: AnomalyResult
    investigation: InvestigationResponse


class SensorReadingSummary(BaseModel):
    reading_id: int
    machine_id: str
    product_type: Literal["L", "M", "H"]
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    recorded_at: str


class SensorHistoryResponse(BaseModel):
    machine_id: str
    readings: list[SensorReadingSummary]


class MachineSummary(BaseModel):
    machine_id: str
    created_at: str
    last_seen_at: str
    reading_count: int


class MachineListResponse(BaseModel):
    machines: list[MachineSummary]


class FailureRiskResponse(BaseModel):
    machine_id: str
    analysis_id: int
    reading_id: int
    predicted_label: Literal[0, 1]
    model_estimated_failure_probability: float = Field(ge=0, le=1)
    anomaly_score: float = Field(ge=0)
    anomaly_threshold: float = Field(ge=0)
    anomaly_flag: bool
    created_at: str
    note: str


class MaintenanceActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1, max_length=1000)
    reviewed_by: str = Field(min_length=1, max_length=100)
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("action", "reviewed_by")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("This field cannot be blank.")
        return normalized

    @field_validator("notes")
    @classmethod
    def blank_notes_become_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class MaintenanceActionResponse(BaseModel):
    action_id: int
    machine_id: str
    action: str
    reviewed_by: str
    notes: str | None
    created_at: str


class MaintenanceResponse(BaseModel):
    machine_id: str
    latest_analysis_id: int | None
    suggested_checks: list[str]
    actions: list[MaintenanceActionResponse]
