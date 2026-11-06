"""API routes for validating sensor data and analyzing a reading."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from backend.schemas.analysis import (
    AnalysisResponse,
    FailureRiskResponse,
    MachineListResponse,
    MaintenanceActionRequest,
    MaintenanceActionResponse,
    MaintenanceResponse,
    SensorDataAccepted,
    SensorHistoryResponse,
    SensorReadingRequest,
)
from backend.database.storage import Database
from backend.services.investigator import SensorReading
from backend.services.model_service import AnalysisService

router = APIRouter()


def get_analysis_service(request: Request) -> AnalysisService:
    service = getattr(request.app.state, "analysis_service", None)
    if not isinstance(service, AnalysisService):
        raise HTTPException(
            status_code=503,
            detail="Analysis models are not loaded. Check the saved model artifacts.",
        )
    return service


def get_database(request: Request) -> Database:
    database = getattr(request.app.state, "database", None)
    if not isinstance(database, Database):
        raise HTTPException(
            status_code=503,
            detail="SQLite storage is not initialized.",
        )
    return database


def to_sensor_reading(reading: SensorReadingRequest) -> SensorReading:
    return SensorReading(
        product_type=reading.product_type,
        air_temperature_k=reading.air_temperature_k,
        process_temperature_k=reading.process_temperature_k,
        rotational_speed_rpm=reading.rotational_speed_rpm,
        torque_nm=reading.torque_nm,
        tool_wear_min=reading.tool_wear_min,
    )


@router.post("/sensor-data", response_model=SensorDataAccepted)
def accept_sensor_data(
    reading: SensorReadingRequest,
    request: Request,
) -> SensorDataAccepted:
    reading_id = get_database(request).record_sensor_reading(
        reading.machine_id,
        to_sensor_reading(reading),
    )
    return SensorDataAccepted(
        status="accepted",
        machine_id=reading.machine_id,
        reading_id=reading_id,
        message="Sensor reading validated and saved to SQLite.",
        reading=reading,
    )


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_reading(
    reading: SensorReadingRequest,
    request: Request,
) -> AnalysisResponse:
    service = get_analysis_service(request)
    sensor_reading = to_sensor_reading(reading)
    result = service.analyze(sensor_reading)
    reading_id, analysis_id = get_database(request).record_analysis(
        reading.machine_id,
        sensor_reading,
        result,
    )
    return result.model_copy(
        update={
            "machine_id": reading.machine_id,
            "reading_id": reading_id,
            "analysis_id": analysis_id,
        }
    )


@router.get("/machines", response_model=MachineListResponse)
def list_machines(request: Request) -> MachineListResponse:
    return MachineListResponse(machines=get_database(request).list_machines())


@router.get(
    "/machines/{machine_id}/readings",
    response_model=SensorHistoryResponse,
)
def get_sensor_history(
    machine_id: str,
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> SensorHistoryResponse:
    readings = get_database(request).sensor_history(machine_id, limit)
    if readings is None:
        raise HTTPException(status_code=404, detail="Machine not found.")
    return SensorHistoryResponse(machine_id=machine_id, readings=readings)


@router.get(
    "/machines/{machine_id}/analysis/latest",
    response_model=AnalysisResponse | None,
)
def get_latest_analysis(machine_id: str, request: Request) -> AnalysisResponse | None:
    database = get_database(request)
    if not database.machine_exists(machine_id):
        raise HTTPException(status_code=404, detail="Machine not found.")
    result = database.latest_analysis(machine_id)
    if result is None:
        return None
    return AnalysisResponse(**result)


@router.get("/failure-risk/{machine_id}", response_model=FailureRiskResponse)
def get_latest_failure_risk(machine_id: str, request: Request) -> FailureRiskResponse:
    database = get_database(request)
    if not database.machine_exists(machine_id):
        raise HTTPException(status_code=404, detail="Machine not found.")
    risk = database.latest_risk(machine_id)
    if risk is None:
        raise HTTPException(status_code=404, detail="No analysis is available for this machine.")
    return FailureRiskResponse(
        **risk,
        note=(
            "Classifier output is a model estimate from synthetic data, not a "
            "calibrated real-world failure risk."
        ),
    )


@router.get("/maintenance/{machine_id}", response_model=MaintenanceResponse)
def get_maintenance(machine_id: str, request: Request) -> MaintenanceResponse:
    history = get_database(request).maintenance_history(machine_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Machine not found.")
    return MaintenanceResponse(**history)


@router.post(
    "/maintenance/{machine_id}/actions",
    response_model=MaintenanceActionResponse,
    status_code=201,
)
def record_maintenance_action(
    machine_id: str,
    action: MaintenanceActionRequest,
    request: Request,
) -> MaintenanceActionResponse:
    saved_action = get_database(request).record_maintenance_action(
        machine_id,
        action.action,
        action.reviewed_by,
        action.notes,
    )
    if saved_action is None:
        raise HTTPException(status_code=404, detail="Machine not found.")
    return MaintenanceActionResponse(**saved_action)
