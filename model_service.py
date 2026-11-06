"""Load trained artifacts and run model and investigator inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from backend.schemas.analysis import (
    AnalysisResponse,
    AnomalyResult,
    ClassifierResult,
    EvidenceResponse,
    InvestigationResponse,
)
from backend.services.anomaly_model import Autoencoder
from backend.services.investigator import ModelOutputs, SensorReading, investigate
from models.training.prepare_data import FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVED_MODELS_DIR = PROJECT_ROOT / "models" / "saved"
CLASSIFIER_PATH = SAVED_MODELS_DIR / "failure_classifier.joblib"
AUTOENCODER_PATH = SAVED_MODELS_DIR / "anomaly_autoencoder.pt"
ANOMALY_METADATA_PATH = SAVED_MODELS_DIR / "anomaly_preprocessor.joblib"


class AnalysisService:
    """Holds loaded model artifacts and analyzes validated sensor readings."""

    def __init__(
        self,
        classifier: Pipeline,
        preprocessor: ColumnTransformer,
        autoencoder: Autoencoder,
        anomaly_threshold: float,
    ) -> None:
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.autoencoder = autoencoder.eval()
        self.anomaly_threshold = anomaly_threshold

    @classmethod
    def from_saved_models(cls) -> AnalysisService:
        classifier_artifact = joblib.load(CLASSIFIER_PATH)
        anomaly_artifact = joblib.load(ANOMALY_METADATA_PATH)
        if not isinstance(classifier_artifact, dict) or not isinstance(anomaly_artifact, dict):
            raise ValueError("Saved model artifacts must contain dictionaries.")

        required_classifier_keys = {"model", "features", "target"}
        required_anomaly_keys = {"preprocessor", "features", "threshold"}
        if not required_classifier_keys.issubset(classifier_artifact):
            raise ValueError("Classifier artifact is missing required metadata.")
        if not required_anomaly_keys.issubset(anomaly_artifact):
            raise ValueError("Autoencoder artifact is missing required metadata.")
        if classifier_artifact["features"] != FEATURES or anomaly_artifact["features"] != FEATURES:
            raise ValueError("Saved models do not use the expected input features.")

        classifier = classifier_artifact["model"]
        if not isinstance(classifier, Pipeline):
            raise ValueError("Saved classifier is not a scikit-learn Pipeline.")
        if not hasattr(classifier, "predict_proba"):
            raise ValueError("Saved classifier does not support probability estimates.")

        preprocessor = anomaly_artifact["preprocessor"]
        if not isinstance(preprocessor, ColumnTransformer):
            raise ValueError("Saved Autoencoder preprocessor is not a ColumnTransformer.")

        anomaly_threshold = float(anomaly_artifact["threshold"])
        if not np.isfinite(anomaly_threshold) or anomaly_threshold < 0:
            raise ValueError("Saved anomaly threshold must be a finite non-negative value.")

        state = torch.load(AUTOENCODER_PATH, map_location="cpu", weights_only=True)
        if "encoder.0.weight" not in state:
            raise ValueError("Saved Autoencoder weights are missing the input layer.")
        autoencoder = Autoencoder(int(state["encoder.0.weight"].shape[1]))
        autoencoder.load_state_dict(state)

        return cls(
            classifier=classifier,
            preprocessor=preprocessor,
            autoencoder=autoencoder,
            anomaly_threshold=anomaly_threshold,
        )

    def analyze(self, reading: SensorReading) -> AnalysisResponse:
        feature_frame = pd.DataFrame(
            [
                {
                    "Type": reading.product_type,
                    "Air temperature [K]": reading.air_temperature_k,
                    "Process temperature [K]": reading.process_temperature_k,
                    "Rotational speed [rpm]": reading.rotational_speed_rpm,
                    "Torque [Nm]": reading.torque_nm,
                    "Tool wear [min]": reading.tool_wear_min,
                }
            ],
            columns=FEATURES,
        )
        predicted_label = int(self.classifier.predict(feature_frame)[0])
        classes = list(self.classifier.classes_)
        if 1 not in classes:
            raise ValueError("Saved classifier does not include the failure label 1.")
        failure_probability = float(
            self.classifier.predict_proba(feature_frame)[0][classes.index(1)]
        )

        transformed = self.preprocessor.transform(feature_frame)
        to_array = getattr(transformed, "toarray", None)
        if callable(to_array):
            transformed = to_array()
        values = torch.from_numpy(np.asarray(transformed, dtype=np.float32))
        with torch.inference_mode():
            reconstruction = self.autoencoder(values)
            anomaly_score = float(torch.mean((values - reconstruction) ** 2, dim=1)[0])
        anomaly_detected = anomaly_score > self.anomaly_threshold

        model_outputs = ModelOutputs(
            anomaly_score=anomaly_score,
            anomaly_threshold=self.anomaly_threshold,
            failure_probability=failure_probability,
        )
        report = investigate(reading, model_outputs)

        return AnalysisResponse(
            classifier=ClassifierResult(
                predicted_label=predicted_label,
                model_estimated_failure_probability=failure_probability,
                decision_threshold=0.5,
                note=(
                    "This is a model estimate from synthetic training data, not a "
                    "calibrated real-world failure risk."
                ),
            ),
            anomaly=AnomalyResult(
                reconstruction_error=anomaly_score,
                threshold=self.anomaly_threshold,
                flagged_as_anomaly=anomaly_detected,
                note="An anomaly is an unusual pattern, not proof of failure.",
            ),
            investigation=InvestigationResponse(
                summary=report.summary,
                evidence=[
                    EvidenceResponse(
                        rule_id=item.rule_id,
                        observation=item.observation,
                        interpretation=item.interpretation,
                    )
                    for item in report.evidence
                ],
                recommended_checks=report.recommended_checks,
                limitations=report.limitations,
            ),
        )
