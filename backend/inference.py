from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .schemas import BookingFeatures


@dataclass(frozen=True)
class InferenceResult:
    predicted_outcome: int
    cancellation_probability: float | None
    latency_ms: float


class ModelService:
    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self.model = None
        self.feature_names: list[str] = []
        self.sha256 = ""

    @property
    def version(self) -> str:
        if not self.sha256:
            raise RuntimeError("model is not loaded")
        return f"xgb-{self.sha256[:12]}"

    def load(self) -> None:
        if not self.artifact_path.is_file():
            raise RuntimeError(f"model artifact not found: {self.artifact_path}")
        raw = self.artifact_path.read_bytes()
        if self.artifact_path.suffix.lower() in {".ubj", ".json"}:
            from xgboost import XGBClassifier

            metadata_path = self.artifact_path.with_name("model_metadata.json")
            if not metadata_path.is_file():
                raise RuntimeError(f"model metadata not found: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model = XGBClassifier()
            model.load_model(self.artifact_path)
            features = metadata["feature_names"]
        else:
            model, features = joblib.load(self.artifact_path)
        if not hasattr(model, "predict"):
            raise RuntimeError("model artifact does not provide predict()")
        self.model = model
        self.feature_names = list(features)
        self.sha256 = hashlib.sha256(raw).hexdigest()

    def predict(self, features: BookingFeatures) -> InferenceResult:
        if self.model is None:
            raise RuntimeError("model is not loaded")
        row = features.as_model_row()
        missing = sorted(set(self.feature_names) - set(row))
        if missing:
            raise RuntimeError(f"model feature schema mismatch: {missing}")
        frame = pd.DataFrame([row], columns=self.feature_names)
        started = time.perf_counter()
        predicted = int(self.model.predict(frame)[0])
        probability = None
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(frame)[0][1])
        return InferenceResult(
            predicted_outcome=predicted,
            cancellation_probability=probability,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
