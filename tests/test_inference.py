from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.inference import ModelService
from backend.schemas import BookingFeatures

VALID_FEATURES = {
    "number_of_adults": 1,
    "number_of_children": 1,
    "number_of_weekend_nights": 2,
    "number_of_week_nights": 5,
    "type_of_meal": 0,
    "car_parking_space": 0,
    "room_type": 1,
    "lead_time": 224,
    "market_segment_type": 0,
    "repeated": 0,
    "previous_cancellations": 0,
    "previous_not_cancelled": 0,
    "average_price": 88,
    "special_requests": 0,
}


def test_real_model_loads_and_predicts() -> None:
    service = ModelService(Path("model_xgb.ubj"))
    service.load()
    result = service.predict(BookingFeatures(**VALID_FEATURES))
    assert result.predicted_outcome in {0, 1}
    assert result.cancellation_probability is not None
    assert 0 <= result.cancellation_probability <= 1
    assert len(service.sha256) == 64
    assert service.version.startswith("xgb-")


def test_feature_ranges_are_validated() -> None:
    with pytest.raises(ValidationError):
        BookingFeatures(**{**VALID_FEATURES, "lead_time": -1})


def test_unknown_features_are_rejected() -> None:
    with pytest.raises(ValidationError):
        BookingFeatures(**VALID_FEATURES, unsupported=1)
