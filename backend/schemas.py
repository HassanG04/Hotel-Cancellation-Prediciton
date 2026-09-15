from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BookingFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number_of_adults: int = Field(ge=0, le=4)
    number_of_children: int = Field(ge=0, le=10)
    number_of_weekend_nights: int = Field(ge=0, le=7)
    number_of_week_nights: int = Field(ge=0, le=17)
    type_of_meal: int = Field(ge=0, le=3)
    car_parking_space: int = Field(ge=0, le=1)
    room_type: int = Field(ge=1, le=7)
    lead_time: int = Field(ge=0, le=443)
    market_segment_type: int = Field(ge=0, le=4)
    repeated: int = Field(ge=0, le=1)
    previous_cancellations: int = Field(ge=0, le=13)
    previous_not_cancelled: int = Field(ge=0, le=58)
    average_price: float = Field(ge=0, le=540)
    special_requests: int = Field(ge=0, le=5)

    def as_model_row(self) -> dict[str, int | float]:
        return {
            "number of adults": self.number_of_adults,
            "number of children": self.number_of_children,
            "number of weekend nights": self.number_of_weekend_nights,
            "number of week nights": self.number_of_week_nights,
            "type of meal": self.type_of_meal,
            "car parking space": self.car_parking_space,
            "room type": self.room_type,
            "lead time": self.lead_time,
            "market segment type": self.market_segment_type,
            "repeated": self.repeated,
            "P-C": self.previous_cancellations,
            "P-not-C": self.previous_not_cancelled,
            "average price": self.average_price,
            "special requests": self.special_requests,
        }


class PredictionRequest(BaseModel):
    features: BookingFeatures
    actual_outcome: int | None = Field(default=None, ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction_id: int
    predicted_outcome: int
    prediction_label: str
    cancellation_probability: float | None
    is_correct: bool | None
    model_version: str
    latency_ms: float
