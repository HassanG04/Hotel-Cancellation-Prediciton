from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('owner', 'admin')", name="ck_users_role"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="owner")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    hotel: Mapped[Hotel | None] = relationship(
        back_populates="owner", cascade="all, delete-orphan", passive_deletes=True
    )


class Hotel(Base):
    __tablename__ = "hotels"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )
    name: Mapped[str] = mapped_column(String(160))
    location: Mapped[str] = mapped_column(String(160))
    photo_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    owner: Mapped[User] = relationship(back_populates="hotel")
    observations: Mapped[list[Observation]] = relationship(
        back_populates="hotel", cascade="all, delete-orphan", passive_deletes=True
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (Index("ix_model_versions_name_active", "name", "is_active"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(64), unique=True)
    artifact_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    artifact_path: Mapped[str] = mapped_column(String(500))
    feature_schema: Mapped[list[str]] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    predictions: Mapped[list[Prediction]] = relationship(back_populates="model_version")


class Observation(Base):
    __tablename__ = "observations"
    __table_args__ = (
        CheckConstraint(
            "actual_outcome IN (0, 1) OR actual_outcome IS NULL", name="ck_actual_binary"
        ),
        Index("ix_observations_hotel_created", "hotel_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    hotel_id: Mapped[int] = mapped_column(ForeignKey("hotels.id", ondelete="CASCADE"))
    features: Mapped[dict[str, Any]] = mapped_column(JSON)
    actual_outcome: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    hotel: Mapped[Hotel] = relationship(back_populates="observations")
    prediction: Mapped[Prediction | None] = relationship(
        back_populates="observation", cascade="all, delete-orphan", passive_deletes=True
    )


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("predicted_outcome IN (0, 1)", name="ck_prediction_binary"),
        Index("ix_predictions_model_created", "model_version_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    observation_id: Mapped[int] = mapped_column(
        ForeignKey("observations.id", ondelete="CASCADE"), unique=True
    )
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    predicted_outcome: Mapped[int] = mapped_column(Integer)
    cancellation_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    observation: Mapped[Observation] = relationship(back_populates="prediction")
    model_version: Mapped[ModelVersion] = relationship(back_populates="predictions")
