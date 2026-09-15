from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from backend.db import SessionLocal
from backend.models import Hotel, Observation, User


def test_user_delete_cascades_to_hotel_and_observations() -> None:
    with SessionLocal() as session:
        user = User(
            email="owner@example.com",
            password_hash="hash",
            role="owner",
            hotel=Hotel(name="Example Hotel", location="Cairo"),
        )
        session.add(user)
        session.commit()
        session.add(Observation(hotel_id=user.hotel.id, features={"lead time": 10}))
        session.commit()
        session.delete(user)
        session.commit()
        assert session.scalar(select(func.count(Hotel.id))) == 0
        assert session.scalar(select(func.count(Observation.id))) == 0


def test_unique_email_and_role_constraints():
    with SessionLocal() as session:
        session.add(User(email="unique@example.com", password_hash="hash", role="owner"))
        session.commit()
        session.add(User(email="unique@example.com", password_hash="hash", role="owner"))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        session.add(User(email="invalid@example.com", password_hash="hash", role="superuser"))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


def test_foreign_key_and_binary_outcome_constraints():
    with SessionLocal() as session:
        session.add(Hotel(owner_id=999, name="Orphan", location="N/A"))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        user = User(
            email="valid@example.com",
            password_hash="hash",
            role="owner",
            hotel=Hotel(name="Demo", location="Cairo"),
        )
        session.add(user)
        session.commit()
        session.add(Observation(hotel_id=user.hotel.id, features={}, actual_outcome=2))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
