"""Replayed login cookies must not preserve revoked database privileges."""

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.db import SessionLocal
from backend.main import app, create_user
from backend.models import User
from tests.test_inference import VALID_FEATURES


def seed_accounts():
    with SessionLocal() as session:
        admin = create_user(
            session,
            email="admin@example.com",
            password="password123",
            role="admin",
            hotel_name="Admin Hotel",
            location="Cairo",
            photo_url="",
        )
        owner = create_user(
            session,
            email="owner@example.com",
            password="password123",
            role="owner",
            hotel_name="Owner Hotel",
            location="Cairo",
            photo_url="",
        )
        return admin.id, owner.id


def login_cookie(client):
    response = client.post(
        "/login",
        data={"email": "admin@example.com", "password": "password123"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert client.get("/dashboard").status_code == 200
    return client.cookies.get("session")


def protected_requests(owner_id):
    return [
        ("GET", "/dashboard", {}),
        ("GET", f"/dashboard/users/{owner_id}", {}),
        (
            "POST",
            "/dashboard/add-user",
            {
                "email": "unexpected@example.com",
                "password": "password123",
                "hotel_name": "Unexpected Hotel",
                "location": "Cairo",
                "role": "admin",
            },
        ),
        ("POST", f"/dashboard/users/{owner_id}/role", {"role": "admin"}),
        ("POST", f"/dashboard/users/{owner_id}/delete", {}),
    ]


def replay(client, cookie, method, path, data=None):
    client.cookies.clear()
    client.cookies.set("session", cookie)
    return client.request(method, path, data=data, follow_redirects=False)


def test_demoted_admin_cookie_cannot_read_or_mutate_admin_resources():
    with TestClient(app) as client:
        admin_id, owner_id = seed_accounts()
        cookie = login_cookie(client)
        with SessionLocal() as session:
            session.get(User, admin_id).role = "owner"
            session.commit()
        for method, path, data in protected_requests(owner_id):
            response = replay(client, cookie, method, path, data)
            assert response.status_code == 302, (method, path)
            assert response.headers["location"] == ("/predict" if path == "/dashboard" else "/")
        with SessionLocal() as session:
            assert session.scalar(select(func.count(User.id))) == 2
            assert session.get(User, owner_id).role == "owner"
        client.cookies.clear()
        client.cookies.set("session", cookie)
        assert (
            client.post("/api/v1/predictions", json={"features": VALID_FEATURES}).status_code == 200
        )


def test_deleted_admin_cookie_is_anonymous_and_cannot_mutate_accounts():
    with TestClient(app) as client:
        admin_id, owner_id = seed_accounts()
        cookie = login_cookie(client)
        with SessionLocal() as session:
            session.delete(session.get(User, admin_id))
            session.commit()
        for method, path, data in protected_requests(owner_id):
            response = replay(client, cookie, method, path, data)
            assert response.status_code == 302, (method, path)
            assert response.headers["location"] == "/"
        assert replay(client, cookie, "GET", "/predict").headers["location"] == "/"
        assert replay(client, cookie, "GET", "/").status_code == 200
        client.cookies.clear()
        client.cookies.set("session", cookie)
        assert (
            client.post("/api/v1/predictions", json={"features": VALID_FEATURES}).status_code == 401
        )
        with SessionLocal() as session:
            assert session.scalar(select(func.count(User.id))) == 1
            assert session.get(User, owner_id).role == "owner"
