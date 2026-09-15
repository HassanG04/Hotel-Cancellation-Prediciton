from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from backend.config import BASE_DIR
from backend.db import SessionLocal, get_session, settings
from backend.inference import ModelService
from backend.models import Hotel, ModelVersion, Observation, Prediction, User
from backend.schemas import BookingFeatures, PredictionRequest, PredictionResponse
from backend.security import hash_password, verify_password

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("hotel_api")
CSV_FILE = BASE_DIR / "cellula_hotel.csv"
model_service = ModelService(settings.model_path)
model_version_id: int | None = None


def compute_stats_from_csv() -> dict[str, dict[str, float]]:
    if not CSV_FILE.exists():
        return {}
    frame = pd.read_csv(CSV_FILE)
    frame.columns = frame.columns.str.strip()
    stats: dict[str, dict[str, float]] = {}
    for column in BookingFeatures.model_fields:
        model_name = column.replace("_", " ")
        source_name = {
            "previous cancellations": "P-C",
            "previous not cancelled": "P-not-C",
        }.get(model_name, model_name)
        if source_name not in frame:
            continue
        values = pd.to_numeric(frame[source_name], errors="coerce").dropna()
        if values.empty:
            continue
        stats[source_name] = {
            "low": round(float(values.quantile(0.05)), 2),
            "high": round(float(values.quantile(0.95)), 2),
            "avg": round(float(values.mean()), 2),
        }
    return stats


STATS = compute_stats_from_csv()


def ensure_model_version(session: Session) -> ModelVersion:
    global model_version_id
    existing = session.scalar(
        select(ModelVersion).where(ModelVersion.artifact_sha256 == model_service.sha256)
    )
    if existing is None:
        session.execute(update(ModelVersion).values(is_active=False))
        existing = ModelVersion(
            name="hotel-cancellation-xgboost",
            version=model_service.version,
            artifact_sha256=model_service.sha256,
            artifact_path=str(model_service.artifact_path),
            feature_schema=model_service.feature_names,
            is_active=True,
        )
        session.add(existing)
    else:
        existing.is_active = True
    session.commit()
    session.refresh(existing)
    model_version_id = existing.id
    return existing


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model_service.load()
    with SessionLocal() as session:
        ensure_model_version(session)
    LOGGER.info(
        "model_loaded version=%s sha256=%s features=%d",
        model_service.version,
        model_service.sha256,
        len(model_service.feature_names),
    )
    yield


app = FastAPI(
    title="Hotel Cancellation Prediction API",
    version="1.0.0",
    description="Validated XGBoost inference with PostgreSQL prediction lineage.",
    lifespan=lifespan,
)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    same_site="lax",
    https_only=settings.cookie_secure,
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.middleware("http")
async def request_observability(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception("request_failed method=%s path=%s", request.method, request.url.path)
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    LOGGER.info(
        "request_complete method=%s path=%s status=%s latency_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    return response


def template(request: Request, name: str, **context):
    return templates.TemplateResponse(
        request=request,
        name=name,
        context={
            "logged_in": bool(request.session.get("user_id")),
            "role": (request.session.get("role") or "").lower(),
            **context,
        },
    )


def logged_in_user_id(request: Request) -> int | None:
    try:
        return int(request.session.get("user_id"))
    except (TypeError, ValueError):
        return None


def session_user(request: Request, session: Session) -> User | None:
    """Resolve current account state; signed cookie roles are not authorization evidence."""
    user_id = logged_in_user_id(request)
    if user_id is None:
        return None
    user = session.get(User, user_id)
    if user is None:
        request.session.clear()
        return None
    request.session["role"] = user.role
    return user


def is_admin(request: Request, session: Session) -> bool:
    user = session_user(request, session)
    return user is not None and user.role == "admin"


def flash(request: Request, message: str, kind: str = "ok") -> None:
    request.session["flash"] = {"kind": kind, "text": message}


def pop_flash(request: Request):
    return request.session.pop("flash", None)


def create_user(
    session: Session,
    *,
    email: str,
    password: str,
    role: str,
    hotel_name: str,
    location: str,
    photo_url: str,
) -> User:
    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters")
    normalized_role = role.lower() if role.lower() in {"owner", "admin"} else "owner"
    user = User(
        email=email.strip().lower(),
        password_hash=hash_password(password),
        role=normalized_role,
        hotel=Hotel(
            name=hotel_name.strip(),
            location=location.strip(),
            photo_url=photo_url.strip() or None,
        ),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def record_prediction(
    session: Session,
    *,
    hotel: Hotel,
    features: BookingFeatures,
    actual_outcome: int | None,
) -> PredictionResponse:
    if model_version_id is None:
        raise RuntimeError("model registry is not initialized")
    result = model_service.predict(features)
    observation = Observation(
        hotel_id=hotel.id,
        features=features.as_model_row(),
        actual_outcome=actual_outcome,
    )
    prediction = Prediction(
        observation=observation,
        model_version_id=model_version_id,
        predicted_outcome=result.predicted_outcome,
        cancellation_probability=result.cancellation_probability,
        is_correct=(
            result.predicted_outcome == actual_outcome if actual_outcome is not None else None
        ),
        latency_ms=result.latency_ms,
    )
    session.add(prediction)
    session.commit()
    session.refresh(prediction)
    LOGGER.info(
        "prediction_complete prediction_id=%s model=%s outcome=%s latency_ms=%s",
        prediction.id,
        model_service.version,
        prediction.predicted_outcome,
        prediction.latency_ms,
    )
    return PredictionResponse(
        prediction_id=prediction.id,
        predicted_outcome=prediction.predicted_outcome,
        prediction_label="Cancelled" if prediction.predicted_outcome else "Not Cancelled",
        cancellation_probability=prediction.cancellation_probability,
        is_correct=prediction.is_correct,
        model_version=model_service.version,
        latency_ms=prediction.latency_ms,
    )


@app.get("/health")
def health(session: Session = Depends(get_session)):
    session.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "database": "reachable",
        "model_version": model_service.version,
        "feature_count": len(model_service.feature_names),
    }


@app.get("/", response_class=HTMLResponse)
def login_page(request: Request, session: Session = Depends(get_session)):
    if session_user(request, session) is not None:
        return RedirectResponse("/predict", status_code=status.HTTP_302_FOUND)
    return template(request, "login.html", title="Login", error=None)


@app.post("/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session),
):
    user = session.scalar(select(User).where(User.email == email.strip().lower()))
    if user is None or not verify_password(password, user.password_hash):
        return template(request, "login.html", title="Login", error="Invalid email or password")
    request.session.clear()
    request.session["user_id"] = user.id
    request.session["role"] = user.role
    return RedirectResponse("/predict", status_code=status.HTTP_302_FOUND)


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return template(request, "register.html", title="Register", error=None)


@app.post("/register")
def register(
    request: Request,
    hotel_name: str = Form(...),
    location: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    photo_url: str = Form(""),
    session: Session = Depends(get_session),
):
    try:
        user = create_user(
            session,
            email=email,
            password=password,
            role="owner",
            hotel_name=hotel_name,
            location=location,
            photo_url=photo_url,
        )
    except (IntegrityError, ValueError) as exc:
        session.rollback()
        message = "Email already exists" if isinstance(exc, IntegrityError) else str(exc)
        return template(request, "register.html", title="Register", error=message)
    request.session.clear()
    request.session["user_id"] = user.id
    request.session["role"] = user.role
    return RedirectResponse("/predict", status_code=status.HTTP_302_FOUND)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=status.HTTP_302_FOUND)


@app.get("/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request, session: Session = Depends(get_session)):
    user = session_user(request, session)
    if user is None:
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)
    if user.role != "admin":
        return RedirectResponse("/predict", status_code=status.HTTP_302_FOUND)
    rows = session.execute(
        select(User, Hotel, func.count(Observation.id))
        .outerjoin(Hotel, Hotel.owner_id == User.id)
        .outerjoin(Observation, Observation.hotel_id == Hotel.id)
        .group_by(User.id, Hotel.id)
        .order_by(User.created_at)
    ).all()
    users = [
        {
            "id": user.id,
            "name": hotel.name if hotel else user.email,
            "email": user.email,
            "role": user.role,
            "submissions": count,
        }
        for user, hotel, count in rows
    ]
    return template(
        request,
        "admin_dashboard.html",
        title="Admin Dashboard",
        total_users=len(users),
        users=users,
        flash=pop_flash(request),
        add_error=None,
    )


@app.post("/dashboard/add-user")
def admin_add_user(
    request: Request,
    hotel_name: str = Form(...),
    location: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("owner"),
    photo_url: str = Form(""),
    session: Session = Depends(get_session),
):
    if not logged_in_user_id(request) or not is_admin(request, session):
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)
    try:
        create_user(
            session,
            email=email,
            password=password,
            role=role,
            hotel_name=hotel_name,
            location=location,
            photo_url=photo_url,
        )
        flash(request, "User created successfully")
    except (IntegrityError, ValueError) as exc:
        session.rollback()
        flash(
            request, "Email already exists" if isinstance(exc, IntegrityError) else str(exc), "bad"
        )
    return RedirectResponse("/dashboard", status_code=status.HTTP_302_FOUND)


@app.get("/dashboard/users/{user_id}", response_class=HTMLResponse)
def admin_user_page(request: Request, user_id: int, session: Session = Depends(get_session)):
    if not logged_in_user_id(request) or not is_admin(request, session):
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)
    user = session.get(User, user_id)
    if user is None:
        return RedirectResponse("/dashboard", status_code=status.HTTP_302_FOUND)
    count = session.scalar(
        select(func.count(Observation.id)).where(Observation.hotel_id == user.hotel.id)
    )
    message = pop_flash(request)
    return template(
        request,
        "admin_user.html",
        title="Configure user",
        u={
            "id": user.id,
            "email": user.email,
            "role": user.role,
            "name": user.hotel.name,
            "location": user.hotel.location,
            "submissions": count,
        },
        msg=message["text"] if message and message["kind"] == "ok" else None,
        error=message["text"] if message and message["kind"] == "bad" else None,
    )


@app.post("/dashboard/users/{user_id}/role")
def admin_set_role(
    request: Request, user_id: int, role: str = Form(...), session: Session = Depends(get_session)
):
    current_id = logged_in_user_id(request)
    if current_id is None or not is_admin(request, session):
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)
    user = session.get(User, user_id)
    if user is None:
        return RedirectResponse("/dashboard", status_code=status.HTTP_302_FOUND)
    if user_id == current_id and role != "admin":
        flash(request, "You cannot remove your own admin role.", "bad")
    else:
        user.role = role if role in {"owner", "admin"} else "owner"
        session.commit()
        flash(request, "Role updated.")
    return RedirectResponse(f"/dashboard/users/{user_id}", status_code=status.HTTP_302_FOUND)


@app.post("/dashboard/users/{user_id}/delete")
def admin_delete_user(request: Request, user_id: int, session: Session = Depends(get_session)):
    current_id = logged_in_user_id(request)
    if current_id is None or not is_admin(request, session):
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)
    if user_id == current_id:
        flash(request, "You cannot delete your own admin account.", "bad")
        return RedirectResponse("/dashboard", status_code=status.HTTP_302_FOUND)
    user = session.get(User, user_id)
    if user is not None:
        session.delete(user)
        session.commit()
        flash(request, "User and related records deleted.")
    return RedirectResponse("/dashboard", status_code=status.HTTP_302_FOUND)


@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request, session: Session = Depends(get_session)):
    if session_user(request, session) is None:
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)
    return template(
        request,
        "predict.html",
        title="Prediction",
        stats=STATS,
        prediction_text=None,
        is_correct=None,
        prediction_probability=None,
        input_error=None,
    )


def booking_from_form(**values) -> BookingFeatures:
    return BookingFeatures(
        number_of_adults=values["number_of_adults"],
        number_of_children=values["number_of_children"],
        number_of_weekend_nights=values["number_of_weekend_nights"],
        number_of_week_nights=values["number_of_week_nights"],
        type_of_meal=values["type_of_meal"],
        car_parking_space=values["car_parking_space"],
        room_type=values["room_type"],
        lead_time=values["lead_time"],
        market_segment_type=values["market_segment_type"],
        repeated=values["repeated"],
        previous_cancellations=values["p_c"],
        previous_not_cancelled=values["p_not_c"],
        average_price=values["average_price"],
        special_requests=values["special_requests"],
    )


@app.post("/predict", response_class=HTMLResponse)
def predict_submit(
    request: Request,
    number_of_adults: int = Form(...),
    number_of_children: int = Form(...),
    number_of_weekend_nights: int = Form(...),
    number_of_week_nights: int = Form(...),
    type_of_meal: int = Form(...),
    car_parking_space: int = Form(...),
    room_type: int = Form(...),
    lead_time: int = Form(...),
    market_segment_type: int = Form(...),
    repeated: int = Form(...),
    p_c: int = Form(...),
    p_not_c: int = Form(...),
    average_price: float = Form(...),
    special_requests: int = Form(...),
    actual_outcome: str = Form(""),
    session: Session = Depends(get_session),
):
    user = session_user(request, session)
    if user is None:
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)
    try:
        features = booking_from_form(**locals())
    except ValidationError as exc:
        return template(
            request,
            "predict.html",
            title="Prediction",
            stats=STATS,
            prediction_text=None,
            is_correct=None,
            prediction_probability=None,
            input_error=json.dumps(exc.errors(include_url=False)),
        )
    if user.hotel is None:
        raise HTTPException(status_code=401, detail="login required")
    response = record_prediction(
        session,
        hotel=user.hotel,
        features=features,
        actual_outcome=int(actual_outcome) if actual_outcome in {"0", "1"} else None,
    )
    return template(
        request,
        "predict.html",
        title="Prediction",
        stats=STATS,
        prediction_text=response.prediction_label,
        is_correct=response.is_correct,
        prediction_probability=(
            round(response.cancellation_probability * 100, 1)
            if response.cancellation_probability is not None
            else None
        ),
        input_error=None,
    )


@app.post("/api/v1/predictions", response_model=PredictionResponse)
def predict_api(
    payload: PredictionRequest,
    request: Request,
    session: Session = Depends(get_session),
):
    user = session_user(request, session)
    if user is None or user.hotel is None:
        raise HTTPException(status_code=401, detail="login required")
    return record_prediction(
        session,
        hotel=user.hotel,
        features=payload.features,
        actual_outcome=payload.actual_outcome,
    )
