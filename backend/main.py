from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import io

import pandas as pd
import joblib

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from bson import ObjectId
from bson.binary import Binary

from backend.db import db
from backend.security import hash_password, verify_password


# ---------------------------
# Env + Paths
# ---------------------------
ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ROOT_ENV)

BASE_DIR = Path(__file__).resolve().parents[1]  # project root
MODEL_FILE = BASE_DIR / "model_xgb.pkl"
CSV_FILE = BASE_DIR / "cellula_hotel.csv"

SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is missing in .env")

FEATURE_ORDER = [
    "number of adults",
    "number of children",
    "number of weekend nights",
    "number of week nights",
    "type of meal",
    "car parking space",
    "room type",
    "lead time",
    "market segment type",
    "repeated",
    "P-C",
    "P-not-C",
    "average price",
    "special requests",
]

MODEL_CACHE = {"model": None, "features": None, "model_id": None}
STATS: dict[str, dict[str, float]] = {}


# ---------------------------
# Helpers
# ---------------------------
def ctx(request: Request, **kwargs):
    return {
        "request": request,
        "logged_in": bool(request.session.get("user_id")),
        "role": (request.session.get("role") or "").lower(),
        **kwargs,
    }


def require_login(request: Request) -> str | None:
    return request.session.get("user_id")


def require_admin(request: Request) -> bool:
    return (request.session.get("role") or "").lower() == "admin"


def oid_or_none(x: str):
    try:
        return ObjectId(x)
    except Exception:
        return None


def flash(request: Request, text: str, kind: str = "ok"):
    # kind: "ok" or "bad"
    request.session["flash"] = {"kind": kind, "text": text}


def pop_flash(request: Request):
    return request.session.pop("flash", None)


def compute_stats_from_csv() -> dict[str, dict[str, float]]:
    """
    'Typical range' = 5th–95th percentile, plus avg.
    """
    if not CSV_FILE.exists():
        return {}

    df = pd.read_csv(CSV_FILE)

    stats: dict[str, dict[str, float]] = {}
    for col in FEATURE_ORDER:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        low = float(s.quantile(0.05))
        high = float(s.quantile(0.95))
        avg = float(s.mean())

        stats[col] = {"low": round(low, 2), "high": round(high, 2), "avg": round(avg, 2)}

    return stats


async def ensure_model_in_mongo():
    """
    Requirement: model stored in MongoDB models collection.
    Seeds/repairs it using the raw bytes of model_xgb.pkl.
    """
    existing = await db.models.find_one({"name": "model_xgb"})
    if existing and existing.get("payload"):
        return

    if not MODEL_FILE.exists():
        raise RuntimeError(f"model_xgb.pkl not found at: {MODEL_FILE}")

    raw_bytes = MODEL_FILE.read_bytes()
    model_obj, model_features = joblib.load(MODEL_FILE)

    doc = {
        "name": "model_xgb",
        "format": "joblib",
        "created_at": datetime.utcnow(),
        "payload": Binary(raw_bytes),
        "features": list(model_features),
    }

    if existing:
        await db.models.update_one({"_id": existing["_id"]}, {"$set": doc})
    else:
        await db.models.insert_one(doc)


async def load_model_cache():
    doc = await db.models.find_one({"name": "model_xgb"})
    if not doc:
        raise RuntimeError("No model found in MongoDB models collection.")

    payload = doc.get("payload")
    if not payload:
        raise RuntimeError("Model doc exists but payload is missing.")

    model_obj, model_features = joblib.load(io.BytesIO(bytes(payload)))

    MODEL_CACHE["model"] = model_obj
    MODEL_CACHE["features"] = list(model_features)
    MODEL_CACHE["model_id"] = str(doc["_id"])


# ---------------------------
# App
# ---------------------------
app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    same_site="lax",
    https_only=False,  # True when deploying behind HTTPS
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.on_event("startup")
async def startup():
    global STATS
    await db.users.create_index("email", unique=True)
    await db.dataset.create_index("owner_user_id")
    await db.predictions.create_index("owner_user_id")
    await db.models.create_index("name", unique=True)

    await ensure_model_in_mongo()
    await load_model_cache()

    STATS = compute_stats_from_csv()


@app.get("/health")
async def health():
    cols = await db.list_collection_names()
    return {"ok": True, "db": db.name, "collections": cols}


# ---------------------------
# Auth
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    if require_login(request):
        return RedirectResponse("/predict", status_code=302)
    return templates.TemplateResponse("login.html", ctx(request, title="Login", error=None))


@app.post("/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    user = await db.users.find_one({"email": email.strip().lower()})
    if not user or not verify_password(password, user["password_hash"]):
        return templates.TemplateResponse(
            "login.html", ctx(request, title="Login", error="Invalid email or password")
        )

    request.session["user_id"] = str(user["_id"])
    request.session["role"] = (user.get("role", "owner") or "owner").lower()
    return RedirectResponse("/predict", status_code=302)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    if require_login(request):
        return RedirectResponse("/predict", status_code=302)
    return templates.TemplateResponse("register.html", ctx(request, title="Register", error=None))


@app.post("/register")
async def register(
    request: Request,
    hotel_name: str = Form(...),
    location: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    photo_url: str = Form(""),
):
    doc = {
        "email": email.strip().lower(),
        "password_hash": hash_password(password),
        "role": "owner",
        "hotel": {
            "hotel_name": hotel_name.strip(),
            "location": location.strip(),
            "photo_url": photo_url.strip(),
        },
        "created_at": datetime.utcnow(),
    }

    try:
        res = await db.users.insert_one(doc)
    except Exception:
        return templates.TemplateResponse(
            "register.html", ctx(request, title="Register", error="Email already exists")
        )

    # Auto-login after register
    request.session["user_id"] = str(res.inserted_id)
    request.session["role"] = "owner"
    return RedirectResponse("/predict", status_code=302)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=302)


# ---------------------------
# Admin dashboard (admin-only)
# ---------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    uid = require_login(request)
    if not uid:
        return RedirectResponse("/", status_code=302)
    if not require_admin(request):
        return RedirectResponse("/predict", status_code=302)

    total_users = await db.users.count_documents({})

    # count submissions per user (dataset)
    pipeline = [{"$group": {"_id": "$owner_user_id", "submissions": {"$sum": 1}}}]
    counts = {}
    async for d in db.dataset.aggregate(pipeline):
        counts[str(d["_id"])] = int(d["submissions"])

    users = []
    async for u in db.users.find({}, {"password_hash": 0}):
        uid_str = str(u["_id"])
        hotel_name = (u.get("hotel") or {}).get("hotel_name") or u.get("email")
        users.append({
            "id": uid_str,
            "name": hotel_name,
            "email": u.get("email"),
            "role": (u.get("role") or "owner").lower(),
            "submissions": counts.get(uid_str, 0),
        })

    f = pop_flash(request)

    return templates.TemplateResponse(
        "admin_dashboard.html",
        ctx(
            request,
            title="Admin Dashboard",
            total_users=total_users,
            users=users,
            flash=f,
            add_error=None,
        ),
    )


@app.post("/dashboard/add-user")
async def admin_add_user(
    request: Request,
    hotel_name: str = Form(...),
    location: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("owner"),
    photo_url: str = Form(""),
):
    uid = require_login(request)
    if not uid:
        return RedirectResponse("/", status_code=302)
    if not require_admin(request):
        return RedirectResponse("/predict", status_code=302)

    role = (role or "owner").lower()
    if role not in ("owner", "admin"):
        role = "owner"

    doc = {
        "email": email.strip().lower(),
        "password_hash": hash_password(password),
        "role": role,
        "hotel": {
            "hotel_name": hotel_name.strip(),
            "location": location.strip(),
            "photo_url": photo_url.strip(),
        },
        "created_at": datetime.utcnow(),
    }

    try:
        await db.users.insert_one(doc)
        flash(request, "User created successfully", "ok")
    except Exception:
        flash(request, "Email already exists (user not created)", "bad")

    return RedirectResponse("/dashboard", status_code=302)


@app.get("/dashboard/users/{user_id}", response_class=HTMLResponse)
async def admin_user_page(request: Request, user_id: str):
    uid = require_login(request)
    if not uid:
        return RedirectResponse("/", status_code=302)
    if not require_admin(request):
        return RedirectResponse("/predict", status_code=302)

    oid = oid_or_none(user_id)
    if not oid:
        return RedirectResponse("/dashboard", status_code=302)

    u = await db.users.find_one({"_id": oid}, {"password_hash": 0})
    if not u:
        return RedirectResponse("/dashboard", status_code=302)

    submissions = await db.dataset.count_documents({"owner_user_id": oid})

    view = {
        "id": str(u["_id"]),
        "email": u.get("email"),
        "role": (u.get("role") or "owner").lower(),
        "name": (u.get("hotel") or {}).get("hotel_name") or u.get("email"),
        "location": (u.get("hotel") or {}).get("location", ""),
        "submissions": submissions,
    }

    f = pop_flash(request)
    msg = f["text"] if f and f.get("kind") == "ok" else None
    err = f["text"] if f and f.get("kind") == "bad" else None

    return templates.TemplateResponse(
        "admin_user.html",
        ctx(request, title="Configure user", u=view, msg=msg, error=err),
    )


@app.post("/dashboard/users/{user_id}/role")
async def admin_set_role(request: Request, user_id: str, role: str = Form(...)):
    uid = require_login(request)
    if not uid:
        return RedirectResponse("/", status_code=302)
    if not require_admin(request):
        return RedirectResponse("/predict", status_code=302)

    oid = oid_or_none(user_id)
    if not oid:
        return RedirectResponse("/dashboard", status_code=302)

    role = (role or "owner").lower()
    if role not in ("owner", "admin"):
        role = "owner"

    # prevent demoting yourself (keeps you from losing dashboard)
    if str(oid) == uid and role != "admin":
        flash(request, "You cannot remove admin role from your own account.", "bad")
        return RedirectResponse(f"/dashboard/users/{user_id}", status_code=302)

    await db.users.update_one({"_id": oid}, {"$set": {"role": role}})
    flash(request, "Role updated.", "ok")
    return RedirectResponse(f"/dashboard/users/{user_id}", status_code=302)


@app.post("/dashboard/users/{user_id}/delete")
async def admin_delete_user(request: Request, user_id: str):
    uid = require_login(request)
    if not uid:
        return RedirectResponse("/", status_code=302)
    if not require_admin(request):
        return RedirectResponse("/predict", status_code=302)

    oid = oid_or_none(user_id)
    if not oid:
        return RedirectResponse("/dashboard", status_code=302)

    # prevent deleting yourself
    if str(oid) == uid:
        flash(request, "You cannot delete your own admin account.", "bad")
        return RedirectResponse("/dashboard", status_code=302)

    # cascade delete user data
    await db.dataset.delete_many({"owner_user_id": oid})
    await db.predictions.delete_many({"owner_user_id": oid})
    await db.users.delete_one({"_id": oid})

    flash(request, "User deleted (and their data removed).", "ok")
    return RedirectResponse("/dashboard", status_code=302)


# ---------------------------
# Prediction (main page)
# ---------------------------
@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    uid = require_login(request)
    if not uid:
        return RedirectResponse("/", status_code=302)

    return templates.TemplateResponse(
        "predict.html",
        ctx(
            request,
            title="Prediction",
            stats=STATS,
            prediction_text=None,
            is_correct=None,
        ),
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_submit(
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

    actual_outcome: str = Form(""),  # "", "0", "1"
):
    uid = require_login(request)
    if not uid:
        return RedirectResponse("/", status_code=302)

    user = await db.users.find_one({"_id": ObjectId(uid)}, {"password_hash": 0})

    features = {
        "number of adults": number_of_adults,
        "number of children": number_of_children,
        "number of weekend nights": number_of_weekend_nights,
        "number of week nights": number_of_week_nights,
        "type of meal": type_of_meal,
        "car parking space": car_parking_space,
        "room type": room_type,
        "lead time": lead_time,
        "market segment type": market_segment_type,
        "repeated": repeated,
        "P-C": p_c,
        "P-not-C": p_not_c,
        "average price": average_price,
        "special requests": special_requests,
    }

    model = MODEL_CACHE["model"]
    model_features = MODEL_CACHE["features"]
    model_id = MODEL_CACHE["model_id"]

    if model is None or model_features is None or model_id is None:
        await load_model_cache()
        model = MODEL_CACHE["model"]
        model_features = MODEL_CACHE["features"]
        model_id = MODEL_CACHE["model_id"]

    df = pd.DataFrame([features], columns=model_features)
    y = int(model.predict(df)[0])  # 0/1

    prediction_text = "Cancelled" if y == 1 else "Not Cancelled"

    actual = None
    is_correct = None
    if actual_outcome in ("0", "1"):
        actual = int(actual_outcome)
        is_correct = (actual == y)

    now = datetime.utcnow()

    # dataset: for data collection
    await db.dataset.insert_one({
        "owner_user_id": ObjectId(uid),
        "hotel_name": (user.get("hotel") or {}).get("hotel_name"),
        "location": (user.get("hotel") or {}).get("location"),
        "features": features,
        "actual": actual,
        "created_at": now,
    })

    # predictions: store prediction + correctness (if actual provided)
    await db.predictions.insert_one({
        "owner_user_id": ObjectId(uid),
        "model_id": ObjectId(model_id),
        "features": features,
        "prediction": y,
        "prediction_text": prediction_text,
        "actual": actual,
        "is_correct": is_correct,
        "created_at": now,
    })

    return templates.TemplateResponse(
        "predict.html",
        ctx(
            request,
            title="Prediction",
            stats=STATS,
            prediction_text=prediction_text,
            is_correct=is_correct,
        ),
    )
