import os
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Always load .env from the PROJECT ROOT (advancedb_project/.env)
ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ROOT_ENV)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB  = os.getenv("MONGO_DB")

if not MONGO_URI or not MONGO_DB:
    raise RuntimeError(f"Missing env vars. MONGO_URI={MONGO_URI!r}, MONGO_DB={MONGO_DB!r}. Loaded from {ROOT_ENV}")

client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB]
