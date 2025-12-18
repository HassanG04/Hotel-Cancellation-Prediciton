import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from pymongo import MongoClient
from passlib.context import CryptContext
from backend.security import hash_password
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

def main():
    if len(sys.argv) != 3:
        print("Usage: python backend/create_admin.py <email> <password>")
        sys.exit(1)

    email = sys.argv[1].strip().lower()
    password = sys.argv[2]

    load_dotenv()  # loads advancedb_project/.env when you run from project root
    uri = os.environ["MONGO_URI"]
    db_name = os.environ["MONGO_DB"]

    client = MongoClient(uri)
    db = client[db_name]

    doc = {
        "email": email,
        "password_hash": hash_password(password),
        "role": "admin",
        "created_at": datetime.utcnow(),
        "hotel": None,
    }

    try:
        db.users.insert_one(doc)
        print(f"Admin created: {email}")
    except Exception as e:
        print("Failed:", e)
        sys.exit(2)

if __name__ == "__main__":
    main()
