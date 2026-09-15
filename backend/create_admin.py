from __future__ import annotations

import argparse
import getpass

from sqlalchemy import select

from backend.db import SessionLocal
from backend.main import create_user
from backend.models import User


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the first hotel application administrator."
    )
    parser.add_argument("--email", required=True)
    parser.add_argument("--hotel-name", default="Administration")
    parser.add_argument("--location", default="N/A")
    args = parser.parse_args()
    password = getpass.getpass("Admin password (minimum 8 characters): ")

    with SessionLocal() as session:
        if session.scalar(select(User).where(User.email == args.email.strip().lower())):
            raise SystemExit("A user with that email already exists")
        create_user(
            session,
            email=args.email,
            password=password,
            role="admin",
            hotel_name=args.hotel_name,
            location=args.location,
            photo_url="",
        )
    print(f"Admin created: {args.email.strip().lower()}")


if __name__ == "__main__":
    main()
