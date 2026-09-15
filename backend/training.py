"""Validated, reproducible retraining; no blanket removal of rare booking categories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .schemas import BookingFeatures

MEALS = {"Meal Plan 1": 0, "Not Selected": 1, "Meal Plan 2": 2, "Meal Plan 3": 3}
MARKETS = {"Offline": 0, "Online": 1, "Corporate": 2, "Aviation": 3, "Complementary": 4}
FEATURE_MAP = dict(
    zip(
        BookingFeatures.model_fields,
        [
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
        ],
        strict=True,
    )
)
FEATURE_NAMES = list(FEATURE_MAP.values())


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    frame = frame.rename(columns=lambda name: name.strip()).copy()
    required = {*FEATURE_NAMES, "booking status", "Booking_ID"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    input_rows = len(frame)
    duplicates = int(frame.duplicated("Booking_ID").sum())
    frame = frame.drop_duplicates("Booking_ID")
    frame["type of meal"] = frame["type of meal"].map(MEALS)
    frame["market segment type"] = frame["market segment type"].map(MARKETS)
    frame["room type"] = pd.to_numeric(
        frame["room type"].str.extract(r"^Room_Type ([1-7])$")[0], errors="coerce"
    )
    target = frame["booking status"].map({"Not_Canceled": 0, "Canceled": 1})
    for name in FEATURE_NAMES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    accepted, rejected = [], []
    for index, row in frame.iterrows():
        try:
            if pd.isna(row["Booking_ID"]) or pd.isna(target.loc[index]):
                raise ValueError("Missing booking ID or invalid target")
            features = BookingFeatures(**{api: row[column] for api, column in FEATURE_MAP.items()})
            accepted.append((index, features.as_model_row()))
        except (ValueError, ValidationError):
            rejected.append(str(row["Booking_ID"]))
    if not accepted:
        raise ValueError("No valid booking rows")
    indices, rows = zip(*accepted, strict=True)
    x = pd.DataFrame(rows, columns=FEATURE_NAMES).reset_index(drop=True)
    y = target.loc[list(indices)].astype(int).reset_index(drop=True)
    if y.nunique() != 2 or y.value_counts().min() < 10:
        raise ValueError("Both classes require at least 10 valid rows")
    return (
        x,
        y,
        {
            "input_rows": input_rows,
            "accepted_rows": len(x),
            "duplicate_ids": duplicates,
            "rejected_rows": len(rejected),
            "rejected_booking_ids": rejected,
        },
    )


def metrics(model, x, y):
    probability = model.predict_proba(x)
    prediction = probability.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, probability[:, 1])),
        "log_loss": float(log_loss(y, probability)),
        "classification_report": classification_report(
            y, prediction, output_dict=True, zero_division=0
        ),
    }


def train(input_path: Path, output: Path, seed=42, estimators=150) -> dict:
    if estimators < 1:
        raise ValueError("estimators must be positive")
    x, y, quality = prepare_data(pd.read_csv(input_path))
    indices = np.arange(len(x))
    train_ids, other_ids = train_test_split(indices, test_size=0.3, stratify=y, random_state=seed)
    val_ids, test_ids = train_test_split(
        other_ids, test_size=0.5, stratify=y.iloc[other_ids], random_state=seed
    )
    baseline = DummyClassifier(strategy="prior").fit(x.iloc[train_ids], y.iloc[train_ids])
    config = {
        "n_estimators": estimators,
        "max_depth": 5,
        "learning_rate": 0.07,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": 2,
    }
    model = XGBClassifier(**config).fit(x.iloc[train_ids], y.iloc[train_ids])
    report = {
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "seed": seed,
        "quality": quality,
        "split_rows": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "configuration": config,
        "validation": {
            "baseline": metrics(baseline, x.iloc[val_ids], y.iloc[val_ids]),
            "xgboost": metrics(model, x.iloc[val_ids], y.iloc[val_ids]),
        },
        "test": metrics(model, x.iloc[test_ids], y.iloc[test_ids]),
    }
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "model_xgb.ubj"
    model.save_model(artifact)
    metadata = {
        "format": "xgboost_ubj",
        "feature_names": FEATURE_NAMES,
        "categorical_mappings": {
            "type of meal": MEALS,
            "market segment type": MARKETS,
            "room type": "numeric suffix 1..7",
        },
        "classes": {"0": "Not_Canceled", "1": "Canceled"},
        "trained_with_xgboost": xgboost.__version__,
        "source_sha256": report["source_sha256"],
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("cellula_hotel.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=150)
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.output, args.seed, args.estimators), indent=2))


if __name__ == "__main__":
    main()
