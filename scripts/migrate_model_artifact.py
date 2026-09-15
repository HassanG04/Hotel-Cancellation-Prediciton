from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import xgboost


def main() -> None:
    source = Path("model_xgb.pkl")
    destination = Path("model_xgb.ubj")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model, feature_names = joblib.load(source)
    model.save_model(destination)
    metadata = {
        "format": "xgboost_ubj",
        "source_artifact": source.name,
        "feature_names": list(feature_names),
        "converted_with_xgboost": xgboost.__version__,
    }
    Path("model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {destination} and model_metadata.json")


if __name__ == "__main__":
    main()
