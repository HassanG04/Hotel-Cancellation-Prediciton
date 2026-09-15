from pathlib import Path

import pandas as pd
import pytest

from backend.inference import ModelService
from backend.schemas import BookingFeatures
from backend.training import prepare_data, train
from tests.test_inference import VALID_FEATURES


def test_training_contract_categorical_codes_and_rejections():
    source = pd.read_csv("cellula_hotel.csv").iloc[:100].copy()
    source.loc[0, "room type"] = "unknown"
    source.loc[1, "lead time"] = -1
    source = pd.concat([source, source.iloc[[2]]], ignore_index=True)
    x, y, quality = prepare_data(source)
    assert quality["duplicate_ids"] == 1
    assert quality["rejected_rows"] == 2
    assert quality["accepted_rows"] == 98
    assert x["room type"].between(1, 7).all()
    assert set(y) == {0, 1}
    with pytest.raises(ValueError, match="Missing columns"):
        prepare_data(pd.DataFrame())


def test_training_exports_loadable_model(tmp_path):
    source = tmp_path / "input.csv"
    pd.read_csv("cellula_hotel.csv").iloc[:300].to_csv(source, index=False)
    report = train(source, tmp_path / "output", estimators=5)
    assert sum(report["split_rows"].values()) == report["quality"]["accepted_rows"]
    model = ModelService(Path(tmp_path / "output/model_xgb.ubj"))
    model.load()
    assert model.predict(BookingFeatures(**VALID_FEATURES)).predicted_outcome in {0, 1}
