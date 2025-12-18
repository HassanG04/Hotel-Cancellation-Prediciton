from django.shortcuts import render
from django.conf import settings
from pathlib import Path
import pandas as pd
import joblib

MODEL_PATH = Path(settings.BASE_DIR) / "model_xgb.pkl"
model_xgb, model_features = joblib.load(MODEL_PATH)

def home(request):
    prediction = None

    if request.method == "POST":
        data = request.POST.copy()
        data.pop("csrfmiddlewaretoken", None)

        # Flask did: k.replace("_"," ") and float(v)
        input_dict = {}
        for k, v in data.items():
            input_dict[k.replace("_", " ")] = float(v)

        input_df = pd.DataFrame([input_dict], columns=model_features)
        y = model_xgb.predict(input_df)[0]
        prediction = "Cancelled" if int(y) == 1 else "Not Cancelled"

    return render(request, "index.html", {"prediction": prediction})
