from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from config import MODEL_PATH
from src.features import create_features


class ChurnPredictor:
    """
    Production inference service.

    The model is loaded once and reused for every request.
    """

    def __init__(self, model_path=MODEL_PATH):
        artifact = joblib.load(
            model_path
        )

        self.model = artifact["model"]
        self.threshold = float(
            artifact["threshold"]
        )
        self.features = artifact["features"]

    def predict(
        self,
        customer: dict[str, Any],
    ) -> dict[str, Any]:

        df = pd.DataFrame(
            [customer]
        )

        df = create_features(df)

        if "customerID" in df.columns:
            df = df.drop(
                columns=["customerID"]
            )

        # Prevent training/inference column mismatch.
        df = df.reindex(
            columns=self.features
        )

        probability = float(
            self.model.predict_proba(
                df
            )[0, 1]
        )

        prediction = (
            probability >= self.threshold
        )

        if probability >= 0.70:
            risk = "HIGH"
        elif probability >= 0.40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "prediction": (
                "Likely to Churn"
                if prediction
                else "Likely to Stay"
            ),
            "churn": bool(prediction),
            "probability": round(
                probability * 100,
                2,
            ),
            "risk": risk,
        }
