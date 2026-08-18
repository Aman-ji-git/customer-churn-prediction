from __future__ import annotations

import os
import tempfile

import pandas as pd
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)

from src.database import (
    init_db,
    log_prediction,
)
from src.predict import ChurnPredictor


app = Flask(__name__)

MODEL_PATH = os.getenv(
    "CHURN_MODEL_PATH",
    "models/churn_model.joblib",
)

predictor = ChurnPredictor(
    MODEL_PATH
)

init_db()


FIELDS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def parse_value(
    field,
    value,
):
    if field in {
        "SeniorCitizen",
        "tenure",
    }:
        return int(value)

    if field in {
        "MonthlyCharges",
        "TotalCharges",
    }:
        return float(value)

    return value


@app.get("/")
def index():
    return render_template(
        "index.html",
        result=None,
        error=None,
    )


@app.post("/predict")
def predict():

    try:
        customer = {
            field: parse_value(
                field,
                request.form[field],
            )
            for field in FIELDS
        }

        result = predictor.predict(
            customer
        )

        log_prediction(
            customer_id=None,
            probability=(
                result["probability"] / 100
            ),
            prediction=result[
                "prediction"
            ],
            risk=result["risk"],
        )

        return render_template(
            "index.html",
            result=result,
            error=None,
        )

    except Exception as exc:

        return render_template(
            "index.html",
            result=None,
            error=str(exc),
        ), 400


@app.post("/api/predict")
def api_predict():

    try:
        payload = request.get_json(
            force=True
        )

        result = predictor.predict(
            payload
        )

        return jsonify(result)

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 400


@app.post("/batch-predict")
def batch_predict():

    uploaded = request.files.get(
        "file"
    )

    if uploaded is None:
        return jsonify({
            "error": "CSV file is required."
        }), 400

    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False,
    ) as tmp:

        uploaded.save(
            tmp.name
        )

        temporary_path = tmp.name

    try:

        df = pd.read_csv(
            temporary_path
        )

        probabilities = (
            predictor.model.predict_proba(
                predictor_prepare(df)
            )[:, 1]
        )

        output = df.copy()

        output["ChurnProbability"] = (
            probabilities
        )

        output["ChurnPrediction"] = (
            probabilities >=
            predictor.threshold
        ).map({
            True: "Yes",
            False: "No",
        })

        output["Risk"] = pd.cut(
            probabilities,
            bins=[
                -0.001,
                0.40,
                0.70,
                1.001,
            ],
            labels=[
                "LOW",
                "MEDIUM",
                "HIGH",
            ],
        )

        return output.to_json(
            orient="records"
        )

    finally:

        try:
            os.remove(
                temporary_path
            )
        except OSError:
            pass


def predictor_prepare(df):
    from src.features import create_features

    data = create_features(df)

    if "Churn" in data.columns:
        data = data.drop(
            columns=["Churn"]
        )

    if "customerID" in data.columns:
        data = data.drop(
            columns=["customerID"]
        )

    return data.reindex(
        columns=predictor.features
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
