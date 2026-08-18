from __future__ import annotations

import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw columns without using target information."""
    data = df.copy()

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce",
        )

    for column in ["tenure", "MonthlyCharges"]:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    return data


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering used identically during training and inference.
    No feature uses the Churn target.
    """
    data = clean_raw_data(df)

    # Customer economics
    safe_tenure = data["tenure"].replace(0, np.nan)

    data["AvgMonthlySpend"] = (
        data["TotalCharges"] / safe_tenure
    ).replace([np.inf, -np.inf], np.nan)

    data["AvgMonthlySpend"] = data[
        "AvgMonthlySpend"
    ].fillna(data["MonthlyCharges"])

    data["ChargesPerTenure"] = (
        data["MonthlyCharges"] / (data["tenure"] + 1)
    )

    # Product/service adoption
    available = [
        c for c in SERVICE_COLUMNS
        if c in data.columns
    ]

    data["ServiceCount"] = sum(
        (data[c] == "Yes").astype(int)
        for c in available
    )

    # Behavioral indicators
    data["IsMonthToMonth"] = (
        data["Contract"] == "Month-to-month"
    ).astype(int)

    data["IsFiber"] = (
        data["InternetService"] == "Fiber optic"
    ).astype(int)

    data["UsesElectronicCheck"] = (
        data["PaymentMethod"] == "Electronic check"
    ).astype(int)

    data["NoTechSupport"] = (
        data["TechSupport"] == "No"
    ).astype(int)

    data["NoOnlineSecurity"] = (
        data["OnlineSecurity"] == "No"
    ).astype(int)

    # Nonlinear tenure representation
    data["TenureBucket"] = pd.cut(
        data["tenure"],
        bins=[-1, 6, 12, 24, 48, 72, np.inf],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72", "73+"],
    ).astype("object")

    return data
