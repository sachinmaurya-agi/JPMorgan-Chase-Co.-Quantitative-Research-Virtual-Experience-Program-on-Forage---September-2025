"""
pd_expected_loss.py

Usage:
1. Load the trained model and call predict_pd or expected_loss.
   - The trained pipeline `pd_model.pkl` is expected to be in the same directory.

Functions:
- predict_pd(input_df, model_path="pd_model.pkl"):
    returns a numpy array of predicted probabilities of default for each row in input_df.

- expected_loss(input_df, exposure_col=None, recovery_rate=0.10, model_path="pd_model.pkl"):
    returns expected loss for each row: PD * (1 - recovery_rate) * exposure
    If exposure_col is None, the script will try to use the exposure column found during training.

The script also includes a simple example under __main__ showing how to call these functions.
"""

import joblib
import pandas as pd
import numpy as np
import os

def _load_artifacts(model_path="pd_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please ensure pd_model.pkl is present.")
    data = joblib.load(model_path)
    pipeline = data["pipeline"]
    exposure_col = data.get("exposure_col", None)
    target_col = data.get("target_col", None)
    return pipeline, exposure_col, target_col

def predict_pd(input_df, model_path="pd_model.pkl"):
    """Predict probability of default for rows in input_df (pandas DataFrame)."""
    pipeline, _, _ = _load_artifacts(model_path=model_path)
    # Ensure DataFrame
    if not isinstance(input_df, pd.DataFrame):
        input_df = pd.DataFrame([input_df])
    probs = pipeline.predict_proba(input_df)[:,1]
    return probs

def expected_loss(input_df, exposure_col=None, recovery_rate=0.10, model_path="pd_model.pkl"):
    """Compute expected loss = PD * (1 - recovery_rate) * exposure

    input_df: pandas DataFrame or dict
    exposure_col: name of column containing exposure (EAD). If None, uses artifact's exposure_col.
    recovery_rate: e.g. 0.10 for 10% recovery
    Returns: numpy array of expected losses aligned with rows
    """
    pipeline, exposure_col_artifact, _ = _load_artifacts(model_path=model_path)
    if exposure_col is None:
        exposure_col = exposure_col_artifact
    if exposure_col is None:
        raise ValueError("No exposure column detected. Please pass exposure_col explicitly.")
    if not isinstance(input_df, pd.DataFrame):
        input_df = pd.DataFrame([input_df])
    if exposure_col not in input_df.columns:
        raise ValueError(f"Exposure column '{exposure_col}' not found in input dataframe. Columns: {input_df.columns.tolist()}")
    pd_probs = pipeline.predict_proba(input_df)[:,1]
    exposure = input_df[exposure_col].astype(float).values
    expected_loss = pd_probs * (1.0 - recovery_rate) * exposure
    return expected_loss

if __name__ == "__main__":
    # Simple demo: load the model and show predictions for the first 5 rows of the training CSV if available
    csv_path = "Task 3 and 4_Loan_Data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df_sample = df.head(5)
        print("Sample predictions (first 5 rows):")
        try:
            probs = predict_pd(df_sample)
            print("PDs:", probs)
            el = expected_loss(df_sample)
            print("Expected losses:", el)
        except Exception as e:
            print("Could not compute predictions —", e)
    else:
        print("CSV not found locally. Place the original CSV next to this script to run the demo.")
