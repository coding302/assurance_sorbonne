# shap_service.py — SHAP TreeExplainer sur le booster XGB du pipeline
from __future__ import annotations

import pandas as pd
import shap

from config import SELECTED_FEATURES
from risk_explain import explain_risk_from_shap


def compute_shap_for_payload(pipeline, payload: dict):
    """
    Retourne (shap_values, shap_details) pour le client courant.
    """
    preprocess_local = pipeline.named_steps["preprocess"]
    xgb_local = pipeline.named_steps["model"]

    X_input = pd.DataFrame([payload])[SELECTED_FEATURES]
    X_processed = preprocess_local.transform(X_input)
    X_processed_df = pd.DataFrame(X_processed, columns=SELECTED_FEATURES)

    # Passer le booster natif à SHAP pour éviter les problèmes de version sklearn
    explainer = shap.TreeExplainer(xgb_local.get_booster())
    shap_values = explainer(X_processed_df)
    shap_details = explain_risk_from_shap(shap_values, SELECTED_FEATURES)

    return shap_values, shap_details
