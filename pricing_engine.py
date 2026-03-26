# pricing_engine.py — score de risque, classe, prime indicative
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (
    CHARGEMENTS_ASSURANCE,
    COEFS,
    DECISIONS,
    PRIME_MAXIMALE,
    PRIME_MINIMALE,
    SELECTED_FEATURES,
    SEUILS_RISQUE,
)


def _predict_proba_safe(pipeline, X: pd.DataFrame) -> float:
    """
    Retourne P(défaut) en appelant le booster XGBoost directement,
    ce qui contourne le check_is_fitted/get_tags de sklearn 1.6
    incompatible avec certains modèles XGBoost sérialisés.
    """
    preprocess = pipeline.named_steps["preprocess"]
    xgb_model = pipeline.named_steps["model"]

    X_proc = preprocess.transform(X)

    # Appel natif XGBoost — bypass complet du Pipeline sklearn
    booster = xgb_model.get_booster()
    dmat = xgb.DMatrix(
        X_proc,
        feature_names=getattr(booster, "feature_names", None),
    )
    proba_positive = booster.predict(dmat)
    return float(proba_positive[0])


def classe_risque_depuis_score(score: float) -> str:
    if score < SEUILS_RISQUE[0]:
        return "Risque faible"
    if score < SEUILS_RISQUE[1]:
        return "Risque moyen"
    return "Risque élevé"


def predict_and_price(pipeline, data: dict) -> dict:
    """
    Calcule le score (probabilité de défaut), la classe, la décision affichée
    et la prime indicative.
    """
    X = pd.DataFrame([data])[SELECTED_FEATURES]
    score_risque = _predict_proba_safe(pipeline, X)

    classe = classe_risque_depuis_score(score_risque)
    coef = COEFS[classe]
    decision = DECISIONS[classe]

    prime_theorique = score_risque * data["montant_assurance"] * (1 + CHARGEMENTS_ASSURANCE)
    prime_finale = max(prime_theorique * coef, PRIME_MINIMALE)
    prime_finale = min(prime_finale, PRIME_MAXIMALE)

    return {
        "score_risque": score_risque,
        "classe_risque": classe,
        "decision": decision,
        "prime_theorique": float(prime_theorique),
        "prime_minimale": PRIME_MINIMALE,
        "prime_finale": float(prime_finale),
    }
