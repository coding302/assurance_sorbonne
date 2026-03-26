# model_loader.py
#
# Charge le modèle dans un format 100% compatible toutes versions Python/XGBoost.
#
# Format préféré (prioritaire, compatible Python 3.14+) :
#   xgb_booster.json        — booster XGBoost natif JSON
#   preprocessor_params.npz — paramètres StandardScaler en numpy (pas de pickle sklearn)
#
# Format legacy (fallback local Python 3.9 uniquement) :
#   xgboost_final_model.joblib

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

# Ordre des features attendu par le modèle
FEATURE_ORDER = [
    "revenu_par_incident",
    "assurance_sur_revenu",
    "historique_credit",
    "dette_totale",
    "charges_totales",
    "score_credit",
    "montant_assurance",
    "ratio_dette_revenu",
    "age",
]


class NumpyScaler:
    """StandardScaler reconstruit depuis les paramètres numpy (aucun pickle sklearn)."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, X):
        """Accepte un DataFrame pandas ou un array numpy."""
        if hasattr(X, "values"):
            X = X[FEATURE_ORDER].values.astype(float)
        return (X - self.mean_) / self.scale_


class SplitModel:
    """
    Wrapper exposant la même interface qu'un Pipeline sklearn
    (compatible avec pricing_engine.py et shap_service.py existants).
    """

    def __init__(self, scaler: NumpyScaler, booster: xgb.Booster):
        self._scaler = scaler
        self._booster = booster

    @property
    def named_steps(self):
        return {"preprocess": self._scaler, "model": self}

    def get_booster(self) -> xgb.Booster:
        return self._booster

    def predict_proba(self, X):
        X_proc = self._scaler.transform(X)
        dmat = xgb.DMatrix(X_proc)
        proba_1 = self._booster.predict(dmat)
        return np.column_stack([1 - proba_1, proba_1])


def project_root() -> Path:
    return Path(__file__).resolve().parent


def _load_split_model(base: Path) -> Optional[SplitModel]:
    """Charge le format numpy+JSON (compatible Python 3.14+)."""
    booster_path = base / "xgb_booster.json"
    params_path = base / "preprocessor_params.npz"

    if not (booster_path.exists() and params_path.exists()):
        return None

    # Booster XGBoost natif JSON — pas de pickle
    booster = xgb.Booster()
    booster.load_model(str(booster_path))

    # Paramètres StandardScaler depuis numpy — pas de pickle sklearn
    params = np.load(str(params_path))
    scaler = NumpyScaler(mean=params["mean"], scale=params["scale"])

    return SplitModel(scaler, booster)


def _load_legacy_pipeline(base: Path):
    """Fallback : charge l'ancien pipeline joblib complet (Python 3.9 local)."""
    import joblib

    candidates = [
        base / "notebooks" / "xgboost_final_model.joblib",
        base / "xgboost_final_model.joblib",
    ]
    for p in candidates:
        if p.exists():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                return joblib.load(str(p))
    raise FileNotFoundError(
        "Modèle introuvable : ni xgb_booster.json+preprocessor_params.npz, "
        "ni xgboost_final_model.joblib ne sont présents."
    )


def load_model_from_project(base_dir: Optional[Path] = None):
    """
    Charge le modèle en privilégiant le format JSON+numpy (compatible Python 3.14+).
    Retombe sur le pipeline joblib legacy si les fichiers npz/json sont absents.
    """
    base = base_dir or project_root()
    model = _load_split_model(base)
    if model is not None:
        return model
    return _load_legacy_pipeline(base)
