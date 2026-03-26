# model_loader.py
# Charge le modèle dans un format compatible toutes versions Python / XGBoost.
#
# Format préféré (prioritaire) :
#   xgb_booster.json    — booster XGBoost natif JSON (indépendant de Python)
#   preprocessor.joblib — pipeline sklearn de prétraitement
#
# Format legacy (fallback) :
#   xgboost_final_model.joblib — pipeline sklearn complet (Python 3.9 uniquement)

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib
import xgboost as xgb


def project_root() -> Path:
    return Path(__file__).resolve().parent


class SplitModel:
    """Wrapper qui expose la même interface qu'un Pipeline sklearn."""

    def __init__(self, preprocess, booster: xgb.Booster):
        self._preprocess = preprocess
        self._booster = booster

    @property
    def named_steps(self):
        return {"preprocess": self._preprocess, "model": self}

    def get_booster(self) -> xgb.Booster:
        return self._booster

    def predict_proba(self, X):
        """Prédit P(défaut) sans passer par le pipeline sklearn."""
        import numpy as np
        X_proc = self._preprocess.transform(X)
        dmat = xgb.DMatrix(X_proc)
        proba_1 = self._booster.predict(dmat)
        return np.column_stack([1 - proba_1, proba_1])


def _load_split_model(base: Path) -> Optional[SplitModel]:
    """Charge booster JSON + préprocesseur joblib si disponibles."""
    booster_path = base / "xgb_booster.json"
    preprocess_path = base / "preprocessor.joblib"
    if not (booster_path.exists() and preprocess_path.exists()):
        return None

    booster = xgb.Booster()
    booster.load_model(str(booster_path))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        preprocess = joblib.load(str(preprocess_path))

    return SplitModel(preprocess, booster)


def _load_legacy_pipeline(base: Path):
    """Charge l'ancien pipeline joblib complet (Python 3.9 local uniquement)."""
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
        "Modèle introuvable : ni xgb_booster.json+preprocessor.joblib, "
        "ni xgboost_final_model.joblib ne sont présents."
    )


def load_model_from_project(base_dir: Optional[Path] = None):
    """
    Charge le modèle en privilégiant le format JSON natif (compatible Python 3.14+).
    Retombe sur l'ancien joblib si les fichiers JSON ne sont pas présents.
    """
    base = base_dir or project_root()
    model = _load_split_model(base)
    if model is not None:
        return model
    return _load_legacy_pipeline(base)
