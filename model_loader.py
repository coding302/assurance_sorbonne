# model_loader.py — chargement du pipeline sklearn + joblib
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib


def _apply_sklearn_pickle_compat() -> None:
    """
    Shim de compatibilité pickle pour sklearn.

    `_RemainderColsList` est une sous-classe privée de `list` introduite dans
    sklearn 1.1 pour stocker les colonnes "remainder" du ColumnTransformer.
    Elle a été supprimée / renommée en sklearn 1.6+.

    Quand un modèle entraîné avec sklearn < 1.6 est dépicklé sur sklearn 1.6+
    (ex. Streamlit Cloud), pickle ne trouve plus la classe → AttributeError.

    Solution : réinjecter `_RemainderColsList = list` dans le module avant le
    chargement. `list` est suffisant car la classe n'était qu'une liste avec
    un nom différent ; le comportement à l'exécution est identique.
    """
    try:
        import sklearn.compose._column_transformer as _ct
        if not hasattr(_ct, "_RemainderColsList"):
            _ct._RemainderColsList = list  # type: ignore[attr-defined]
    except Exception:
        pass  # Ne jamais bloquer le démarrage pour un shim de compat


def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_model_path(base_dir: Optional[Path] = None) -> Path:
    """Cherche le fichier modèle à la racine ou dans notebooks/."""
    base = base_dir or project_root()
    candidates = [
        base / "notebooks" / "xgboost_final_model.joblib",
        base / "xgboost_final_model.joblib",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Modèle introuvable : attendu `notebooks/xgboost_final_model.joblib` "
        "ou `xgboost_final_model.joblib` à la racine du projet."
    )


def load_pipeline(path: "str | Path"):
    # Appliquer le shim avant tout chargement pickle
    _apply_sklearn_pickle_compat()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*older version of XGBoost.*", category=UserWarning)
        return joblib.load(str(path))


def load_model_from_project(base_dir: Optional[Path] = None):
    """Charge le pipeline depuis le projet."""
    p = resolve_model_path(base_dir)
    return load_pipeline(p)
