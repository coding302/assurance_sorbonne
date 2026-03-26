# model_loader.py — chargement du pipeline sklearn + joblib
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib


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
    # Le modèle peut avoir été sauvegardé avec une ancienne version de XGBoost.
    # L'avertissement est attendu et inoffensif : le booster se charge correctement.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*older version of XGBoost.*",
            category=UserWarning,
        )
        return joblib.load(str(path))


def load_model_from_project(base_dir: Optional[Path] = None):
    """Charge le pipeline depuis le projet."""
    p = resolve_model_path(base_dir)
    return load_pipeline(p)
