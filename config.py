# config.py — constantes métier et liste de features du modèle

SELECTED_FEATURES = [
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

SEUILS_RISQUE = [0.2, 0.6]
PRIME_MINIMALE = 500
PRIME_MAXIMALE = 5000

COEFS = {
    "Risque faible": 1.0,
    "Risque moyen": 1.2,
    "Risque élevé": 1.4,
}

DECISIONS = {
    "Risque faible": "🟢 Acceptation standard",
    "Risque moyen": "🟠 Acceptation avec surprime",
    "Risque élevé": "🔴 Étude approfondie",
}

CHARGEMENTS_ASSURANCE = 0.3

APP_INTRO = """
Cette application évalue le risque financier d'un client et propose une tarification adaptée. Elle fournit également une décision métier, des explications visuelles des facteurs de risque via SHAP, et un assistant métier pour répondre aux questions.
"""
