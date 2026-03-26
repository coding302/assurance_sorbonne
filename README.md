# Évaluation du Profil de Risque Client (Streamlit)

Cette application Streamlit estime un risque client, affiche une tarification indicative, et fournit des explications explicables via SHAP. Un assistant métier basé sur l'API OpenAI complète les réponses.

## Prérequis

- Python 3.9+
- Les variables d'environnement requises (voir section suivante)

## Configuration (clés / secrets)

Le projet utilise `OPENAI_API_KEY` pour l'assistant (FAQ + LLM).  
Copie le fichier `.env.example` en `.env` **sans le committer**, puis renseigne la clé.

Variables attendues :
- `OPENAI_API_KEY`

## Installation

```bash
pip install -r requirements.txt
```

## Lancer

```bash
streamlit run app.py
```

Le navigateur s'ouvrira sur `http://localhost:850x` (le port peut varier si déjà utilisé).

## Note sur le modèle

Le fichier de modèle est chargé par `app.py`. L'app cherche :
- `notebooks/xgboost_final_model.joblib` puis
- `xgboost_final_model.joblib`

Assurez-vous que le fichier `xgboost_final_model.joblib` est présent dans le dépôt.

## Structure du code (modules)

- `config.py` — constantes métier (seuils, primes, features, texte d’intro)
- `model_loader.py` — résolution du chemin et chargement du pipeline `joblib`
- `pricing_engine.py` — score, classe de risque, prime indicative
- `shap_service.py` — calcul SHAP (TreeExplainer) à partir du pipeline
- `app.py` — interface Streamlit

## Dépannage : `AttributeError: __sklearn_tags__`

Cette erreur apparaît souvent avec **scikit-learn 1.6+** et certains modèles **XGBoost** chargés depuis un fichier `.joblib`.

**Solution :** réinstallez les dépendances du fichier `requirements.txt` (notamment `scikit-learn<1.6`) :

```bash
pip install -r requirements.txt
```

