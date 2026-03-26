# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import hashlib, json

from config import APP_INTRO, PRIME_MAXIMALE, SELECTED_FEATURES
from model_loader import load_model_from_project
from pricing_engine import predict_and_price
from shap_service import compute_shap_for_payload
from risk_explain import FEATURE_LABELS
from assistant_hybride import assistant_hybride
import streamlit.components.v1 as components

# ============================================================
# CONFIG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Évaluation du Profil de Risque Client",
    layout="wide"
)

st.title("Évaluation du Profil de Risque Client")
st.write(APP_INTRO)

# ============================================================
# MENU
# ============================================================

page = st.sidebar.radio(
    "Menu",
    ["Profil de risque", "Tarification & Décision", "Assistant métier"]
)

# ============================================================
# SAISIE CLIENT (SIDEBAR)
# ============================================================

st.sidebar.header("Saisie des données client")

with st.sidebar.expander("Informations personnelles", expanded=True):
    age = st.slider("Âge", 18, 100, 40)

with st.sidebar.expander("Situation financière", expanded=True):
    revenu_par_incident = st.number_input("Revenu par incident (€)", 0, 200000, 30000)
    dette_totale = st.number_input("Dette totale (€)", 0, 500000, 20000)
    charges_totales = st.number_input("Charges mensuelles (€)", 0, 10000, 1500)
    ratio_dette_revenu = st.slider("Ratio dette / revenu", 0.0, 5.0, 0.3)

with st.sidebar.expander("Crédit & Assurance", expanded=True):
    historique_credit = st.slider("Historique de crédit", 0, 100, 70)
    score_credit = st.slider("Score crédit", 0, 1000, 650)
    assurance_sur_revenu = st.slider("Assurance sur revenu (%)", 0, 100, 30)
    montant_assurance = st.number_input("Montant de l'assurance (€)", 0, 100000, 10000)

payload = {
    "age": age,
    "revenu_par_incident": revenu_par_incident,
    "assurance_sur_revenu": assurance_sur_revenu,
    "historique_credit": historique_credit,
    "dette_totale": dette_totale,
    "charges_totales": charges_totales,
    "score_credit": score_credit,
    "montant_assurance": montant_assurance,
    "ratio_dette_revenu": ratio_dette_revenu
}

# ============================================================
# CHARGEMENT MODELE
# ============================================================

@st.cache_resource
def load_model():
    return load_model_from_project()

model_local = load_model()

# ============================================================
# MOTEUR METIER + SHAP
# ============================================================

result = predict_and_price(model_local, payload)

# ============================================================
# SIGNATURE CLIENT & MEMOIRE CONVERSATIONNELLE
# ============================================================

def client_signature(payload: dict) -> str:
    """Retourne un hash unique du client selon ses données"""
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

current_client_signature = client_signature(payload)

if "client_signature" not in st.session_state:
    st.session_state.client_signature = current_client_signature
    st.session_state.chat_history = []
    st.session_state.user_question = ""
    st.session_state.use_llm = True

elif st.session_state.client_signature != current_client_signature:
    st.session_state.client_signature = current_client_signature
    st.session_state.chat_history = []
    st.session_state.user_question = ""
    st.session_state.use_llm = True

# ============================================================
# SHAP
# ============================================================

shap_values, shap_details = compute_shap_for_payload(model_local, payload)

# ============================================================
# PAGES
# ============================================================

# --------------------
# PAGE PROFIL DE RISQUE
# --------------------
if page == "Profil de risque":
    st.markdown("### Profil de risque du client")
    col1, col2, col3 = st.columns(3)
    col1.metric("Score de risque", f"{result['score_risque']:.3f}")
    col2.metric("Classe de risque", result["classe_risque"])
    col3.metric(
        "Niveau de vigilance",
        "Faible" if result["classe_risque"] == "Risque faible"
        else "Modéré" if result["classe_risque"] == "Risque moyen"
        else "Élevé"
    )

    if result["classe_risque"] == "Risque faible":
        st.success("Profil financier sain. Aucun facteur de risque majeur détecté.")
    elif result["classe_risque"] == "Risque moyen":
        st.warning("Profil globalement équilibré, avec certains points de vigilance.")
    else:
        st.error("Profil à risque. Plusieurs facteurs contribuent à une exposition élevée.")

    st.markdown("### Principaux facteurs explicatifs du risque")
    shap_df = pd.DataFrame({
        "feature": SELECTED_FEATURES,
        "impact": shap_values.values[0]
    })
    shap_df["impact_abs"] = shap_df["impact"].abs()
    top_features = shap_df.sort_values("impact_abs", ascending=False).head(5)
    top_features["label"] = top_features["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_features["impact"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top_features["label"], top_features["impact"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impact sur le score de risque")
    ax.set_title("Variables les plus influentes")
    plt.gca().invert_yaxis()
    st.pyplot(fig)
    st.caption("🔴 Augmente le risque  |  🟢 Réduit le risque — comparaison par rapport à un client moyen du portefeuille.")
    
    powerbi_iframe = """
    <iframe title="Solvability"
            width="100%"
            height="600"
            src="https://app.powerbi.com/reportEmbed?reportId=8e106567-3161-4e8e-bbe5-ceceb677f09db6&autoAuth=true&ctid=dbd6664d-4eb9-46eb-99d8-5c43ba153c61&actionBarEnabled=true&reportCopilotInEmbed=true"
            frameborder="0"
            allowFullScreen="true">
    </iframe>
    """

    components.html(powerbi_iframe, height=620)

# ------------------------------
# PAGE TARIFICATION & DECISION
# ------------------------------
elif page == "Tarification & Décision":
    st.header("Tarification et décision du client")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prime théorique", f"{int(result['prime_theorique'])} €")
    col2.metric("Plancher tarifaire", f"{result['prime_minimale']} €")
    col3.metric("Plafond tarifaire", f"{PRIME_MAXIMALE} €")
    col4.metric("Prime finale", f"{int(result['prime_finale'])} €")

    st.markdown(
        "La **prime finale** est calculée à partir de la prime théorique, "
        "ajustée selon le coefficient lié à la classe de risque, "
        "en respectant le plancher et le plafond tarifaire."
    )
    st.markdown("---")

    st.subheader("Décision proposée")
    if result["classe_risque"] == "Risque faible":
        st.success(f"{result['decision']} – Acceptation standard. Aucun ajustement requis.")
    elif result["classe_risque"] == "Risque moyen":
        st.warning(f"{result['decision']} – Acceptation avec ajustement tarifaire. Surveillance recommandée.")
    else:
        st.error(f"{result['decision']} – Étude approfondie requise avant acceptation.")

# ------------------------------
# PAGE ASSISTANT METIER
# ------------------------------
elif page == "Assistant métier":
    st.subheader("Assistant métier")
    st.caption(
        "Posez une question sur le risque, la prime, la décision, ou demandez un résumé ou un courrier type. "
        "Les réponses s’appuient sur le profil courant et les facteurs SHAP."
    )
    with st.expander("Conseils pour de meilleures réponses", expanded=False):
        st.markdown(
            "- Soyez précis : *« Pourquoi le score est-il élevé ? »*, *« Explique la prime finale »*.\n"
            "- Pour un courrier : précisez le ton (information, relance) si besoin.\n"
            "- Sans clé OpenAI dans `.env`, l’assistant utilise la FAQ locale (moins riche)."
        )

    
    # Initialisation session_state
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "use_llm" not in st.session_state:
        st.session_state.use_llm = True

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""

    if "submit_question" not in st.session_state:
        st.session_state.submit_question = False

  
    # Checkbox LLM
  
    st.checkbox(
        "Utiliser l’assistant IA (OpenAI) — sinon réponses FAQ locales",
        key="use_llm"
    )

    
    # Fonction appelée à l'entrée 
    
    def on_enter():
        question = st.session_state.question_input.strip()
        if question:
            st.session_state.pending_question = question
            st.session_state.submit_question = True

    
    # Champ texte pour question
    
    st.text_input(
        "💬 Votre question",
        key="question_input",
        value="",
        on_change=on_enter
    )
    
    # Bouton génération courrier client
    
    if st.button("✉️ Générer un courrier client"):
        st.session_state.pending_question = (
            "Courrier formel et professionnel à destination du client. "
        )
        st.session_state.submit_question = True

  
    # Préparer SHAP
    
    shap_pos = [
        f"{e['label']} (impact {e['impact']})"
        for e in shap_details if "augmente" in e["direction"]
    ]
    shap_neg = [
        f"{e['label']} (impact {e['impact']})"
        for e in shap_details if "réduit" in e["direction"]
    ]

    
    # Appel assistant si question saisie
   
    if st.session_state.submit_question:
        question = st.session_state.pending_question
        response = assistant_hybride(
            question=question,
            result=result,
            shap_pos=shap_pos,
            shap_neg=shap_neg,
            chat_history=st.session_state.chat_history,
            use_llm=st.session_state.use_llm
        )

        
        # Vérifier doublons avant ajout
        
        last_question = st.session_state.chat_history[-2]["content"] if len(st.session_state.chat_history) >= 2 else None
        last_response = st.session_state.chat_history[-1]["content"] if len(st.session_state.chat_history) >= 1 else None

        if question != last_question:
            st.session_state.chat_history.append({"role": "Utilisateur", "content": question})

        if response != last_response:
            st.session_state.chat_history.append({"role": "Assistant", "content": response})

        # Reset champ texte
        st.session_state.pending_question = ""
        st.session_state.submit_question = False

    
    # Affichage type chat avec scroll
    
    chat_container = st.container()
    chat_html = "<div style='max-height:500px; overflow-y:auto; border:1px solid #ddd; padding:10px;'>"
    for msg in st.session_state.chat_history:
        if msg["role"] == "Utilisateur":
            chat_html += f"<div style='text-align:right; margin:4px 0;'>💬 {msg['content']}</div>"
        else:
            chat_html += f"<div style='text-align:left; margin:4px 0;'>🤖 {msg['content']}</div>"
    chat_html += "</div>"

    chat_container.markdown(chat_html, unsafe_allow_html=True)
