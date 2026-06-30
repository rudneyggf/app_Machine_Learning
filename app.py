import numpy as np
import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Configuração da página
st.set_page_config(
    page_title="Previsão de Risco de Saúde Mental",
    page_icon="🧠",
    layout="centered"
)

# Carrega o modelo treinado (cacheado para não recarregar a cada interação)
@st.cache_resource
def carregar_modelo():
    artefato = joblib.load("modelo_mental_health.joblib")
    return artefato["model"], artefato["atributos"]

modelo, ATRIBUTOS = carregar_modelo()

RISCO_LABELS = {
    0: ("Baixo risco", "🟢"),
    1: ("Risco moderado", "🟡"),
    2: ("Alto risco", "🔴"),
}

# Cabeçalho
st.title("Previsão de Risco de Saúde Mental")
st.markdown(
    "Este app usa um **modelo de Árvore de Decisão**para estimar " \
    "um nível de risco a partir de hábitos e indicadores pessoais.\n\n"
)

st.divider()

# Formulário de entrada
st.subheader("Preencha as informações abaixo")

col1, col2 = st.columns(2)

with col1:
    sleep_hours = st.slider(
        "Horas de sono por noite", min_value=3.0, max_value=10.0, value=7.0, step=0.1
    )
    depression_score = st.slider(
        "Nível de depressão (1 = baixo, 10 = alto)", 1, 10, 3
    )
    anxiety_score = st.slider(
        "Nível de ansiedade (1 = baixo, 10 = alto)", 1, 10, 3
    )
    social_support_score = st.slider(
        "Nível de apoio social (1 = baixo, 10 = alto)", 1, 10, 7
    )

with col2:
    financial_stress_level = st.slider(
        "Nível de estresse financeiro (1 = baixo, 10 = alto)", 1, 10, 5
    )
    work_stress_level = st.slider(
        "Nível de estresse no trabalho (1 = baixo, 10 = alto)", 1, 10, 5
    )
    panic_attack_history = st.radio(
        "Histórico de ataques de pânico?", options=[0, 1],
        format_func=lambda x: "Sim" if x == 1 else "Não", horizontal=True
    )
    family_history = st.radio(
        "Histórico familiar de doença mental?", options=[0, 1],
        format_func=lambda x: "Sim" if x == 1 else "Não", horizontal=True
    )

st.divider()

# -----------------------------------------------------------------------
# Predição
# -----------------------------------------------------------------------
if st.button(" Calcular risco", type="primary", use_container_width=True):
    entrada = pd.DataFrame([{
        "sleep_hours": sleep_hours,
        "depression_score": depression_score,
        "panic_attack_history": panic_attack_history,
        "anxiety_score": anxiety_score,
        "family_history_mental_illness": family_history,
        "social_support_score": social_support_score,
        "financial_stress_level": financial_stress_level,
        "work_stress_level": work_stress_level,
    }])[ATRIBUTOS]

    pred = modelo.predict(entrada)[0]
    proba = modelo.predict_proba(entrada)[0]
    classes = modelo.classes_.tolist()
    
    BEST_THR = 0.2414
    idx_high = classes.index(2)  

   
    if proba[idx_high] >= BEST_THR:
        pred = 2  
    else:
        probs_sem_high = proba.copy()
        probs_sem_high[idx_high] = -1 
        pred = classes[np.argmax(probs_sem_high)]

    label, emoji = RISCO_LABELS.get(pred, ("Desconhecido", "❓"))

    st.markdown(f"## Resultado: {emoji} {label}")

    proba_df = pd.DataFrame({
        "Classe de risco": [RISCO_LABELS[c][0] for c in modelo.classes_],
        "Probabilidade": proba
    }).sort_values("Probabilidade", ascending=False)

    chart = (
        alt.Chart(proba_df)
        .mark_bar()
        .encode(
            x=alt.X("Classe de risco:N", sort=None,axis=alt.Axis(labelAngle=0)),  # N para dados nominais/texto
            y=alt.Y("Probabilidade:Q", scale=alt.Scale(domain=[0, 1])),  # Q para quantitativo
        )
    )
    st.altair_chart(chart, use_container_width=True)


st.divider()
st.caption(
    "Modelo: DecisionTreeClassifier (scikit-learn) · "
    "max_depth=8 · min_samples_split=20 · min_samples_leaf=10"
)