"""Frontend Streamlit opcional para demonstração local do assistente médico.

Uso:
    streamlit run app/streamlit_app.py

Requer a API rodando em paralelo (uvicorn src.api.main:app).
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/consulta"

st.set_page_config(page_title="Assistente Médico — Hospital Vida Plena", page_icon=":hospital:")
st.title("Assistente Virtual Médico")
st.caption("Hospital Vida Plena (fictício) — Tech Challenge Fase 3")

with st.sidebar:
    st.header("Contexto do paciente")
    paciente_id = st.selectbox(
        "Paciente (prontuário mock)", ["Nenhum", "P-0001", "P-0002", "P-0003"]
    )
    st.markdown(
        "- **P-0001**: suspeita de sepse\n"
        "- **P-0002**: crise hipertensiva\n"
        "- **P-0003**: suspeita de dengue"
    )

pergunta = st.text_input("Pergunta clínica:")

if st.button("Consultar") and pergunta:
    payload = {"pergunta": pergunta}
    if paciente_id != "Nenhum":
        payload["paciente_id"] = paciente_id

    with st.spinner("Consultando protocolos internos e gerando resposta..."):
        try:
            resp = requests.post(API_URL, json=payload, timeout=120)
            resp.raise_for_status()
            dados = resp.json()

            if dados["alertas"]:
                for alerta in dados["alertas"]:
                    st.error(alerta)

            st.markdown(dados["resposta"])

            if dados["guardrail_flags"]:
                st.warning(f"Guardrails acionados: {', '.join(dados['guardrail_flags'])}")

            with st.expander("Fontes consultadas"):
                st.write(dados["fontes"])

        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao consultar a API: {e}")
