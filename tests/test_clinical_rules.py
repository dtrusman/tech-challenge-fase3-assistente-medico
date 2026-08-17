from src.agent.clinical_rules import avaliar_alertas


def test_sem_prontuario_nao_gera_alerta():
    assert avaliar_alertas(None) == []


def test_qsofa_alto_gera_alerta_de_sepse():
    prontuario = {
        "sinais_vitais": {"frequencia_respiratoria": 24, "pa_sistolica": 90, "glasgow": 14},
        "exames_resultados": {},
        "hipotese_diagnostica": "",
    }
    alertas = avaliar_alertas(prontuario)
    assert any("qSOFA" in a for a in alertas)


def test_pressao_alta_gera_alerta_hipertensivo():
    prontuario = {
        "sinais_vitais": {"pa_sistolica": 190, "pa_diastolica": 125},
        "exames_resultados": {},
        "hipotese_diagnostica": "",
    }
    alertas = avaliar_alertas(prontuario)
    assert any("hipertensiva" in a for a in alertas)


def test_lactato_critico_gera_alerta():
    prontuario = {
        "sinais_vitais": {},
        "exames_resultados": {"lactato": 5.2},
        "hipotese_diagnostica": "",
    }
    alertas = avaliar_alertas(prontuario)
    assert any("lactato" in a.lower() for a in alertas)


def test_paciente_estavel_sem_alertas():
    prontuario = {
        "sinais_vitais": {
            "frequencia_respiratoria": 16,
            "pa_sistolica": 120,
            "pa_diastolica": 80,
            "glasgow": 15,
        },
        "exames_resultados": {},
        "hipotese_diagnostica": "",
    }
    assert avaliar_alertas(prontuario) == []
