from src.security.guardrails import check_response, DISCLAIMER


def test_bloqueia_topico_proibido():
    resultado = check_response("Qual a dose letal de X?", has_sources=False)
    assert resultado["blocked"] is True
    assert "topico_bloqueado" in resultado["flags"]


def test_detecta_prescricao_direta():
    resultado = check_response("Tome 500mg de dipirona agora.", has_sources=False)
    assert resultado["blocked"] is False
    assert "linguagem_de_prescricao_direta_detectada" in resultado["flags"]


def test_resposta_sempre_recebe_disclaimer():
    resultado = check_response("Resposta qualquer.", has_sources=False)
    assert resultado["safe_response"].endswith(DISCLAIMER.strip())


def test_sinaliza_ausencia_de_citacao_quando_ha_fontes():
    resultado = check_response("Resposta sem citar nada.", has_sources=True)
    assert "resposta_sem_citacao_de_fonte_explicita" in resultado["flags"]


def test_nao_sinaliza_ausencia_de_citacao_quando_cita_fonte():
    resultado = check_response("Conforme [Fonte 0: protocolo.txt], siga X.", has_sources=True)
    assert "resposta_sem_citacao_de_fonte_explicita" not in resultado["flags"]
