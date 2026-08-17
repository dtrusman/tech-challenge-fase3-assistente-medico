from src.preprocessing.anonymize import anonymize_text


def test_remove_cpf():
    assert "[CPF_REMOVIDO]" in anonymize_text("CPF: 123.456.789-00")


def test_remove_email():
    assert "[EMAIL_REMOVIDO]" in anonymize_text("contato: fulano@example.com")


def test_remove_nome_com_rotulo():
    resultado = anonymize_text("Paciente: Maria Aparecida Souza Lima")
    assert "[NOME_REMOVIDO]" in resultado
    assert "Maria" not in resultado


def test_preserva_texto_sem_pii():
    texto = "Iniciar antibioticoterapia empírica em até 60 minutos."
    assert anonymize_text(texto) == texto
