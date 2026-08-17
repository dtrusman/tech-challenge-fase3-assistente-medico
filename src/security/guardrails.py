"""Guardrails de segurança: limites de atuação do assistente virtual médico.

Regra central do desafio: o assistente NUNCA prescreve diretamente sem
validação humana. Este módulo detecta linguagem de prescrição direta,
garante que toda resposta clínica traga o disclaimer de validação humana e
garante que respostas fundamentadas em protocolo sempre citem a fonte
(explainability).
"""

import re

DISCLAIMER = (
    "\n\n⚠️ Esta é uma sugestão de apoio à decisão clínica baseada nos "
    "protocolos internos do Hospital Vida Plena. NÃO substitui o julgamento "
    "médico e exige validação humana antes de qualquer conduta ou prescrição."
)

# Padrões que indicam uma prescrição direta e imperativa (sem condicional/sugestão)
DIRECT_PRESCRIPTION_PATTERNS = [
    re.compile(r"\bprescrevo\b", re.IGNORECASE),
    re.compile(r"\btome\s+\d", re.IGNORECASE),
    re.compile(r"\badministre\s+\d", re.IGNORECASE),
    re.compile(r"\baplique\s+\d", re.IGNORECASE),
    re.compile(r"\b\d+\s?mg\b.{0,15}\bde\s+\d+\s?em\s?\d+\s?horas\b", re.IGNORECASE),
]

BLOCKED_TOPICS_PATTERNS = [
    re.compile(r"\bdose\s+letal\b", re.IGNORECASE),
    re.compile(r"\bcomo\s+(cometer|induzir)\s+suic[ií]dio\b", re.IGNORECASE),
]


def detects_direct_prescription(text: str) -> bool:
    return any(p.search(text) for p in DIRECT_PRESCRIPTION_PATTERNS)


def detects_blocked_topic(text: str) -> bool:
    return any(p.search(text) for p in BLOCKED_TOPICS_PATTERNS)


def has_source_citation(text: str) -> bool:
    return bool(re.search(r"\[Fonte\s*\d+", text, re.IGNORECASE))


def check_response(response_text: str, has_sources: bool) -> dict:
    """Avalia uma resposta gerada e retorna decisão + flags de guardrail.

    Retorna:
        {
          "blocked": bool,        # resposta não deve ser entregue como está
          "flags": [str, ...],    # motivos identificados
          "safe_response": str,   # resposta ajustada (com disclaimer, se aplicável)
        }
    """
    flags = []

    if detects_blocked_topic(response_text):
        return {
            "blocked": True,
            "flags": ["topico_bloqueado"],
            "safe_response": (
                "Não posso ajudar com esse pedido. Por favor, contate a equipe "
                "médica responsável imediatamente."
            ),
        }

    if detects_direct_prescription(response_text):
        flags.append("linguagem_de_prescricao_direta_detectada")

    if has_sources and not has_source_citation(response_text):
        flags.append("resposta_sem_citacao_de_fonte_explicita")

    safe_response = response_text.strip()
    if not safe_response.endswith(DISCLAIMER.strip()):
        safe_response += DISCLAIMER

    return {
        "blocked": False,
        "flags": flags,
        "safe_response": safe_response,
    }
