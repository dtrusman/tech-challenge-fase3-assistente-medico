"""Anonimização de dados clínicos sensíveis (PII) antes do uso em fine-tuning ou RAG."""

import re

# Padrões de identificadores pessoais comuns em documentos clínicos brasileiros
PATTERNS = {
    "CPF": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "RG": re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-?[0-9Xx]\b"),
    "TELEFONE": re.compile(r"\b(?:\+?55\s?)?\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b"),
    "CEP": re.compile(r"\b\d{5}-?\d{3}\b"),
    "DATA_NASCIMENTO": re.compile(
        r"\b(?:data\s+de\s+nascimento|nascido\s+em|dn)\s*[:\-]?\s*\d{1,2}/\d{1,2}/\d{2,4}",
        re.IGNORECASE,
    ),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PRONTUARIO": re.compile(r"\b(?:prontu[aá]rio|registro)\s*n?[ºo°]?\s*[:\-]?\s*\d{4,}\b", re.IGNORECASE),
}

# Nomes próprios completos (heurística simples: duas ou mais palavras capitalizadas seguidas,
# precedidas por rótulo de paciente/médico) — cobre os campos mais comuns em laudos/receitas.
NAME_LABEL_PATTERN = re.compile(
    r"\b(paciente|nome do paciente|m[eé]dico respons[aá]vel|dr\.?|dra\.?)\s*[:\-]?\s*"
    r"([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+){1,4})",
    re.IGNORECASE,
)


def anonymize_text(text: str) -> str:
    """Substitui identificadores pessoais diretos por marcadores neutros.

    Não é um anonimizador clinicamente certificado — é uma camada de curadoria
    para remover PII óbvio de exemplos de treinamento/indexação antes do uso.
    Para dados hospitalares reais, validar com o time de compliance/DPO do hospital.
    """
    anonymized = text

    for label, pattern in PATTERNS.items():
        anonymized = pattern.sub(f"[{label}_REMOVIDO]", anonymized)

    def _replace_name(match: re.Match) -> str:
        rotulo = match.group(1)
        return f"{rotulo}: [NOME_REMOVIDO]"

    anonymized = NAME_LABEL_PATTERN.sub(_replace_name, anonymized)

    return anonymized


def anonymize_record(record: dict, fields: list[str]) -> dict:
    """Aplica anonimização nos campos de texto indicados de um registro (dict)."""
    out = dict(record)
    for field in fields:
        if field in out and isinstance(out[field], str):
            out[field] = anonymize_text(out[field])
    return out
