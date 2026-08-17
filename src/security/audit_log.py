"""Logging estruturado para rastreamento e auditoria das interações do assistente.

Cada interação (pergunta, contexto usado, resposta, decisões de guardrail) é
gravada em JSON Lines, permitindo auditoria posterior de qualquer resposta
gerada pelo assistente — quem perguntou, o que foi recuperado do RAG, qual
fonte foi citada e se algum guardrail de segurança foi acionado.
"""

import json
import logging
import os
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(BASE_DIR, "logs")
AUDIT_LOG_PATH = os.path.join(LOG_DIR, "audit.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)

_logger = logging.getLogger("assistente_medico.audit")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _handler = logging.FileHandler(AUDIT_LOG_PATH, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.propagate = False


def log_event(
    event_type: str,
    paciente_id: str | None = None,
    query: str | None = None,
    sources: list[str] | None = None,
    response_summary: str | None = None,
    guardrail_flags: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    """Registra um evento de auditoria e retorna o registro gravado."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "paciente_id": paciente_id,
        "query": query,
        "sources": sources or [],
        "response_summary": response_summary,
        "guardrail_flags": guardrail_flags or [],
        "extra": extra or {},
    }
    _logger.info(json.dumps(entry, ensure_ascii=False))
    return entry


def read_audit_log(path: str = AUDIT_LOG_PATH) -> list[dict]:
    """Lê o log de auditoria completo (uso em relatórios/depuração)."""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
