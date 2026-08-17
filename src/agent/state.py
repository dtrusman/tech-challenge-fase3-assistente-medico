"""Estado compartilhado do grafo LangGraph do assistente médico."""

from typing import TypedDict


class AssistenteMedicoState(TypedDict, total=False):
    paciente_id: str | None
    pergunta: str
    prontuario: dict | None
    resumo_prontuario: str
    alertas: list[str]
    contexto_protocolo: list[dict]
    resposta_llm: str
    guardrail_flags: list[str]
    bloqueado: bool
    resposta_final: str
