"""Nós do fluxo LangGraph do assistente médico."""

from typing import Callable, Optional

try:
    from . import clinical_rules, llm_client, prompts
    from .state import AssistenteMedicoState
    from ..rag.patient_db import format_patient_summary, get_patient_record
    from ..rag.retriever import get_or_build_retriever, retrieve_with_sources
    from ..security.audit_log import log_event
    from ..security.guardrails import check_response
except ImportError:  # execução solta (ex.: dentro do notebook)
    import clinical_rules
    import llm_client
    import prompts
    from state import AssistenteMedicoState

    from rag.patient_db import format_patient_summary, get_patient_record
    from rag.retriever import get_or_build_retriever, retrieve_with_sources
    from security.audit_log import log_event
    from security.guardrails import check_response


_RETRIEVER = None


def _retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = get_or_build_retriever()
    return _RETRIEVER


def carregar_prontuario(state: AssistenteMedicoState) -> AssistenteMedicoState:
    """Nó 1: consulta a base estruturada de prontuários (se um paciente foi informado)."""
    paciente_id = state.get("paciente_id")
    prontuario = get_patient_record(paciente_id) if paciente_id else None

    return {
        **state,
        "prontuario": prontuario,
        "resumo_prontuario": format_patient_summary(prontuario) if prontuario else "",
    }


def verificar_exames_e_alertas(state: AssistenteMedicoState) -> AssistenteMedicoState:
    """Nó 2: aplica as regras clínicas do hospital e gera alertas para a equipe."""
    alertas = clinical_rules.avaliar_alertas(state.get("prontuario"))
    return {**state, "alertas": alertas}


def consultar_rag(state: AssistenteMedicoState) -> AssistenteMedicoState:
    """Nó 3: recupera trechos relevantes dos protocolos internos (LangChain + FAISS).

    A busca combina a pergunta do médico com a hipótese diagnóstica e os
    alertas ativos do paciente, para que o retriever traga o protocolo
    clinicamente correto mesmo quando a pergunta em si é genérica
    (ex.: "quais cuidados iniciais?").
    """
    prontuario = state.get("prontuario") or {}
    hipotese = prontuario.get("hipotese_diagnostica", "")
    alertas_texto = " ".join(state.get("alertas", []))

    query_enriquecida = " ".join(
        parte for parte in [state["pergunta"], hipotese, alertas_texto] if parte
    )

    chunks = retrieve_with_sources(query_enriquecida, retriever=_retriever())
    return {**state, "contexto_protocolo": chunks}


def gerar_resposta(
    state: AssistenteMedicoState, generator: Optional[Callable[[str], str]] = None
) -> AssistenteMedicoState:
    """Nó 4: gera a resposta do assistente combinando LLM + prontuário + protocolos."""
    contexto_chunks = state.get("contexto_protocolo", [])
    contexto_texto = "\n\n".join(
        f"[Fonte {i}: {c['source']}]\n{c['content']}" for i, c in enumerate(contexto_chunks)
    )

    prompt = prompts.build_prompt(
        resumo_prontuario=state.get("resumo_prontuario", ""),
        alertas=state.get("alertas", []),
        contexto=contexto_texto,
        pergunta=state["pergunta"],
    )

    resposta = llm_client.generate_response(
        prompt=prompt,
        contexto_chunks=contexto_chunks,
        pergunta=state["pergunta"],
        generator=generator,
    )

    return {**state, "resposta_llm": resposta}


def aplicar_guardrails(state: AssistenteMedicoState) -> AssistenteMedicoState:
    """Nó 5: aplica limites de segurança — nunca prescrever direto, exigir validação humana."""
    resultado = check_response(
        state["resposta_llm"], has_sources=bool(state.get("contexto_protocolo"))
    )
    return {
        **state,
        "bloqueado": resultado["blocked"],
        "guardrail_flags": resultado["flags"],
        "resposta_final": resultado["safe_response"],
    }


def registrar_auditoria(state: AssistenteMedicoState) -> AssistenteMedicoState:
    """Nó 6: grava o evento completo no log de auditoria (rastreabilidade)."""
    log_event(
        event_type="consulta_assistente_medico",
        paciente_id=state.get("paciente_id"),
        query=state.get("pergunta"),
        sources=[c["source"] for c in state.get("contexto_protocolo", [])],
        response_summary=state.get("resposta_final", "")[:500],
        guardrail_flags=state.get("guardrail_flags", []),
        extra={"alertas": state.get("alertas", []), "bloqueado": state.get("bloqueado", False)},
    )
    return state
