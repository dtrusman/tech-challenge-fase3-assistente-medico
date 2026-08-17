"""Construção do fluxo LangGraph do assistente virtual médico.

Fluxo:
    carregar_prontuario -> verificar_exames_e_alertas -> consultar_rag
    -> gerar_resposta -> aplicar_guardrails -> registrar_auditoria -> FIM

Cada nó é uma etapa isolada e testável (ver src/agent/nodes.py), coordenadas
pelo LangGraph como uma máquina de estados sobre `AssistenteMedicoState`.
"""

from functools import partial
from typing import Callable, Optional

from langgraph.graph import END, StateGraph

try:
    from . import nodes
    from .state import AssistenteMedicoState
except ImportError:
    import nodes
    from state import AssistenteMedicoState


def build_graph(generator: Optional[Callable[[str], str]] = None):
    graph = StateGraph(AssistenteMedicoState)

    graph.add_node("carregar_prontuario", nodes.carregar_prontuario)
    graph.add_node("verificar_exames_e_alertas", nodes.verificar_exames_e_alertas)
    graph.add_node("consultar_rag", nodes.consultar_rag)
    graph.add_node("gerar_resposta", partial(nodes.gerar_resposta, generator=generator))
    graph.add_node("aplicar_guardrails", nodes.aplicar_guardrails)
    graph.add_node("registrar_auditoria", nodes.registrar_auditoria)

    graph.set_entry_point("carregar_prontuario")
    graph.add_edge("carregar_prontuario", "verificar_exames_e_alertas")
    graph.add_edge("verificar_exames_e_alertas", "consultar_rag")
    graph.add_edge("consultar_rag", "gerar_resposta")
    graph.add_edge("gerar_resposta", "aplicar_guardrails")
    graph.add_edge("aplicar_guardrails", "registrar_auditoria")
    graph.add_edge("registrar_auditoria", END)

    return graph.compile()


def run_assistente(
    pergunta: str,
    paciente_id: str | None = None,
    generator: Optional[Callable[[str], str]] = None,
) -> dict:
    app = build_graph(generator=generator)
    return app.invoke({"pergunta": pergunta, "paciente_id": paciente_id})


if __name__ == "__main__":
    resultado = run_assistente(
        pergunta="Quais cuidados iniciais para esse paciente?",
        paciente_id="P-0001",
    )
    print("ALERTAS:", resultado["alertas"])
    print("\nRESPOSTA FINAL:\n", resultado["resposta_final"])
    print("\nGUARDRAIL FLAGS:", resultado["guardrail_flags"])
