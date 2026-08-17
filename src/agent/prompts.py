"""Prompts usados pelo assistente médico."""

SYSTEM_PROMPT = (
    "Você é um assistente virtual médico do Hospital Vida Plena. Responda de "
    "forma objetiva, em português, usando exclusivamente as informações do "
    "prontuário do paciente e dos trechos de protocolo fornecidos como "
    "contexto. Sempre cite a fonte do protocolo usado, no formato [Fonte N]. "
    "Nunca prescreva medicação de forma direta e definitiva: apresente "
    "sugestões de conduta que exigem validação de um médico responsável."
)

RESPONSE_TEMPLATE = (
    "{system_prompt}\n\n"
    "### Resumo do paciente:\n{resumo_prontuario}\n\n"
    "### Alertas ativos:\n{alertas}\n\n"
    "### Contexto dos protocolos internos:\n{contexto}\n\n"
    "### Pergunta do médico:\n{pergunta}\n\n"
    "### Resposta:\n"
)


def build_prompt(resumo_prontuario: str, alertas: list[str], contexto: str, pergunta: str) -> str:
    alertas_texto = "\n".join(f"- {a}" for a in alertas) if alertas else "Nenhum alerta ativo."
    return RESPONSE_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        resumo_prontuario=resumo_prontuario or "Nenhum paciente vinculado a esta consulta.",
        alertas=alertas_texto,
        contexto=contexto or "Nenhum trecho de protocolo recuperado.",
        pergunta=pergunta,
    )
