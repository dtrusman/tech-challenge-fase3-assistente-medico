# Diagrama do fluxo — Assistente Médico (LangGraph)

```mermaid
flowchart TD
    A[Início: pergunta do médico + paciente_id opcional] --> B[carregar_prontuario]
    B -->|consulta base estruturada de prontuários| C[verificar_exames_e_alertas]
    C -->|regras clínicas: qSOFA, PA, lactato, glicemia, dengue| D[consultar_rag]
    D -->|LangChain + FAISS sobre protocolos internos, citando fonte| E[gerar_resposta]
    E -->|LLM fine-tuned + contexto do paciente + protocolos| F[aplicar_guardrails]
    F -->|bloqueia tópicos proibidos, impede prescrição direta,\nexige disclaimer de validação humana| G[registrar_auditoria]
    G -->|log estruturado JSONL: pergunta, fontes, alertas, flags| H[Fim: resposta final ao médico]

    C -.->|alerta crítico| I[[Equipe médica / enfermagem]]
```

## Descrição das etapas

| Nó | Responsabilidade | Módulo |
|---|---|---|
| `carregar_prontuario` | Consulta a base estruturada de prontuários (mock) do paciente informado | `src/rag/patient_db.py` |
| `verificar_exames_e_alertas` | Aplica regras clínicas dos protocolos (qSOFA, PA, lactato, glicemia, dengue) e gera alertas para a equipe | `src/agent/clinical_rules.py` |
| `consultar_rag` | Recupera trechos relevantes dos protocolos internos via LangChain + FAISS, combinando a pergunta com a hipótese diagnóstica/alertas do paciente | `src/rag/retriever.py` |
| `gerar_resposta` | Gera a resposta com o LLM fine-tuned, usando prontuário + alertas + protocolos como contexto | `src/agent/llm_client.py` |
| `aplicar_guardrails` | Bloqueia tópicos proibidos, sinaliza linguagem de prescrição direta e garante o disclaimer de validação humana | `src/security/guardrails.py` |
| `registrar_auditoria` | Grava o evento completo (pergunta, fontes citadas, alertas, flags de guardrail) em log estruturado para auditoria | `src/security/audit_log.py` |
