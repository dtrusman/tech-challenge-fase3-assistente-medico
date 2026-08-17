# Assistente Virtual Médico — Hospital Vida Plena

**Tech Challenge — Fase 3 | Pós-graduação em IA para Devs (FIAP)**

Assistente virtual médico treinado com dados internos de um hospital fictício ("Hospital Vida
Plena"), capaz de auxiliar condutas clínicas, responder dúvidas de médicos e sugerir
procedimentos com base em protocolos internos — com fluxos de decisão automatizados e seguros,
coordenados com **LangChain** e **LangGraph**.

> ⚠️ **Aviso:** todos os dados clínicos, protocolos, prontuários e o hospital em si são
> **fictícios**, criados apenas para fins didáticos. Nenhum dado real de paciente foi utilizado.
> Nenhuma resposta deste assistente deve ser usada como orientação clínica real — toda sugestão
> exige validação humana obrigatória de um médico habilitado.

## Visão geral da solução

| Requisito do desafio | Implementação |
|---|---|
| Fine-tuning de LLM com dados médicos internos | LoRA sobre `TinyLlama-1.1B-Chat` — `src/finetuning/` |
| Preprocessing, anonimização e curadoria | `src/preprocessing/` |
| Assistente com LangChain + consulta a base estruturada | `src/rag/` (FAISS + prontuários mock) |
| Fluxos automatizados com LangGraph | `src/agent/` |
| Segurança, limites de atuação, logging, explainability | `src/security/` |
| Projeto modularizado em Python | pacote `src/` |

O ponto de entrada principal do projeto é o notebook
[`notebooks/tech_challenge_fase3_colab.ipynb`](notebooks/tech_challenge_fase3_colab.ipynb),
pensado para rodar no **Google Colab** (GPU T4) do início ao fim: preparação de dados,
fine-tuning, avaliação, indexação RAG e execução do fluxo LangGraph completo.

Uma API (FastAPI) e um frontend (Streamlit) opcionais também estão incluídos para
demonstração/deploy local, reaproveitando os mesmos módulos do notebook.

## Estrutura do repositório

```
.
├── notebooks/
│   └── tech_challenge_fase3_colab.ipynb   # ponto de entrada principal (Colab)
├── src/
│   ├── preprocessing/   # curadoria, anonimização, preparo do dataset de fine-tuning
│   ├── finetuning/       # treino (LoRA) e avaliação do LLM
│   ├── rag/              # indexação FAISS, retriever com citação de fonte, prontuários mock
│   ├── agent/             # grafo LangGraph, regras clínicas, prompts, cliente do LLM
│   ├── security/          # guardrails e logging de auditoria
│   └── api/                # API FastAPI opcional
├── app/
│   └── streamlit_app.py    # frontend opcional para demonstração local
├── data/
│   ├── raw/                 # protocolos internos (.txt) e exemplo com PII para demo de anonimização
│   └── processed/           # prontuários mock; dataset/índice gerados pelo pipeline (não versionados)
├── docs/
│   └── fluxo_langgraph.md   # diagrama do fluxo (Mermaid)
├── reports/
│   └── relatorio_tecnico.md # relatório técnico exigido na entrega
├── tests/                    # testes unitários (guardrails, regras clínicas, anonimização)
└── requirements.txt
```

## Como rodar — Google Colab (recomendado)

1. Suba este repositório para o GitHub (ex.: `tech-challenge-fase3-assistente-medico`).
2. Abra o notebook `notebooks/tech_challenge_fase3_colab.ipynb` no Google Colab.
3. Em **Ambiente de execução → Alterar tipo de ambiente de execução**, selecione GPU (T4).
4. Na primeira célula de setup, edite `REPO_URL` com a URL do seu repositório.
5. Execute as células em ordem, de cima para baixo. O notebook cobre:
   - preparação e anonimização dos dados;
   - fine-tuning (LoRA) do LLM;
   - avaliação do modelo (ROUGE-L, base vs. fine-tuned);
   - indexação dos protocolos internos (LangChain + FAISS);
   - execução do fluxo LangGraph completo, com exemplos de pacientes;
   - demonstração do guardrail de segurança;
   - visualização dos logs de auditoria.

Esse notebook é o material de referência para gravar o vídeo de demonstração exigido na
entrega.

### Usando o dataset MedQuAD real (opcional)

Por padrão, o pipeline usa uma pequena amostra embutida no estilo MedQuAD. Para usar o dataset
completo (milhares de pares de pergunta/resposta clínicas), descomente no notebook (ou rode
localmente):

```bash
git clone https://github.com/abachaa/MedQuAD.git data/raw/MedQuAD
```

O script `src/preprocessing/prepare_dataset.py` detecta automaticamente a pasta e passa a usá-la.

## Como rodar — ambiente local

Recomenda-se um ambiente virtual dedicado, para não conflitar com outras instalações Python:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Preparar os dados de fine-tuning:

```bash
python -m src.preprocessing.prepare_dataset
```

Fine-tuning (LoRA):

```bash
python -m src.finetuning.train
# teste rápido, com modelo minúsculo e poucos passos:
python -m src.finetuning.train --model sshleifer/tiny-gpt2 --smoke-test
```

Avaliação do modelo:

```bash
python -m src.finetuning.evaluate
```

Indexar os protocolos internos (RAG):

```bash
python -m src.rag.indexing
```

Rodar o fluxo LangGraph diretamente (usa o fallback extrativo se não houver modelo fine-tuned):

```bash
python -m src.agent.graph
```

Rodar a API e o frontend de demonstração:

```bash
uvicorn src.api.main:app --reload
# em outro terminal:
streamlit run app/streamlit_app.py
```

Rodar os testes:

```bash
pip install pytest
pytest tests/ -v
```

## Segurança e limites de atuação

- O assistente **nunca prescreve diretamente** — toda resposta inclui um disclaimer explícito
  de que exige validação humana, e linguagem de prescrição direta é sinalizada pelo guardrail.
- Perguntas fora do escopo clínico seguro são bloqueadas antes de chegar ao usuário.
- Toda resposta fundamentada em protocolo cita a fonte (`[Fonte N: arquivo.txt]`) —
  explainability.
- Cada interação é registrada em log estruturado (`logs/audit.jsonl`, gerado em tempo de
  execução) com pergunta, paciente, fontes citadas, alertas disparados e flags de guardrail.

Detalhes completos em [`reports/relatorio_tecnico.md`](reports/relatorio_tecnico.md) e
[`docs/fluxo_langgraph.md`](docs/fluxo_langgraph.md).

## Base de referência

Este projeto foi construído a partir de material de estudo da DSA Academy (fine-tuning de LLM,
RAG com LangChain e indexação vetorial), adaptado e estendido para o hospital fictício e os
requisitos específicos deste Tech Challenge (LangGraph, guardrails de segurança, auditoria e
consulta a prontuários estruturados).
