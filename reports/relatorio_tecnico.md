# Relatório Técnico — Assistente Virtual Médico (Hospital Vida Plena)

**Tech Challenge — Fase 3 | Pós-graduação em IA para Devs (FIAP)**

> Todos os dados clínicos, protocolos, pacientes e o próprio "Hospital Vida Plena" usados neste
> projeto são **fictícios**, criados exclusivamente para fins didáticos. Nenhum dado real de
> paciente foi utilizado.

## 1. Objetivo

Desenvolver um assistente virtual médico treinado com dados internos do hospital, capaz de
auxiliar condutas clínicas, responder dúvidas de médicos e sugerir procedimentos com base em
protocolos internos — com fluxos de decisão automatizados e seguros, coordenados via LangChain
e LangGraph.

## 2. Processo de fine-tuning

- **Modelo base:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (aberto, sem gating no Hugging Face),
  compatível com fine-tuning em GPU T4 (Colab) e Apple Silicon (MPS).
- **Técnica:** LoRA (Low-Rank Adaptation) via `peft`, ajustando apenas uma fração dos parâmetros
  (rank 8, alpha 16), reduzindo custo computacional e memória necessária.
- **Dados de treino:** combinação de
  1. amostra no estilo MedQuAD/PubMedQA (ou o MedQuAD real, se baixado — ver README);
  2. perguntas frequentes de médicos extraídas do FAQ interno do hospital, fundamentadas nos
     protocolos clínicos (sepse, crise hipertensiva, dengue, insulinoterapia).
- **Preparo dos dados:** curadoria (remoção de duplicatas e respostas muito curtas) e
  anonimização (CPF, RG, telefone, e-mail, data de nascimento, número de prontuário e nomes
  próprios em campos identificados) — ver `src/preprocessing/`.
- **Pipeline:** `src/finetuning/train.py` — script único, parametrizável, reutilizado tanto para
  o smoke test local quanto para o treinamento completo no notebook Colab
  (`notebooks/tech_challenge_fase3_colab.ipynb`).

_Preencher após a execução completa no Colab:_ número final de exemplos de treino, tempo de
treinamento, loss final.

## 3. Descrição do assistente médico

O assistente combina três camadas:

1. **RAG sobre protocolos internos** (LangChain + FAISS): cada protocolo é dividido em chunks,
   convertido em embeddings locais (`sentence-transformers/all-MiniLM-L6-v2`) e indexado.
   Toda resposta fundamentada em protocolo cita a fonte no formato `[Fonte N: arquivo.txt]`.
2. **Base estruturada de prontuários** (mock): simula a consulta a um sistema de prontuário
   eletrônico, fornecendo sinais vitais, exames pendentes/resultados, hipótese diagnóstica,
   comorbidades e alergias do paciente.
3. **LLM fine-tuned**: gera a resposta final combinando pergunta do médico + resumo do
   prontuário + alertas ativos + trechos de protocolo recuperados.

## 4. Fluxo de decisão automatizado (LangGraph)

Ver diagrama completo em [`docs/fluxo_langgraph.md`](../docs/fluxo_langgraph.md).

O grafo coordena seis etapas: `carregar_prontuario → verificar_exames_e_alertas →
consultar_rag → gerar_resposta → aplicar_guardrails → registrar_auditoria`.

Alertas críticos (ex.: qSOFA ≥ 2, lactato ≥ 4 mmol/L, glicemia < 70 mg/dL, PA ≥ 180x120 mmHg)
são calculados programaticamente a partir dos limiares definidos nos próprios protocolos
internos (`src/agent/clinical_rules.py`), garantindo que o disparo do alerta não dependa da
interpretação do LLM.

## 5. Segurança e validação

- **Limites de atuação:** o assistente nunca prescreve diretamente. Todo resultado passa pelo
  guardrail (`src/security/guardrails.py`), que detecta linguagem de prescrição direta
  (ex.: "tome X mg") e sempre anexa o disclaimer de validação humana obrigatória.
- **Bloqueio de tópicos:** perguntas fora do escopo clínico seguro (ex.: dose letal) são
  bloqueadas antes de chegar ao médico.
- **Explainability:** respostas fundamentadas em protocolo devem citar a fonte
  (`[Fonte N: arquivo]`); a ausência de citação é sinalizada como flag de guardrail
  (`resposta_sem_citacao_de_fonte_explicita`).
- **Auditoria:** cada interação é registrada em log estruturado JSON Lines
  (`src/security/audit_log.py`) com timestamp, paciente, pergunta, fontes citadas, alertas
  disparados e flags de guardrail acionadas.

## 6. Avaliação do modelo e análise dos resultados

Métrica utilizada: **ROUGE-L (F1)** entre a resposta gerada e a resposta de referência do
protocolo/FAQ interno, comparando o modelo base (sem fine-tuning) com o modelo ajustado
(`src/finetuning/evaluate.py`).

_Preencher após a execução completa no Colab (célula "Avaliação" do notebook):_

| Métrica | Modelo base | Modelo fine-tuned |
|---|---|---|
| ROUGE-L médio (F1) | — | — |
| Nº de amostras avaliadas | — | — |

**Análise qualitativa:** _(preencher com observações sobre a diferença de aderência às
respostas de referência, uso correto da terminologia dos protocolos internos, e eventuais casos
em que o modelo fine-tuned citou informações mais alinhadas ao Hospital Vida Plena do que o
modelo base.)_

## 7. Limitações e próximos passos

- O dataset de fine-tuning usado neste repositório é uma amostra didática; um hospital real
  precisaria de uma base de treino ordens de magnitude maior, com validação clínica formal.
- A base de prontuários é um mock local; em produção seria substituída por integração real com
  o sistema de prontuário eletrônico (HL7/FHIR).
- As regras de alerta (`clinical_rules.py`) cobrem um subconjunto didático dos protocolos; um
  hospital real manteria essas regras versionadas e validadas pelo corpo clínico.
- Toda resposta do assistente continua sujeita a validação humana obrigatória antes de qualquer
  conduta — este projeto não deve ser usado como ferramenta clínica real.
