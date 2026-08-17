"""Cliente do LLM usado pelo assistente médico dentro do fluxo LangGraph.

Suporta dois modos:
  1) Modelo real (fine-tuned ou base) via um `generator` compatível com a API
     `transformers.pipeline("text-generation")` — usado no notebook Colab
     após o fine-tuning.
  2) Fallback extrativo (sem GPU/download), que compõe uma resposta
     diretamente a partir dos trechos de protocolo recuperados pelo RAG.
     Útil para testar o grafo completo sem depender de um modelo pesado.
"""

from typing import Callable, Optional


def extractive_fallback(contexto_chunks: list[dict], pergunta: str) -> str:
    if not contexto_chunks:
        return (
            "Não encontrei nenhum trecho de protocolo interno relevante para "
            "esta pergunta. Recomendo consultar a equipe médica responsável."
        )

    partes = [f"Com base nos protocolos internos, em resposta a: \"{pergunta}\""]
    for i, chunk in enumerate(contexto_chunks):
        trecho = chunk["content"].strip().replace("\n", " ")
        if len(trecho) > 280:
            trecho = trecho[:280].rsplit(" ", 1)[0] + "..."
        partes.append(f"[Fonte {i}: {chunk['source']}] {trecho}")

    return "\n\n".join(partes)


def generate_response(
    prompt: str,
    contexto_chunks: list[dict],
    pergunta: str,
    generator: Optional[Callable[[str], str]] = None,
) -> str:
    """Gera a resposta do assistente.

    `generator`, se fornecido, deve ser uma função que recebe o prompt
    completo (str) e retorna o texto gerado (str) — por exemplo, um wrapper
    em torno de `pipeline("text-generation", model=..., tokenizer=...)`
    usando o modelo fine-tuned carregado no notebook.
    """
    if generator is not None:
        return generator(prompt)

    return extractive_fallback(contexto_chunks, pergunta)


def make_hf_pipeline_generator(pipe, max_new_tokens: int = 200) -> Callable[[str], str]:
    """Constrói um `generator` a partir de um `transformers.pipeline` já carregado."""

    def _generate(prompt: str) -> str:
        output = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return output[0]["generated_text"].strip()

    return _generate
