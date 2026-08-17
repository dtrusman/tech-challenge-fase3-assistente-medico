"""Retriever de protocolos internos com citação de fonte (explainability)."""

import os

try:
    from .indexing import INDEX_DIR, build_index, load_index
except ImportError:
    from indexing import INDEX_DIR, build_index, load_index


def get_or_build_retriever(index_dir: str = INDEX_DIR, k: int = 4):
    if os.path.isdir(index_dir):
        vectorstore = load_index(index_dir)
    else:
        vectorstore = build_index(index_dir=index_dir)
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})


def retrieve_with_sources(query: str, retriever=None) -> list[dict]:
    """Retorna trechos relevantes dos protocolos, cada um com sua fonte (arquivo)."""
    retriever = retriever or get_or_build_retriever()
    docs = retriever.invoke(query)
    return [
        {"content": doc.page_content, "source": doc.metadata.get("source", "desconhecido")}
        for doc in docs
    ]


def format_context_with_citations(chunks: list[dict]) -> str:
    """Formata os trechos recuperados com um identificador de fonte citável."""
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"[Fonte {i}: {chunk['source']}]\n{chunk['content']}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    resultado = retrieve_with_sources("O que fazer em caso de lactato alto na sepse?")
    print(format_context_with_citations(resultado))
