"""Indexação dos protocolos internos do hospital em um banco vetorial FAISS.

Usa LangChain para carregar os documentos (.txt/.pdf), dividir em chunks e
gerar embeddings com um modelo HuggingFace local (sem custo de API). O índice
é salvo em disco para ser reaberto pelo retriever sem reprocessar tudo.
"""

import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_DIR = os.path.join(BASE_DIR, "data", "raw", "protocolos_hospitalares")
INDEX_DIR = os.path.join(BASE_DIR, "data", "processed", "faiss_index")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(docs_dir: str = DOCS_DIR):
    """Carrega todos os .txt do diretório de protocolos, com metadado de fonte."""
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    for doc in docs:
        # Mantém apenas o nome do arquivo como referência de fonte, para citação
        # legível nas respostas do assistente (explainability).
        doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "desconhecido"))
    return docs


def build_index(docs_dir: str = DOCS_DIR, index_dir: str = INDEX_DIR) -> FAISS:
    docs = load_documents(docs_dir)
    if not docs:
        raise ValueError(f"Nenhum documento .txt encontrado em {docs_dir}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"Índice FAISS criado com {len(chunks)} chunks a partir de {len(docs)} documentos.")
    print(f"Índice salvo em: {index_dir}")

    return vectorstore


def load_index(index_dir: str = INDEX_DIR) -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    build_index()
