"""Models to be used inside the graph."""

from typing import Any

from FlagEmbedding import FlagReranker
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse


def initialize_ollama_llm(model_name: str, temperature: float, **kwargs: Any) -> ChatOllama:  # noqa: ANN401 #type-ignore: [no-any-unimported]
    """Initialize a local llm with ollama."""
    return ChatOllama(model=model_name, temperature=temperature, **kwargs)


def initialize_embedding_model(model_name: str = "bge-m3:567m") -> OllamaEmbeddings:
    """Initialize Embedding Model."""
    return OllamaEmbeddings(model=model_name)


def init_sparse_embed_bm25() -> FastEmbedSparse:
    """Initialize BM25 Spare Vectors."""
    return FastEmbedSparse(model_name="Qdrant/bm25")


def init_reranker(model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True, normalize: bool = False) -> FlagReranker:
    """Initialize Reranking Model."""
    return FlagReranker(model_name, use_fp16=use_fp16, normalize=normalize)
