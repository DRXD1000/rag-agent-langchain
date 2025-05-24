"""Qdrant Hybrid Retriever."""

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from rag_agent.llms.models import init_sparse_embed_bm25


def init_qdrant_hybrid_retriever(client: QdrantClient, collection_name: str) -> QdrantVectorStore:
    """Initialize Qdrant Hybrid Search with BM25."""
    if collection_name not in client.get_collections():
        msg = f"Collection Name {collection_name} does not exist in current client. Please first create a collection."
        raise ValueError(msg)

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        sparse_embedding=init_sparse_embed_bm25(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
