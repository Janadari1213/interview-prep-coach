"""
RAG Context Retrieval Module for Interview Preparation Coach.

Provides search and retrieval interfaces for querying the ChromaDB vector database 
to supply relevant interview guidelines, ideal answer patterns, and domain context.
"""

import chromadb
from chromadb.utils import embedding_functions


def get_relevant_chunks(query: str, top_k: int = 4, db_path: str = "./chroma_db", collection_name: str = "interview_knowledge") -> list:
    """
    Retrieve the top_k most similar knowledge chunks from ChromaDB for a given query.
    
    :param query: Natural language search query or interview question.
    :param top_k: Number of top matching chunks to return.
    :param db_path: Path to persistent ChromaDB directory.
    :param collection_name: Name of ChromaDB collection.
    :return: List of dicts containing 'text' and 'source' metadata.
    """
    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )
    except Exception:
        # Fallback if collection does not exist yet
        return []

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    chunks = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)
        for doc, meta in zip(docs, metas):
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown")
            })
    return chunks


def retrieve_context(query: str, top_k: int = 3, db_path: str = "./chroma_db") -> list:
    """
    Backward-compatible wrapper for retrieve_context.
    """
    return get_relevant_chunks(query, top_k=top_k, db_path=db_path)
