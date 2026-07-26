"""
RAG Context Retrieval Module.

Provides search and retrieval interfaces for querying the ChromaDB vector database 
to supply relevant interview guidelines, ideal answer patterns, and domain context 
to agents.
"""

def retrieve_context(query: str, top_k: int = 3, db_path: str = "chroma_db") -> list:
    """
    Retrieve top-k relevant knowledge snippets from ChromaDB matching the query.
    
    :param query: Search query (e.g., question text or topic keywords).
    :param top_k: Number of relevant context chunks to retrieve.
    :param db_path: Path to persistent ChromaDB storage.
    :return: List of retrieved context strings or document metadata.
    """
    pass
