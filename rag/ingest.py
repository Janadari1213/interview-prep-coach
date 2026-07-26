"""
RAG Document Ingestion Module for Interview Preparation Coach.

Loads PDF and .txt knowledge documents from data/knowledge_base/,
splits them into character chunks using LangChain's RecursiveCharacterTextSplitter,
embeds chunks using Sentence Transformers (all-MiniLM-L6-v2), and persists
embeddings and metadata into ChromaDB.
"""

import os
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf


def load_knowledge_documents(data_dir: str = "data/knowledge_base") -> List[Document]:
    """
    Load all .txt and .pdf files from the target knowledge base directory.
    
    :param data_dir: Path to directory containing source documents.
    :return: List of loaded LangChain Document objects with source metadata.
    """
    documents: List[Document] = []
    base_path = Path(data_dir)

    if not base_path.exists():
        print(f"[Warning] Knowledge base directory '{data_dir}' not found.", flush=True)
        return documents

    for file_path in base_path.rglob("*"):
        if not file_path.is_file():
            continue

        filename = file_path.name
        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
                if text_content.strip():
                    doc = Document(page_content=text_content, metadata={"source": filename})
                    documents.append(doc)
                    print(f"[Ingest] Loaded TXT file: {filename}", flush=True)
            except Exception as e:
                print(f"[Error] Failed to load TXT file {filename}: {e}", flush=True)

        elif suffix == ".pdf":
            try:
                reader = pypdf.PdfReader(str(file_path))
                pdf_text = []
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        pdf_text.append(extracted)
                full_pdf_text = "\n".join(pdf_text)
                if full_pdf_text.strip():
                    doc = Document(page_content=full_pdf_text, metadata={"source": filename})
                    documents.append(doc)
                    print(f"[Ingest] Loaded PDF file: {filename}", flush=True)
            except Exception as e:
                print(f"[Error] Failed to load PDF file {filename}: {e}", flush=True)

    print(f"[Ingest] Total raw documents loaded: {len(documents)}", flush=True)
    return documents


def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Split raw documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    :param documents: List of raw loaded Document objects.
    :param chunk_size: Target maximum characters per chunk (~500).
    :param chunk_overlap: Overlapping characters between adjacent chunks (~50).
    :return: List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"[Ingest] Split {len(documents)} document(s) into {len(chunks)} chunk(s).", flush=True)
    return chunks


def store_in_chromadb(chunks: List[Document], db_path: str = "./chroma_db", collection_name: str = "interview_knowledge") -> None:
    """
    Generate embeddings using sentence-transformers (all-MiniLM-L6-v2) and store 
    chunks with metadata into a persistent ChromaDB collection.
    
    :param chunks: List of chunked Document objects to embed and store.
    :param db_path: Directory path for persistent ChromaDB storage.
    :param collection_name: Name of the ChromaDB collection.
    """
    if not chunks:
        print("[Ingest] No document chunks available to store.", flush=True)
        return

    print(f"[Ingest] Initializing Sentence Transformer embedding model ('all-MiniLM-L6-v2')...", flush=True)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    documents_text = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk_{i:04d}" for i in range(len(chunks))]

    collection.upsert(
        documents=documents_text,
        metadatas=metadatas,
        ids=ids
    )

    print(f"[Ingest] Successfully ingested {len(chunks)} chunks into ChromaDB at '{db_path}' (collection: '{collection_name}').", flush=True)


def ingest_documents(data_dir: str = "data/knowledge_base", db_path: str = "./chroma_db") -> None:
    """
    End-to-end ingestion pipeline runner.
    """
    docs = load_knowledge_documents(data_dir)
    if not docs:
        print("[Ingest] No documents found. Ingestion skipped.", flush=True)
        return
    chunks = split_documents(docs)
    store_in_chromadb(chunks, db_path)


if __name__ == "__main__":
    ingest_documents()
