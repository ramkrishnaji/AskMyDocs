"""
create_db.py

Standalone ingestion script.
Loads a PDF, splits it into chunks, embeds them with Mistral,
and persists the result in a Chroma vector store.

Usage (CLI):
    python create_db.py path/to/file.pdf --persist_dir chroma_db

Usage (import):
    from create_db import ingest_pdf
    vector_store, num_chunks = ingest_pdf("file.pdf", "chroma_db")
"""

import os
import argparse
import shutil

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()


def ingest_pdf(pdf_path: str, persist_dir: str = "chroma_db", overwrite: bool = True, api_key: str = None):
    """
    Load a PDF, chunk it, embed the chunks, and persist them to a Chroma store.

    Args:
        pdf_path: path to the source PDF file.
        persist_dir: directory where the Chroma DB will be written.
        overwrite: if True, wipes any existing DB at persist_dir first.
        api_key: Mistral API key. Falls back to MISTRAL_API_KEY env var if omitted.

    Returns:
        (vector_store, num_chunks)
    """
    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY not set. Add it to your .env file or pass api_key explicitly.")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No such file: {pdf_path}")

    # Load PDF
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
    except Exception as e:
        raise ValueError(f"Could not read PDF (it may be corrupted or password-protected): {e}")

    if not docs:
        raise ValueError("The PDF appears to have no readable pages.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if not chunks:
        # e.g. a scanned/image-only PDF with no extractable text
        return None, 0

    # Create embeddings
    embedding_model = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

    # Fresh DB each time so chunks from different PDFs don't mix
    if overwrite and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    # Store embeddings in Chroma
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )

    return vector_store, len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into a Chroma vector store.")
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest")
    parser.add_argument(
        "--persist_dir", default="chroma_db", help="Directory to persist the Chroma DB (default: chroma_db)"
    )
    args = parser.parse_args()

    _, n_chunks = ingest_pdf(args.pdf_path, args.persist_dir)
    print(f"Number of chunks: {n_chunks}")
    print(f"Vector database created successfully at '{args.persist_dir}'!")
