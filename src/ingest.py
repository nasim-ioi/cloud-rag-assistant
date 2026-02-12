"""Document ingestion module for RAG pipeline."""
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngester:
    """Handles document ingestion, chunking, embedding, and FAISS index creation."""

    def __init__(self):
        """Initialize the ingester with embeddings and text splitter."""
        self.embeddings = self._init_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        self.vector_store = None
        self.index_path = settings.FAISS_INDEX_PATH
        self.metadata_path = settings.FAISS_METADATA_PATH

    def _init_embeddings(self):
        """Initialize embeddings based on configuration."""
        if settings.USE_AZURE:
            logger.info("Initializing Azure OpenAI embeddings...")
            return AzureOpenAIEmbeddings(
                api_key=settings.AZURE_API_KEY,
                azure_endpoint=settings.AZURE_ENDPOINT,
                deployment_id="text-embedding-3-small",
                api_version=settings.AZURE_API_VERSION,
            )
        else:
            logger.info("Initializing OpenAI embeddings...")
            return OpenAIEmbeddings(
                api_key=settings.OPENAI_API_KEY,
                model=settings.EMBEDDING_MODEL,
            )

    def load_pdf_documents(self, file_paths: List[str] = None) -> List[Document]:
        """Load PDF documents from specified paths or document directory."""
        documents = []

        if file_paths is None:
            # Load all PDFs from document directory
            doc_path = Path(settings.DOCUMENT_PATH)
            if not doc_path.exists():
                logger.warning(f"Document path does not exist: {doc_path}")
                return documents
            file_paths = list(doc_path.glob("*.pdf"))

        for file_path in file_paths:
            try:
                logger.info(f"Loading PDF: {file_path}")
                loader = PyPDFLoader(str(file_path))
                pdf_documents = loader.load()
                documents.extend(pdf_documents)
                logger.info(f"Loaded {len(pdf_documents)} pages from {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into manageable chunks."""
        logger.info(f"Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def create_faiss_index(self, documents: List[Document], save_locally: bool = True) -> FAISS:
        """Create FAISS vector store from documents."""
        logger.info("Creating FAISS index...")

        # Create FAISS index
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

        if save_locally:
            self.save_index()

        logger.info("FAISS index created successfully")
        return self.vector_store

    def save_index(self):
        """Save FAISS index to disk."""
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return

        # Create directories if they don't exist
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving FAISS index to {self.index_path}")
        self.vector_store.save_local(self.index_path)
        logger.info("FAISS index saved successfully")

    def load_index(self) -> FAISS:
        """Load FAISS index from disk."""
        if not Path(self.index_path).exists():
            logger.error(f"Index not found at {self.index_path}")
            return None

        logger.info(f"Loading FAISS index from {self.index_path}")
        self.vector_store = FAISS.load_local(
            self.index_path, self.embeddings, allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully")
        return self.vector_store

    def ingest_and_index(self, file_paths: List[str] = None) -> FAISS:
        """Complete ingestion pipeline: load → chunk → embed → index."""
        logger.info("Starting document ingestion pipeline...")

        # Load documents
        documents = self.load_pdf_documents(file_paths)
        if not documents:
            logger.error("No documents loaded")
            return None

        # Chunk documents
        chunks = self.chunk_documents(documents)
        if not chunks:
            logger.error("No chunks created")
            return None

        # Create and save FAISS index
        vector_store = self.create_faiss_index(chunks, save_locally=True)

        logger.info("Document ingestion pipeline completed successfully")
        return vector_store

    def update_index(self, file_paths: List[str]) -> FAISS:
        """Add new documents to existing index."""
        logger.info("Updating existing index with new documents...")

        # Load existing index
        if not self.load_index():
            logger.warning("No existing index found, creating new one")
            return self.ingest_and_index(file_paths)

        # Load and process new documents
        documents = self.load_pdf_documents(file_paths)
        chunks = self.chunk_documents(documents)

        # Add to existing index
        logger.info(f"Adding {len(chunks)} new chunks to index...")
        self.vector_store.add_documents(chunks)
        self.save_index()

        logger.info("Index updated successfully")
        return self.vector_store


def main():
    """Example usage of DocumentIngester."""
    ingester = DocumentIngester()

    # Check if index exists
    index_path = Path(ingester.index_path)
    if index_path.exists():
        logger.info("Loading existing index...")
        ingester.load_index()
    else:
        logger.info("Creating new index...")
        # Place your PDF files in data/documents/ directory
        ingester.ingest_and_index()


if __name__ == "__main__":
    main()
