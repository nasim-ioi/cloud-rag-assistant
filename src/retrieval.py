"""Retrieval module for RAG pipeline."""
import logging
from typing import List, Tuple
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain_core.documents import Document

from config.settings import settings
from src.ingest import DocumentIngester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """Handles document retrieval from FAISS index."""

    def __init__(self):
        """Initialize the retriever with FAISS index."""
        self.ingester = DocumentIngester()
        self.vector_store = None
        self._load_index()

    def _load_index(self):
        """Load FAISS index from disk."""
        if self.vector_store is None:
            self.vector_store = self.ingester.load_index()
            if self.vector_store is None:
                logger.warning("No FAISS index found. Please run ingestion first.")

    def retrieve_documents(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents for a query."""
        if self.vector_store is None:
            logger.error("Vector store not initialized")
            return []

        if k is None:
            k = settings.TOP_K_RESULTS

        logger.info(f"Retrieving {k} documents for query: {query}")

        try:
            # Similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Retrieved {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    def retrieve_and_format(self, query: str, k: int = None) -> str:
        """Retrieve documents and format as context string."""
        results = self.retrieve_documents(query, k)

        if not results:
            return "No relevant documents found."

        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(
                f"Document {i} (Relevance: {score:.2f}):\n{doc.page_content}\n"
            )

        return "\n".join(context_parts)

    def get_retriever(self):
        """Get LangChain retriever object."""
        if self.vector_store is None:
            logger.error("Vector store not initialized")
            return None

        return self.vector_store.as_retriever(
            search_kwargs={"k": settings.TOP_K_RESULTS}
        )


def main():
    """Example usage of Retriever."""
    retriever = Retriever()

    # Example queries
    queries = [
        "How do I set up Azure OpenAI?",
        "What is RAG?",
        "How do I use FAISS for vector search?",
    ]

    for query in queries:
        logger.info(f"\nQuery: {query}")
        context = retriever.retrieve_and_format(query)
        print(context)


if __name__ == "__main__":
    main()
