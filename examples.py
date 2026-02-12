"""Example usage of the RAG system."""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_ingest():
    """Example 1: Ingest documents."""
    from src.ingest import DocumentIngester

    logger.info("\n=== Example 1: Document Ingestion ===")

    ingester = DocumentIngester()

    # Check if documents exist
    doc_path = Path("data/documents")
    pdf_files = list(doc_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in data/documents/. Please add some PDF files first.")
        return

    # Ingest and index
    logger.info(f"Found {len(pdf_files)} PDF files. Starting ingestion...")
    vector_store = ingester.ingest_and_index()

    if vector_store:
        logger.info("Document ingestion completed successfully!")


def example_2_retrieve():
    """Example 2: Retrieve documents."""
    from src.retrieval import Retriever

    logger.info("\n=== Example 2: Document Retrieval ===")

    retriever = Retriever()

    if retriever.vector_store is None:
        logger.error("No index found. Please run example_1_ingest() first.")
        return

    # Example queries
    queries = [
        "How do I set up Azure OpenAI?",
        "What is RAG and how does it work?",
        "How do I use FAISS for vector search?",
    ]

    for query in queries:
        logger.info(f"\nQuery: {query}")
        results = retriever.retrieve_documents(query, k=2)

        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"  Result {i} (Score: {score:.2f})")
            logger.info(f"  Content: {doc.page_content[:100]}...")


def example_3_qa_chain():
    """Example 3: QA chain with LLM."""
    from src.retrieval import Retriever
    from src.llm_config import llm_config

    logger.info("\n=== Example 3: QA Chain ===")

    retriever = Retriever()

    if retriever.vector_store is None:
        logger.error("No index found. Please run example_1_ingest() first.")
        return

    try:
        qa_chain = llm_config.create_qa_chain(retriever.get_retriever())

        query = "What is cloud computing?"
        logger.info(f"Query: {query}")

        response = qa_chain.invoke({"query": query})
        logger.info(f"Answer: {response['result']}")

    except Exception as e:
        logger.error(f"Error in QA chain: {str(e)}")
        logger.info("Make sure your API keys are configured in .env")


def example_4_prompt_chain():
    """Example 4: Prompt chain using LCEL."""
    from src.retrieval import Retriever
    from src.llm_config import llm_config

    logger.info("\n=== Example 4: Prompt Chain (LCEL) ===")

    retriever = Retriever()

    if retriever.vector_store is None:
        logger.error("No index found. Please run example_1_ingest() first.")
        return

    try:
        chain = llm_config.create_prompt_chain(retriever.get_retriever())

        query = "Explain the benefits of cloud storage."
        logger.info(f"Query: {query}")

        answer = chain.invoke(query)
        logger.info(f"Answer: {answer}")

    except Exception as e:
        logger.error(f"Error in prompt chain: {str(e)}")
        logger.info("Make sure your API keys are configured in .env")


def main():
    """Run all examples."""
    logger.info("Starting RAG System Examples...")

    # Uncomment the examples you want to run

    # Example 1: Ingest documents
    example_1_ingest()

    # Example 2: Retrieve documents
    example_2_retrieve()

    # Example 3: QA Chain
    # example_3_qa_chain()

    # Example 4: Prompt Chain
    # example_4_prompt_chain()

    logger.info("\nExamples completed!")


if __name__ == "__main__":
    main()
