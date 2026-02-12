"""Settings and configuration for RAG application."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""

    # Azure OpenAI Configuration
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://your-resource.openai.azure.com/")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY", "your-api-key")
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")

    # OpenAI Configuration (alternative to Azure)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    USE_AZURE = os.getenv("USE_AZURE", "true").lower() == "true"

    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    # FAISS Configuration
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/indexes/faiss_index")
    FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "./data/indexes/metadata.json")

    # Document Configuration
    DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "./data/documents/")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval Configuration
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()
