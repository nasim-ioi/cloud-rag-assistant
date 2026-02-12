"""Main entry point for the RAG application."""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the application."""
    logger.info("Cloud RAG Assistant Starting...")

    try:
        # Import after logging setup
        from src.app import app
        import uvicorn
        from config.settings import settings

        logger.info(f"Starting FastAPI server on {settings.API_HOST}:{settings.API_PORT}")
        logger.info(f"Using {'Azure' if settings.USE_AZURE else 'OpenAI'} for LLM and embeddings")

        uvicorn.run(
            "src.app:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=settings.DEBUG,
            log_level="info",
        )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
