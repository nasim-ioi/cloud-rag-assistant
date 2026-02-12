"""LLM configuration and initialization module."""
import logging
from typing import Optional

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.language_model import BaseLanguageModel

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMConfig:
    """Manages LLM initialization and configuration."""

    def __init__(self):
        """Initialize LLM configuration."""
        self.llm = None
        self._init_llm()

    def _init_llm(self) -> BaseLanguageModel:
        """Initialize LLM based on configuration."""
        if settings.USE_AZURE:
            logger.info("Initializing Azure OpenAI LLM...")
            self.llm = AzureChatOpenAI(
                api_key=settings.AZURE_API_KEY,
                azure_endpoint=settings.AZURE_ENDPOINT,
                deployment_name=settings.AZURE_DEPLOYMENT_NAME,
                api_version=settings.AZURE_API_VERSION,
                temperature=0.7,
                max_tokens=1000,
            )
        else:
            logger.info("Initializing OpenAI LLM...")
            self.llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model=settings.MODEL_NAME,
                temperature=0.7,
                max_tokens=1000,
            )

        return self.llm

    def get_llm(self) -> BaseLanguageModel:
        """Get configured LLM instance."""
        if self.llm is None:
            self._init_llm()
        return self.llm

    def create_qa_chain(self, retriever):
        """Create QA chain with retriever and LLM."""
        from langchain.chains import RetrievalQA

        logger.info("Creating QA chain...")

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.get_llm(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        return qa_chain

    def create_prompt_chain(self, retriever):
        """Create prompt chain using LCEL."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        logger.info("Creating prompt chain...")

        template = """You are an AI assistant specialized in cloud documentation and technologies.
Use the following context to answer the user's question accurately.

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.get_llm()
            | StrOutputParser()
        )

        return chain


# Global LLM config instance
llm_config = LLMConfig()


def get_llm() -> BaseLanguageModel:
    """Get configured LLM instance."""
    return llm_config.get_llm()


def main():
    """Test LLM configuration."""
    config = LLMConfig()
    llm = config.get_llm()
    logger.info("LLM initialized successfully")
    logger.info(f"LLM: {llm}")


if __name__ == "__main__":
    main()
