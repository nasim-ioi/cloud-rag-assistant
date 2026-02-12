"""FastAPI application for RAG service."""
import logging
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config.settings import settings
from src.ingest import DocumentIngester
from src.retrieval import Retriever
from src.llm_config import llm_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
ingester: DocumentIngester = None
retriever: Retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global ingester, retriever
    logger.info("Starting RAG application...")
    ingester = DocumentIngester()
    retriever = Retriever()
    yield
    logger.info("Shutting down RAG application...")


# Initialize FastAPI app
app = FastAPI(
    title="Cloud RAG Assistant",
    description="A RAG-based AI assistant for cloud documentation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    """Query request model."""

    query: str
    top_k: int = settings.TOP_K_RESULTS


class QueryResponse(BaseModel):
    """Query response model."""

    query: str
    answer: str
    context: List[str]
    sources: List[dict]


class IngestRequest(BaseModel):
    """Ingest request model."""

    file_names: List[str]


class IngestResponse(BaseModel):
    """Ingest response model."""

    status: str
    message: str
    documents_processed: int


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Cloud RAG Assistant",
        "version": "1.0.0",
    }


# Retrieval endpoints
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    if retriever.vector_store is None:
        raise HTTPException(status_code=400, detail="No documents indexed. Please ingest documents first.")

    try:
        # Retrieve relevant documents
        results = retriever.retrieve_documents(request.query, k=request.top_k)

        if not results:
            return QueryResponse(
                query=request.query,
                answer="No relevant documents found.",
                context=[],
                sources=[],
            )

        # Extract context and sources
        context_parts = []
        sources = []

        for doc, score in results:
            context_parts.append(doc.page_content)
            sources.append(
                {
                    "content": doc.page_content[:200],
                    "score": float(score),
                    "metadata": doc.metadata,
                }
            )

        # Generate answer using LLM
        try:
            qa_chain = llm_config.create_qa_chain(retriever.get_retriever())
            response = qa_chain.invoke({"query": request.query})
            answer = response.get("result", "Could not generate answer")
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = "Error generating answer from context."

        return QueryResponse(
            query=request.query,
            answer=answer,
            context=context_parts,
            sources=sources,
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrieve")
async def retrieve(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(settings.TOP_K_RESULTS, description="Number of results to retrieve"),
):
    """Retrieve relevant documents without generating an answer."""
    if retriever.vector_store is None:
        raise HTTPException(status_code=400, detail="No documents indexed.")

    try:
        results = retriever.retrieve_documents(query, k=top_k)

        documents = [
            {
                "content": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
            }
            for doc, score in results
        ]

        return {
            "query": query,
            "documents": documents,
            "count": len(documents),
        }

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Ingestion endpoints
@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    """Ingest documents from the documents folder."""
    try:
        vector_store = ingester.ingest_and_index()

        if vector_store is None:
            raise HTTPException(status_code=400, detail="No documents found to ingest.")

        return IngestResponse(
            status="success",
            message="Documents ingested and indexed successfully.",
            documents_processed=len(vector_store.docstore._dict),
        )

    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and ingest PDF documents."""
    try:
        import tempfile
        from pathlib import Path

        uploaded_files = []

        for file in files:
            if not file.filename.endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF.")

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file.flush()
                uploaded_files.append(tmp_file.name)

        # Ingest uploaded files
        vector_store = ingester.update_index(uploaded_files)

        # Clean up temporary files
        for tmp_file in uploaded_files:
            Path(tmp_file).unlink()

        return {
            "status": "success",
            "message": f"Uploaded {len(files)} documents successfully.",
            "files": [f.filename for f in files],
        }

    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    """Get application status including index information."""
    try:
        index_size = len(retriever.vector_store.docstore._dict) if retriever.vector_store else 0

        return {
            "service": "Cloud RAG Assistant",
            "status": "running",
            "index_status": "loaded" if retriever.vector_store else "not loaded",
            "documents_indexed": index_size,
            "llm_provider": "Azure OpenAI" if settings.USE_AZURE else "OpenAI",
            "model": settings.AZURE_DEPLOYMENT_NAME if settings.USE_AZURE else settings.MODEL_NAME,
        }

    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
