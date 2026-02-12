# Cloud RAG Assistant

A modular Retrieval-Augmented Generation (RAG) AI assistant for cloud documentation, built with LangChain, FAISS, and OpenAI/Azure OpenAI. Easily ingest, index, and query your technical documents via a FastAPI service.

---

## 🚀 Features

- **Document Ingestion**: Load and process PDF documents
- **Semantic Chunking**: Split documents into overlapping chunks
- **Vector Embeddings**: Generate embeddings using OpenAI or Azure OpenAI
- **FAISS Indexing**: Store and retrieve documents efficiently
- **RAG Pipeline**: Combine retrieval with LLM for contextual answers
- **FastAPI Service**: RESTful API for querying and document management
- **Modular Architecture**: Clean, extensible codebase

---

## 🗂️ Project Structure

```
cloud-rag-assistant/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration management
├── src/
│   ├── __init__.py
│   ├── ingest.py            # Document ingestion and indexing
│   ├── retrieval.py         # Document retrieval
│   ├── llm_config.py        # LLM configuration
│   └── app.py               # FastAPI application
├── data/
│   ├── documents/           # Place PDF files here
│   └── indexes/             # FAISS index storage
├── requirements.txt
├── .env.example
├── README.md
├── main.py
└── examples.py
```

---

## ⚡ Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update with your credentials:

```bash
cp .env.example .env
# Edit .env with your Azure/OpenAI keys
```

**For Azure OpenAI:**
```env
USE_AZURE=true
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-api-key
AZURE_DEPLOYMENT_NAME=gpt-4o-mini
```

**For OpenAI:**
```env
USE_AZURE=false
OPENAI_API_KEY=your-openai-key
MODEL_NAME=gpt-4o-mini
```

### 3. Add Documents

Place your PDF files in the `data/documents/` directory.

### 4. Ingest Documents

```bash
python -m src.ingest
```

This will:
- Load all PDFs from `data/documents/`
- Split them into chunks
- Generate embeddings
- Create and save FAISS index

### 5. Start the API

```bash
python main.py
```

The API will be available at `http://localhost:8000`

---

## 🛠️ Configuration

Key settings in `config/settings.py`:

- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K_RESULTS`: Number of retrieved documents (default: 3)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)
- `API_PORT`: API port (default: 8000)

---

## 🧩 Modular Workflow

1. **Ingest Phase**:
   - PDF files are loaded using LangChain
   - Documents are split into chunks
   - Chunks are embedded using OpenAI/Azure OpenAI
   - Embeddings are stored in FAISS

2. **Query Phase**:
   - User query is embedded
   - FAISS finds most similar documents
   - Retrieved documents are formatted as context
   - LLM generates answer using context

3. **Response Phase**:
   - Answer and source documents are returned
   - Relevance scores included for transparency

---

## 🧪 Usage Examples

### Python Script
```python
from src.retrieval import Retriever
from src.llm_config import llm_config

retriever = Retriever()
prompt_chain = llm_config.create_prompt_chain(retriever.get_retriever())

query = "What is cloud computing?"
answer = prompt_chain.invoke(query)
print(answer)
```

### Using curl
```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I use FAISS?", "top_k": 3}'

# Retrieve
curl "http://localhost:8000/retrieve?query=cloud%20storage&top_k=3"

# Status
curl http://localhost:8000/status
```

---

## 🌐 API Endpoints

### Health Check
```bash
GET /health
```

### Query the RAG System
```bash
POST /query
Content-Type: application/json

{
  "query": "How do I set up Azure OpenAI?",
  "top_k": 3
}
```

Response:
```json
{
  "query": "How do I set up Azure OpenAI?",
  "answer": "...",
  "context": ["...", "...", "..."],
  "sources": [...]
}
```

### Retrieve Documents Only
```bash
GET /retrieve?query=Your%20question&top_k=3
```

### Ingest New Documents
```bash
POST /ingest
```

### Upload PDF Files
```bash
POST /upload
Content-Type: multipart/form-data

file: (binary PDF file)
```

### Get Status
```bash
GET /status
```

---

## 🧰 Troubleshooting

### No documents indexed
Ensure PDF files are in `data/documents/` and run:
```bash
python -m src.ingest
```

### Authentication errors
Check your API keys and endpoints in `.env`

### Memory issues with large datasets
Adjust `CHUNK_SIZE` or use `faiss-gpu` instead of `faiss-cpu`

---

## 🏗️ Future Enhancements

- [ ] Support for more document formats (DOCX, TXT, etc.)
- [ ] Multi-language support
- [ ] Document filtering by metadata
- [ ] Fine-tuning on domain-specific documents
- [ ] Streaming responses
- [ ] Document versioning
- [ ] Advanced query reformulation

---

## 📄 License

MIT

---

## 💬 Support

For issues and questions, please open an issue on GitHub.
