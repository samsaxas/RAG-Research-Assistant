"""
FastAPI service for the RAG Multi-Document Research Analyst.

Run locally:   uvicorn api.main:app --reload
Interactive docs (Swagger UI): http://localhost:8000/docs
"""
import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.security import APIKeyHeader

from .config import Settings
from .schemas import AskRequest, AskResponse, DocumentInfo, HealthResponse, UploadResponse
from .service import RAGService, ServiceError  # importing .service also puts src/ on sys.path
from generate import get_model_name  # noqa: E402  (src/generate.py)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Settings and the index are created at startup (not import time), so tests and
    # deployments can set environment variables first. A bad config fails fast here.
    settings = Settings.from_env()
    app.state.settings = settings
    app.state.service = RAGService(settings)
    yield


app = FastAPI(
    title="RAG Multi-Document Research Analyst API",
    version="1.0.0",
    description=("Upload PDFs, then ask questions answered strictly from their content, with source and "
                 "page citations. Retrieval: FAISS. Generation: Gemini with a grounding guardrail."),
    lifespan=lifespan,
)


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


# --------------------------------------------------------------------------- dependencies
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_service(request: Request) -> RAGService:
    return request.app.state.service


def require_api_key(request: Request, provided: str | None = Depends(_api_key_header)) -> None:
    """Optional protection: enforced only when the API_KEY environment variable is set."""
    expected = request.app.state.settings.api_key
    if expected and not (provided and secrets.compare_digest(provided.encode(), expected.encode())):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.",
                            headers={"WWW-Authenticate": "ApiKey"})


# --------------------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(service: RAGService = Depends(get_service)):
    """Liveness/readiness probe used by Docker, Render and Cloud Run. Never requires the API key."""
    return HealthResponse(
        status="ok",
        backend=service.backend_name,
        documents=len(service.documents()),
        chunks=service.n_chunks,
        llm_configured=bool(os.environ.get("GEMINI_API_KEY")),
        model=get_model_name(),
    )


@app.get("/documents", response_model=list[DocumentInfo], tags=["documents"],
         dependencies=[Depends(require_api_key)])
def list_documents(service: RAGService = Depends(get_service)):
    """List every indexed document (bundled samples + uploads)."""
    return service.documents()


@app.post("/upload", response_model=UploadResponse, status_code=201, tags=["documents"],
          dependencies=[Depends(require_api_key)])
def upload(file: UploadFile = File(..., description="A text-based PDF."),
           service: RAGService = Depends(get_service)):
    """Upload a PDF. It is parsed, chunked and added to the index; it is searchable as soon as this returns."""
    max_bytes = service.settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)  # read one byte past the limit so we can detect oversize files
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File is larger than {service.settings.max_upload_mb} MB.")
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    info = service.add_document(file.filename, data)
    return UploadResponse(**info, total_documents=len(service.documents()), total_chunks=service.n_chunks)


@app.delete("/documents/{filename}", status_code=204, tags=["documents"],
            dependencies=[Depends(require_api_key)])
def delete_document(filename: str, service: RAGService = Depends(get_service)):
    """Remove an uploaded document from the index. Bundled sample documents are protected."""
    service.delete_document(filename)
    return Response(status_code=204)


@app.post("/ask", response_model=AskResponse, tags=["qa"], dependencies=[Depends(require_api_key)])
def ask(body: AskRequest, service: RAGService = Depends(get_service)):
    """Ask a question. Answers come only from the indexed documents and cite `[file.pdf, p.N]`."""
    result = service.ask(body.question, body.k)
    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        generated=result["generated"],
        sources=[{"source": r["source"], "page": r["page"], "score": r["score"], "text": r["text"]}
                 for r in result["retrieved"]],
    )
