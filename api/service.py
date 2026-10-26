"""
Service layer: owns the in-memory vector index and wraps the existing RAG
pipeline (src/) for use behind a web API.

Design notes
------------
* The index is immutable once built. Uploading or deleting a document builds a
  *new* index off to the side and swaps it in with one assignment, so
  concurrent /ask requests always see a complete, consistent index and never
  block on a rebuild.
* Writers (upload/delete) are serialised with a lock.
* Everything is validated and indexed *before* the file is committed to disk,
  so a bad upload can never leave the index and the upload folder disagreeing.
* Uploaded files live in UPLOAD_DIR, separate from the bundled sample corpus,
  which can never be overwritten or deleted through the API.
"""
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
from dataclasses import dataclass, field

# The pipeline modules live in src/ and use flat imports (`from parse import ...`).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chunking import chunk_pages  # noqa: E402
from embed_store import VectorStore, get_backend  # noqa: E402
from parse import parse_pdf  # noqa: E402
from rag_pipeline import ask as pipeline_ask  # noqa: E402

from .config import Settings  # noqa: E402

log = logging.getLogger("rag_api")

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")


# --------------------------------------------------------------------------- errors
class ServiceError(Exception):
    """Base class; `status_code` is mapped to an HTTP response in api/main.py."""

    status_code = 500


class UnsupportedFileType(ServiceError):
    status_code = 415


class InvalidDocument(ServiceError):
    status_code = 400


class NoExtractableText(ServiceError):
    status_code = 422


class DocumentConflict(ServiceError):
    status_code = 409


class NoDocumentsIndexed(ServiceError):
    status_code = 409


class ProtectedDocument(ServiceError):
    status_code = 403


class DocumentNotFound(ServiceError):
    status_code = 404


class GenerationFailed(ServiceError):
    status_code = 502


# --------------------------------------------------------------------------- helpers
def sanitize_filename(filename: str | None) -> str:
    """Turn a client-supplied filename into a safe, flat file name (no paths, no odd characters)."""
    name = os.path.basename((filename or "").replace("\\", "/")).strip()
    if not name.lower().endswith(".pdf"):
        raise UnsupportedFileType("Only .pdf files are supported.")
    stem = _SAFE_CHARS.sub("_", name[:-4]).strip(" .-_")
    if not stem:
        raise InvalidDocument("The file name is empty after removing unsafe characters.")
    return f"{stem[:100]}.pdf"


def _list_pdfs(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory)
                  if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(directory, f)))


@dataclass(frozen=True)
class IndexState:
    """One immutable snapshot of what is indexed."""

    pages: list = field(default_factory=list)
    store: VectorStore | None = None
    n_chunks: int = 0


# --------------------------------------------------------------------------- service
class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._hf_backend = None  # the HuggingFace model is heavy: load it once, reuse it
        os.makedirs(settings.upload_dir, exist_ok=True)

        pages = []
        for directory in (settings.docs_dir, settings.upload_dir):
            for fname in _list_pdfs(directory):
                try:
                    pages.extend(parse_pdf(os.path.join(directory, fname)))
                except Exception:  # one corrupt file must not stop the service from starting
                    log.exception("Skipping unreadable PDF at startup: %s", fname)
        self._state = self._build_state(pages)
        log.info("Index ready: %d pages, %d chunks, backend=%s",
                 len(self._state.pages), self._state.n_chunks, self.backend_name)

    # ----- introspection
    @property
    def backend_name(self) -> str:
        return self.settings.backend

    @property
    def n_chunks(self) -> int:
        return self._state.n_chunks

    def documents(self) -> list[dict]:
        sample_names = set(_list_pdfs(self.settings.docs_dir))
        counts: dict[str, int] = {}
        for p in self._state.pages:
            counts[p["source"]] = counts.get(p["source"], 0) + 1
        return [{"filename": name,
                 "origin": "sample" if name in sample_names else "uploaded",
                 "pages": n}
                for name, n in sorted(counts.items())]

    # ----- index building
    def _backend(self):
        if self.settings.backend == "huggingface":
            if self._hf_backend is None:
                self._hf_backend = get_backend("huggingface")
            return self._hf_backend
        # TF-IDF is *fitted* to a corpus, so every index gets its own fresh backend object.
        return get_backend("tfidf")

    def _build_state(self, pages: list[dict]) -> IndexState:
        chunks = chunk_pages(pages)
        if not chunks:
            return IndexState()
        store = VectorStore(self._backend()).build(chunks)
        return IndexState(pages=pages, store=store, n_chunks=len(chunks))

    # ----- writes (serialised)
    def add_document(self, filename: str | None, data: bytes) -> dict:
        name = sanitize_filename(filename)
        if not data.startswith(b"%PDF-"):
            raise InvalidDocument("The file does not look like a PDF (missing %PDF header).")

        with self._lock:
            if name in _list_pdfs(self.settings.docs_dir):
                raise DocumentConflict(f"'{name}' is a bundled sample document and cannot be replaced. "
                                       "Rename your file and try again.")
            # Stage the file in a private temp dir *under its final name* (parse_pdf records the
            # file name as the citation source), validate it, and only then commit it.
            stage = tempfile.mkdtemp(prefix=".stage-", dir=self.settings.upload_dir)
            try:
                staged = os.path.join(stage, name)
                with open(staged, "wb") as fh:
                    fh.write(data)
                try:
                    new_pages = parse_pdf(staged)
                except Exception as exc:
                    raise InvalidDocument("Could not read this PDF (it may be corrupt or password-protected).") from exc
                if not new_pages:
                    raise NoExtractableText("No extractable text found — scanned/image-only PDFs need OCR, "
                                            "which this service does not do.")

                replaced = name in _list_pdfs(self.settings.upload_dir)
                pages = [p for p in self._state.pages if p["source"] != name] + new_pages
                new_state = self._build_state(pages)              # 1. build the new index in memory
                os.replace(staged, os.path.join(self.settings.upload_dir, name))  # 2. commit the file
                self._state = new_state                            # 3. swap the index in
            finally:
                shutil.rmtree(stage, ignore_errors=True)

        log.info("Indexed upload %s (%d pages, replaced=%s)", name, len(new_pages), replaced)
        return {"filename": name, "pages": len(new_pages), "replaced": replaced}

    def delete_document(self, filename: str) -> None:
        with self._lock:
            if filename in _list_pdfs(self.settings.docs_dir):
                raise ProtectedDocument("Bundled sample documents cannot be deleted.")
            if filename not in _list_pdfs(self.settings.upload_dir):
                raise DocumentNotFound(f"No uploaded document named '{filename}'.")
            pages = [p for p in self._state.pages if p["source"] != filename]
            new_state = self._build_state(pages)
            os.remove(os.path.join(self.settings.upload_dir, filename))
            self._state = new_state
        log.info("Deleted upload %s", filename)

    # ----- reads
    def ask(self, question: str, k: int) -> dict:
        state = self._state  # snapshot: an upload finishing mid-request can't affect this call
        if state.store is None:
            raise NoDocumentsIndexed("No documents are indexed yet — upload a PDF first.")
        try:
            result = pipeline_ask(question, state.store, k=k, use_llm=True)
        except Exception as exc:
            log.exception("Answer generation failed")
            raise GenerationFailed("The language model request failed. Please try again.") from exc
        result["generated"] = bool(os.environ.get("GEMINI_API_KEY"))
        return result
