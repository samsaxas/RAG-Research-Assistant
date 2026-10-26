"""
Runtime configuration, read from environment variables so the exact same
container image runs locally, on Render, on Cloud Run, or anywhere else
(12-factor style: config lives in the environment, not in the code).
"""
import os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALID_BACKENDS = ("tfidf", "huggingface")


def default_docs_dir() -> str:
    """The real-world document (data/real_docs, fetched with scripts/fetch_real_doc.py) if it has been downloaded,
    otherwise the three tiny bundled sample PDFs. Same rule as src/corpus.py."""
    real = os.path.join(ROOT, "data", "real_docs")
    if os.path.isdir(real) and any(f.lower().endswith(".pdf") for f in os.listdir(real)):
        return real
    return os.path.join(ROOT, "data", "sample_docs")


@dataclass(frozen=True)
class Settings:
    docs_dir: str          # bundled, read-only corpus (real document if present, else samples; baked into the image)
    upload_dir: str        # where /upload stores PDFs (ephemeral on most cloud platforms!)
    backend: str           # "tfidf" (light, no model download) or "huggingface" (production embeddings)
    max_upload_mb: int     # reject uploads larger than this
    api_key: str | None    # if set, /ask, /upload and DELETE require the X-API-Key header

    @classmethod
    def from_env(cls) -> "Settings":
        backend = os.environ.get("EMBEDDING_BACKEND", "tfidf").strip().lower()
        if backend not in VALID_BACKENDS:
            raise ValueError(f"EMBEDDING_BACKEND must be one of {VALID_BACKENDS}, got {backend!r}")
        try:
            max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "10"))
        except ValueError as exc:
            raise ValueError("MAX_UPLOAD_MB must be an integer") from exc
        if max_upload_mb < 1:
            raise ValueError("MAX_UPLOAD_MB must be >= 1")
        return cls(
            docs_dir=os.environ.get("DOCS_DIR") or default_docs_dir(),
            upload_dir=os.environ.get("UPLOAD_DIR") or os.path.join(ROOT, "data", "uploads"),
            backend=backend,
            max_upload_mb=max_upload_mb,
            api_key=os.environ.get("API_KEY") or None,
        )
