"""
Which PDFs does the project index by default?

  data/real_docs/    a real-world document (the NIST AI Risk Management Framework, ~48 pages).
                     Fetch it with:  python scripts/fetch_real_doc.py
  data/sample_docs/  three tiny synthetic PDFs, kept for fast offline tests and CI.

Default ("auto"): use data/real_docs/ if it contains at least one PDF, otherwise fall back to data/sample_docs/.
The DOCS_DIR environment variable overrides everything.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(ROOT, "data", "real_docs")
SAMPLE_DIR = os.path.join(ROOT, "data", "sample_docs")
REAL_DOC_NAME = "NIST.AI.100-1.pdf"
CORPORA = ("auto", "real", "sample")


def has_pdfs(directory: str) -> bool:
    return os.path.isdir(directory) and any(f.lower().endswith(".pdf") for f in os.listdir(directory))


def resolve_docs_dir(corpus: str = "auto") -> str:
    """Map a corpus name to a directory. Raises FileNotFoundError if 'real' is requested but not downloaded."""
    if corpus not in CORPORA:
        raise ValueError(f"corpus must be one of {CORPORA}, got {corpus!r}")
    if corpus == "sample":
        return SAMPLE_DIR
    if corpus == "real":
        if not has_pdfs(REAL_DIR):
            raise FileNotFoundError(
                f"No PDF found in {REAL_DIR}. Download the real document first:  python scripts/fetch_real_doc.py")
        return REAL_DIR
    return os.environ.get("DOCS_DIR") or (REAL_DIR if has_pdfs(REAL_DIR) else SAMPLE_DIR)
