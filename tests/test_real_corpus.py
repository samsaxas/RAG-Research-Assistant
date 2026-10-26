"""
Checks for the real-world corpus (NIST AI 100-1). The question-file schema tests always run; the tests that need the
PDF skip themselves until you have run `python scripts/fetch_real_doc.py`.
"""
import json
import os

import pytest

from chunking import chunk_pages
from corpus import REAL_DIR, REAL_DOC_NAME, has_pdfs, resolve_docs_dir
from parse import parse_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA = os.path.join(ROOT, "eval", "qa_real.json")


def _load_validate():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rag_validate", os.path.join(ROOT, "eval", "validate_qa.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def qa():
    with open(QA) as fh:
        return json.load(fh)


def test_qa_file_schema(qa):
    answerable = [q for q in qa if q["expected_source"]]
    unanswerable = [q for q in qa if not q["expected_source"]]
    assert len(answerable) >= 30 and len(unanswerable) >= 3
    assert len({q["question"] for q in qa}) == len(qa), "duplicate questions"
    for q in answerable:
        assert q["expected_source"] == REAL_DOC_NAME
        assert q["evidence"] and all(isinstance(e, str) and e.strip() for e in q["evidence"])
        assert all(len(e) <= 100 for e in q["evidence"]), f"evidence longer than the chunk overlap: {q['question']}"
        assert q["expected_answer_contains"]
    for q in unanswerable:
        assert q["evidence"] == []


def test_resolve_docs_dir_errors_clearly_when_real_pdf_is_missing(monkeypatch):
    monkeypatch.setattr("corpus.has_pdfs", lambda d: False)
    with pytest.raises(FileNotFoundError, match="fetch_real_doc"):
        resolve_docs_dir("real")


needs_pdf = pytest.mark.skipif(not has_pdfs(REAL_DIR), reason="real PDF not downloaded (scripts/fetch_real_doc.py)")


@needs_pdf
def test_real_document_is_one_long_pdf():
    pdfs = [f for f in os.listdir(REAL_DIR) if f.lower().endswith(".pdf")]
    assert pdfs == [REAL_DOC_NAME]
    pages = parse_directory(REAL_DIR)
    assert 40 <= len(pages) <= 60
    assert len(chunk_pages(pages)) > 150


@needs_pdf
def test_every_evidence_label_exists_in_the_real_pdf(qa):
    chunks = chunk_pages(parse_directory(REAL_DIR))
    assert _load_validate().validate(qa, chunks) == []
