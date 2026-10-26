"""Tests for the core RAG pipeline (parse -> chunk -> embed/index -> retrieve). Fully offline."""
import json
import os

from chunking import chunk_pages
from embed_store import VectorStore, get_backend
from generate import build_prompt
from parse import parse_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(ROOT, "data", "sample_docs")
QA_PATH = os.path.join(ROOT, "eval", "qa_set.json")


def test_parse_keeps_source_and_page_metadata():
    pages = parse_directory(DOCS_DIR)
    assert {p["source"] for p in pages} == {"company_handbook.pdf", "product_release_notes.pdf",
                                            "security_whitepaper.pdf"}
    assert all(p["page"] >= 1 and p["text"].strip() for p in pages)


def test_chunking_overlaps_and_carries_metadata():
    text = "".join(chr(97 + i % 26) for i in range(1500))
    chunks = chunk_pages([{"source": "x.pdf", "page": 2, "text": text}], chunk_size=700, overlap=100)
    assert [c["chunk_id"] for c in chunks] == ["x.pdf_p2_c0", "x.pdf_p2_c1", "x.pdf_p2_c2"]
    assert all(c["source"] == "x.pdf" and c["page"] == 2 for c in chunks)
    assert all(len(c["text"]) <= 700 for c in chunks)
    # consecutive chunks share exactly `overlap` characters, so a fact can't be cut in half at a boundary
    assert chunks[1]["text"][:100] == chunks[0]["text"][-100:]


def test_retrieval_hit_rate_on_labeled_questions():
    """Regression guard for retrieval quality (the shipped baseline is 100% hit-rate@4 with TF-IDF)."""
    with open(QA_PATH) as fh:
        qa_set = [q for q in json.load(fh) if q["expected_source"] is not None]
    store = VectorStore(get_backend("tfidf")).build(chunk_pages(parse_directory(DOCS_DIR)))
    hits = sum(
        any(r["source"] == q["expected_source"] and r["page"] == q["expected_page"]
            for r in store.search(q["question"], k=4))
        for q in qa_set
    )
    assert hits / len(qa_set) >= 0.9


def test_prompt_contains_context_with_citations():
    chunks = [{"source": "a.pdf", "page": 3, "text": "Fact one."}]
    prompt = build_prompt("What is the fact?", chunks)
    assert "[a.pdf, p.3]" in prompt and "Fact one." in prompt and "What is the fact?" in prompt
