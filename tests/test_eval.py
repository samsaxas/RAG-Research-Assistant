"""Unit tests for the evaluation maths (eval/evaluate.py) — no model, no network."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("rag_evaluate", os.path.join(ROOT, "eval", "evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

Q_PHRASE = {"question": "q", "expected_source": "a.pdf", "evidence": ["risk tolerance"],
            "expected_answer_contains": "x"}
Q_PAGE = {"question": "q", "expected_source": "a.pdf", "expected_page": 2, "expected_answer_contains": "x"}


def chunk(source, page, text):
    return {"source": source, "page": page, "text": text}


def test_phrase_labelled_relevance_ignores_page_and_punctuation():
    assert ev.is_relevant(chunk("a.pdf", 9, "…the Risk-Tolerance, as stated…"), Q_PHRASE) is False   # 'risktolerance'
    assert ev.is_relevant(chunk("a.pdf", 9, "does not prescribe risk tolerance."), Q_PHRASE)
    assert not ev.is_relevant(chunk("b.pdf", 9, "does not prescribe risk tolerance."), Q_PHRASE)   # wrong source


def test_page_labelled_relevance():
    assert ev.is_relevant(chunk("a.pdf", 2, "anything"), Q_PAGE)
    assert not ev.is_relevant(chunk("a.pdf", 3, "anything"), Q_PAGE)


def test_first_relevant_rank():
    results = [chunk("a.pdf", 1, "no"), chunk("a.pdf", 5, "risk tolerance here"), chunk("a.pdf", 2, "risk tolerance")]
    assert ev.first_relevant_rank(results, Q_PHRASE) == 2
    assert ev.first_relevant_rank(results[:1], Q_PHRASE) is None


def test_summarize_ranks_hit_rates_and_mrr():
    ranks = [1, 2, 4, None]
    s = ev.summarize_ranks(ranks, ks=[1, 3, 5], mrr_k=10)
    assert s["hit_rate_at_1"] == 0.25 and s["hit_rate_at_3"] == 0.5 and s["hit_rate_at_5"] == 0.75
    assert s["mrr_at_10"] == pytest.approx((1 + 1 / 2 + 1 / 4 + 0) / 4, abs=1e-4)


def test_parse_citations_and_refusal_detection():
    ans = "Four functions [NIST.AI.100-1.pdf, p.23] and more [a b.pdf, p. 4]."
    assert ev.parse_citations(ans) == [("NIST.AI.100-1.pdf", 23), ("a b.pdf", 4)]
    assert ev.is_refusal("I don't know based on the provided documents.")
    assert not ev.is_refusal("The framework has four functions.")


def test_percentile():
    assert ev.percentile([], 95) == 0.0
    assert ev.percentile([1, 2, 3, 4, 100], 95) == 100
    assert ev.percentile([5], 50) == 5


def test_sample_corpus_end_to_end_via_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "OUT_DIR", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert ev.main(["--corpus", "sample", "--backend", "tfidf", "--min-hit-rate", "0.9"]) == 0
    assert (tmp_path / "eval_sample_tfidf.json").exists()
