"""
Evaluation harness for the RAG pipeline. Retrieval and generation are measured separately because they fail
separately (a good answer needs the right chunk *and* a faithful model).

Usage
-----
    python eval/evaluate.py --corpus real   --backend huggingface     # the real 48-page NIST document
    python eval/evaluate.py --corpus real   --backend tfidf           # keyword-style baseline, no downloads
    python eval/evaluate.py --corpus sample --backend tfidf           # the 3 tiny synthetic PDFs (CI-sized)

1. RETRIEVAL (offline, no API key). For every answerable question the first retrieved chunk that actually
   contains the labelled evidence phrase (from the expected source document) defines the rank of the right answer.
   Reported: Hit@1/3/5/10 (= Recall@k when each question has one evidence passage), MRR@10, and retrieval latency.

2. GENERATION (needs GEMINI_API_KEY). Each question is answered by Gemini under the grounding guardrail, then
   scored on: answer correctness (expected keywords present), citation accuracy (the answer cites the right
   source file), citation validity (every cited [file, p.N] really was among the retrieved chunks), correct
   refusal on unanswerable questions, and end-to-end latency (retrieval + LLM).
   Without a key this part is skipped with a message — it never prints made-up numbers.

Question files: eval/qa_real.json (real document, phrase-labelled) and eval/qa_set.json (sample PDFs,
page-labelled). Check labels against your PDF with `python eval/validate_qa.py` before trusting any number.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from chunking import chunk_pages  # noqa: E402
from corpus import CORPORA, resolve_docs_dir  # noqa: E402
from embed_store import VectorStore, get_backend  # noqa: E402
from parse import parse_directory  # noqa: E402
from textutil import contains_keywords, contains_phrase, normalize  # noqa: E402

QA_FILES = {"real": os.path.join(HERE, "qa_real.json"), "sample": os.path.join(HERE, "qa_set.json")}
OUT_DIR = os.path.join(os.path.dirname(HERE), "outputs")
CITATION_RE = re.compile(r"\[([^\]\[,]+?\.pdf),\s*p\.?\s*(\d+)\]", re.IGNORECASE)
REFUSAL_MARKERS = ("don't know", "do not know", "not in the provided", "not provided", "cannot be answered")


# ----------------------------------------------------------------------------- relevance + metrics
def is_relevant(chunk: dict, q: dict) -> bool:
    """A retrieved chunk is relevant if it comes from the expected source and either contains one of the
    labelled evidence phrases (real-document set) or sits on the labelled page (page-labelled set)."""
    if chunk["source"] != q["expected_source"]:
        return False
    if q.get("evidence"):
        text = normalize(chunk["text"])
        return any(contains_phrase(text, phrase) for phrase in q["evidence"])
    return chunk["page"] == q["expected_page"]


def first_relevant_rank(results: list[dict], q: dict) -> int | None:
    """1-based rank of the first relevant chunk, or None if none of `results` is relevant."""
    for rank, chunk in enumerate(results, start=1):
        if is_relevant(chunk, q):
            return rank
    return None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def summarize_ranks(ranks: list[int | None], ks: list[int], mrr_k: int) -> dict:
    n = len(ranks)
    out = {"n_questions": n}
    for k in ks:
        out[f"hit_rate_at_{k}"] = round(sum(1 for r in ranks if r is not None and r <= k) / n, 4) if n else 0.0
    out[f"mrr_at_{mrr_k}"] = round(sum(1 / r for r in ranks if r is not None and r <= mrr_k) / n, 4) if n else 0.0
    return out


def evaluate_retrieval(store, qa_set, k=4, search_depth=10):
    ks = sorted({1, 3, 5, search_depth, k})
    answerable = [q for q in qa_set if q["expected_source"] is not None]
    ranks, latencies, per_question = [], [], []
    for q in answerable:
        t0 = time.perf_counter()
        results = store.search(q["question"], k=search_depth)
        latencies.append((time.perf_counter() - t0) * 1000)
        rank = first_relevant_rank(results, q)
        ranks.append(rank)
        per_question.append({"question": q["question"], "rank": rank,
                             "top_sources": [(r["source"], r["page"]) for r in results[:k]]})
    summary = summarize_ranks(ranks, ks, search_depth)
    summary.update({
        "k": k,
        "hit_rate_at_k": summary[f"hit_rate_at_{k}"],  # kept for the k you generate with
        "retrieval_latency_ms_mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "retrieval_latency_ms_p95": round(percentile(latencies, 95), 2),
        "per_question": per_question,
    })
    return summary


def parse_citations(answer: str) -> list[tuple[str, int]]:
    return [(m.group(1).strip(), int(m.group(2))) for m in CITATION_RE.finditer(answer)]


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def evaluate_generation(store, qa_set, k=4, limit=None, sleep=0.0):
    try:
        from generate import generate_answer
    except ImportError:
        print("[evaluate] google-genai not installed — skipping generation eval.")
        return None
    if not os.environ.get("GEMINI_API_KEY"):
        print("[evaluate] GEMINI_API_KEY not set — skipping generation eval. Retrieval metrics are real; "
              "generation/citation metrics need your own key.")
        return None

    questions = qa_set[:limit] if limit else qa_set
    correct = cited_right_source = refused = 0
    n_factual = n_refusal = n_errors = 0
    valid_citations = total_citations = 0
    latencies, log = [], []
    for i, q in enumerate(questions):
        if i and sleep:
            time.sleep(sleep)
        t0 = time.perf_counter()
        try:
            retrieved = store.search(q["question"], k=k)
            answer = generate_answer(q["question"], retrieved)
        except Exception as exc:  # one flaky API call must not discard the whole run
            n_errors += 1
            log.append({"question": q["question"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [ERROR] {q['question'][:60]} -> {type(exc).__name__}")
            continue
        latencies.append(time.perf_counter() - t0)

        entry = {"question": q["question"], "answer": answer}
        if q["expected_source"] is None:
            n_refusal += 1
            entry["refused"] = is_refusal(answer)
            refused += int(entry["refused"])
        else:
            n_factual += 1
            entry["correct"] = contains_keywords(answer, q["expected_answer_contains"])
            entry["cites_expected_source"] = q["expected_source"].lower() in answer.lower()
            retrieved_pairs = {(r["source"], r["page"]) for r in retrieved}
            cites = parse_citations(answer)
            entry["citations"] = cites
            total_citations += len(cites)
            valid_citations += sum(1 for c in cites if c in retrieved_pairs)
            correct += int(entry["correct"])
            cited_right_source += int(entry["cites_expected_source"])
        log.append(entry)

    def rate(num, den):
        return round(num / den, 4) if den else None

    return {
        "n_questions": len(questions), "n_errors": n_errors,
        "answer_correctness": rate(correct, n_factual),
        "citation_accuracy": rate(cited_right_source, n_factual),
        "citation_validity": rate(valid_citations, total_citations),
        "correct_refusal_rate": rate(refused, n_refusal),
        "end_to_end_latency_s_mean": round(statistics.fmean(latencies), 2) if latencies else None,
        "end_to_end_latency_s_p95": round(percentile(latencies, 95), 2) if latencies else None,
        "log": log,
    }


# ----------------------------------------------------------------------------- CLI
def fmt(value, pct=True):
    if value is None:
        return "n/a"
    return f"{value:.1%}" if pct else f"{value}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate retrieval (and, with GEMINI_API_KEY, generation).")
    ap.add_argument("backend_pos", nargs="?", choices=["tfidf", "huggingface"], help=argparse.SUPPRESS)  # legacy
    ap.add_argument("--backend", choices=["tfidf", "huggingface"], default=None)
    ap.add_argument("--corpus", choices=[c for c in CORPORA if c != "auto"], default="sample")
    ap.add_argument("--qa", help="path to a question file (default depends on --corpus)")
    ap.add_argument("--k", type=int, default=4, help="chunks passed to the LLM / reported as hit_rate@k")
    ap.add_argument("--chunk-size", type=int, default=700)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--gen-limit", type=int, help="only run the first N questions through Gemini (saves quota)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between Gemini calls (free-tier limits)")
    ap.add_argument("--min-hit-rate", type=float, help="exit with status 1 if hit_rate@k falls below this")
    args = ap.parse_args(argv)
    backend_kind = args.backend or args.backend_pos or "tfidf"

    try:
        docs_dir = resolve_docs_dir(args.corpus)
    except FileNotFoundError as exc:
        print(exc)
        return 2
    with open(args.qa or QA_FILES[args.corpus]) as f:
        qa_set = json.load(f)

    pages = parse_directory(docs_dir)
    chunks = chunk_pages(pages, chunk_size=args.chunk_size, overlap=args.overlap)
    docs = sorted({p["source"] for p in pages})
    print(f"Corpus '{args.corpus}': {len(docs)} document(s) {docs} -> {len(pages)} pages -> {len(chunks)} chunks "
          f"(size={args.chunk_size}, overlap={args.overlap})")

    t0 = time.perf_counter()
    backend = get_backend(backend_kind)
    store = VectorStore(backend).build(chunks)
    build_seconds = round(time.perf_counter() - t0, 2)
    print(f"Backend: {backend.name}   (model load + indexing: {build_seconds}s)")

    ret = evaluate_retrieval(store, qa_set, k=args.k)
    print(f"\n[RETRIEVAL] {ret['n_questions']} answerable questions")
    print("  " + "   ".join(f"Hit@{k}={ret[f'hit_rate_at_{k}']:.1%}" for k in sorted(
        int(key.rsplit('_', 1)[1]) for key in ret if key.startswith("hit_rate_at_") and key != "hit_rate_at_k")))
    print(f"  MRR@10={ret['mrr_at_10']:.3f}   latency mean={ret['retrieval_latency_ms_mean']} ms  "
          f"p95={ret['retrieval_latency_ms_p95']} ms")
    for pq in ret["per_question"]:
        mark = f"OK@{pq['rank']}" if pq["rank"] else "MISS"
        print(f"  [{mark:<6}] {pq['question'][:70]}")

    gen = evaluate_generation(store, qa_set, k=args.k, limit=args.gen_limit, sleep=args.sleep)
    if gen:
        print(f"\n[GENERATION] answer_correctness={fmt(gen['answer_correctness'])}  "
              f"citation_accuracy={fmt(gen['citation_accuracy'])}  citation_validity={fmt(gen['citation_validity'])}  "
              f"correct_refusal={fmt(gen['correct_refusal_rate'])}")
        print(f"[GENERATION] end-to-end latency mean={gen['end_to_end_latency_s_mean']} s  "
              f"p95={gen['end_to_end_latency_s_p95']} s   ({gen['n_errors']} API errors)")

    out = {"corpus": args.corpus, "documents": docs, "n_pages": len(pages), "n_chunks": len(chunks),
           "chunk_size": args.chunk_size, "overlap": args.overlap, "backend": backend.name,
           "index_build_seconds": build_seconds, "retrieval": ret, "generation": gen}
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"eval_{args.corpus}_{backend_kind}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved full results to {out_path}")

    if args.min_hit_rate is not None and ret["hit_rate_at_k"] < args.min_hit_rate:
        print(f"FAIL: hit_rate@{args.k} {ret['hit_rate_at_k']:.1%} < required {args.min_hit_rate:.1%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
