"""
End-to-end pipeline: parse documents -> chunk -> index -> retrieve -> generate.
CLI usage:
    python src/rag_pipeline.py "What are the four functions of the AI RMF Core?"
    python src/rag_pipeline.py "..." --backend huggingface   # production embeddings
    python src/rag_pipeline.py "..." --corpus sample          # the 3 tiny synthetic PDFs
"""
import os
import argparse

from parse import parse_directory
from chunking import chunk_pages
from corpus import CORPORA, resolve_docs_dir
from embed_store import VectorStore, get_backend

# Default corpus: data/real_docs/ if it holds a PDF, else data/sample_docs/ (see src/corpus.py).
DOCS_DIR = resolve_docs_dir("auto")


def build_index(docs_dir: str | None = None, backend_kind: str = "tfidf",
                chunk_size: int = 700, overlap: int = 100) -> VectorStore:
    docs_dir = docs_dir or resolve_docs_dir("auto")
    pages = parse_directory(docs_dir)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    backend = get_backend(backend_kind)
    store = VectorStore(backend).build(chunks)
    print(f"[rag_pipeline] Indexed {len(chunks)} chunks from {len(pages)} pages "
          f"(backend: {backend.name})")
    return store


def ask(question: str, store: VectorStore, k: int = 4, use_llm: bool = True) -> dict:
    retrieved = store.search(question, k=k)

    if use_llm and os.environ.get("GEMINI_API_KEY"):
        from generate import generate_answer
        answer = generate_answer(question, retrieved)
    else:
        # No API key available in this environment — return the retrieved
        # context itself rather than fabricating a generated answer.
        answer = ("[No GEMINI_API_KEY set — showing retrieved context instead of a "
                   "generated answer. Set GEMINI_API_KEY to get a real Gemini response.]")

    return {"question": question, "answer": answer, "retrieved": retrieved}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--backend", default="tfidf", choices=["tfidf", "huggingface"])
    parser.add_argument("--corpus", default="auto", choices=CORPORA)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    store = build_index(resolve_docs_dir(args.corpus), backend_kind=args.backend)
    result = ask(args.question, store, k=args.k)

    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}\n")
    print("Retrieved context:")
    for r in result["retrieved"]:
        print(f"  [{r['source']}, p.{r['page']}] score={r['score']:.3f} — {r['text'][:100]}...")
