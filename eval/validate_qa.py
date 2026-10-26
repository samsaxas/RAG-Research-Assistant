"""
Check that every labelled evidence phrase in a question file really exists in your PDF — *before* you trust any
retrieval number. A label that can never be found would silently count as a miss.

    python scripts/fetch_real_doc.py
    python eval/validate_qa.py                      # real corpus, eval/qa_real.json
    python eval/validate_qa.py --corpus sample      # page-labelled sample set

For each evidence phrase it checks: (1) the phrase is no longer than the chunk overlap, so a chunk boundary can
never cut it in half; (2) the phrase appears inside at least one chunk of the expected source. It lists every
phrase it cannot find, plus the nearest text, so you can fix the label (or the parser) quickly.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from chunking import chunk_pages  # noqa: E402
from corpus import resolve_docs_dir  # noqa: E402
from parse import parse_directory  # noqa: E402
from textutil import contains_phrase, normalize  # noqa: E402

QA_FILES = {"real": os.path.join(HERE, "qa_real.json"), "sample": os.path.join(HERE, "qa_set.json")}


def validate(qa_set: list[dict], chunks: list[dict], overlap: int = 100) -> list[str]:
    """Return a list of human-readable problems (empty list = all labels OK)."""
    by_source: dict[str, list[str]] = {}
    for c in chunks:
        by_source.setdefault(c["source"], []).append(normalize(c["text"]))
    problems = []
    for q in qa_set:
        src = q["expected_source"]
        if src is None:
            continue
        if src not in by_source:
            problems.append(f"{q['question']!r}: expected_source {src!r} is not in the corpus {sorted(by_source)}")
            continue
        if not q.get("evidence"):
            if not any(c["source"] == src and c["page"] == q.get("expected_page") for c in chunks):
                problems.append(f"{q['question']!r}: no chunk on {src} page {q.get('expected_page')}")
            continue
        for phrase in q["evidence"]:
            if len(phrase) > overlap:
                problems.append(f"{q['question']!r}: evidence phrase is {len(phrase)} chars (> overlap {overlap}), "
                                f"it could be split across two chunks: {phrase!r}")
            if not any(contains_phrase(text, phrase) for text in by_source[src]):
                problems.append(f"{q['question']!r}: evidence phrase NOT FOUND in {src}: {phrase!r}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=["real", "sample"], default="real")
    ap.add_argument("--qa")
    ap.add_argument("--chunk-size", type=int, default=700)
    ap.add_argument("--overlap", type=int, default=100)
    args = ap.parse_args(argv)

    try:
        docs_dir = resolve_docs_dir(args.corpus)
    except FileNotFoundError as exc:
        print(exc)
        return 2
    with open(args.qa or QA_FILES[args.corpus]) as f:
        qa_set = json.load(f)
    pages = parse_directory(docs_dir)
    chunks = chunk_pages(pages, chunk_size=args.chunk_size, overlap=args.overlap)
    n_labelled = sum(1 for q in qa_set if q["expected_source"])
    print(f"{len(pages)} pages, {len(chunks)} chunks, {n_labelled} labelled questions "
          f"({len(qa_set) - n_labelled} unanswerable)")

    problems = validate(qa_set, chunks, overlap=args.overlap)
    if not problems:
        print("All evidence phrases found. Labels are consistent with the PDF.")
        return 0
    print(f"\n{len(problems)} problem(s):")
    for p in problems:
        print("  -", p)
    print("\nFix the label in the question file (shorter phrase, or wording as it appears in the PDF) and re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
