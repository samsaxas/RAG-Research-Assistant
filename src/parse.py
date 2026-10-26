"""
Document parsing: extracts text per page from PDFs, keeping source metadata
(filename, page number) attached to every chunk downstream — this is what
lets the pipeline cite *which* document and page an answer came from.

Real-world PDFs need a little cleaning before chunking, otherwise retrieval is polluted by boilerplate:
  * running headers/footers ("NIST AI 100-1 AI RMF 1.0", "Page 12") repeat on almost every page, so they are
    detected by frequency and removed;
  * words broken across lines with a hyphen ("explain-\\nability") are re-joined;
  * line breaks are collapsed into spaces so a sentence is not split mid-way by the PDF's line wrapping.

`page` is the 1-based position of the page in the PDF file (what a PDF viewer shows in its page box). Documents
with roman-numeral front matter print a different number on the page itself.
"""
import os
import re

from pypdf import PdfReader

_SOFT_HYPHEN_BREAK = re.compile(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])")
_WHITESPACE = re.compile(r"\s+")
_ROMAN = re.compile(r"[ivxlc]+")
_DIGITS = re.compile(r"\d+")


def _line_key(line: str) -> str:
    """Page-number-insensitive key used to spot repeated headers/footers ('Page 12' and 'Page 13' match)."""
    s = line.strip().lower()
    if _ROMAN.fullmatch(s):
        return "#"
    return _DIGITS.sub("#", s)


MAX_BOILERPLATE_LINE = 60  # running headers/footers are short; long lines are body text and are never removed


def strip_repeated_lines(page_texts: list[str], min_pages: int = 6, threshold: float = 0.5) -> list[str]:
    """Remove short lines (<= MAX_BOILERPLATE_LINE chars) that appear on at least `threshold` of the pages —
    running headers and footers. Digits are ignored when comparing, so "Page 12" and "Page 13" count as the same.

    Documents shorter than `min_pages` are left alone: with only a few pages, "repeated" is not evidence of
    boilerplate.
    """
    if len(page_texts) < min_pages:
        return page_texts
    seen_on_pages: dict[str, int] = {}
    for text in page_texts:
        for key in {_line_key(ln) for ln in text.splitlines() if 0 < len(ln.strip()) <= MAX_BOILERPLATE_LINE}:
            seen_on_pages[key] = seen_on_pages.get(key, 0) + 1
    cutoff = threshold * len(page_texts)
    cleaned = []
    for text in page_texts:
        kept = [ln for ln in text.splitlines()
                if not (0 < len(ln.strip()) <= MAX_BOILERPLATE_LINE and seen_on_pages[_line_key(ln)] >= cutoff)]
        cleaned.append("\n".join(kept))
    return cleaned


def clean_text(text: str) -> str:
    """Re-join hyphenated line breaks and collapse all whitespace (including newlines) to single spaces."""
    text = text.replace("\u00ad", "")
    text = _SOFT_HYPHEN_BREAK.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def parse_pdf(path: str) -> list[dict]:
    """Returns a list of {source, page, text} dicts, one per non-empty page."""
    reader = PdfReader(path)
    filename = os.path.basename(path)
    raw = [(page.extract_text() or "") for page in reader.pages]
    cleaned = strip_repeated_lines(raw)
    pages = []
    for i, text in enumerate(cleaned):
        text = clean_text(text)
        if text:
            pages.append({"source": filename, "page": i + 1, "text": text})
    return pages


def parse_directory(dir_path: str) -> list[dict]:
    """Parses every PDF in a directory, returns the combined page list."""
    all_pages = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.lower().endswith(".pdf"):
            all_pages.extend(parse_pdf(os.path.join(dir_path, fname)))
    return all_pages


if __name__ == "__main__":
    from corpus import resolve_docs_dir

    docs_dir = resolve_docs_dir("auto")
    pages = parse_directory(docs_dir)
    print(f"Parsed {len(pages)} pages from {docs_dir}")
    for p in pages[:2]:
        print(f"--- {p['source']} p{p['page']} ---")
        print(p["text"][:200], "...\n")
