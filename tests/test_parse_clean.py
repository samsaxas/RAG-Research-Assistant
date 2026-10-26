"""Parser behaviour on real-world layouts: repeated headers/footers and hyphenated line breaks."""
from parse import clean_text, parse_pdf, strip_repeated_lines
from tests.helpers import make_multipage_pdf


_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def _page(i):
    return ["NIST AI 100-1 AI RMF 1.0",
            f"Unique body sentence number {i} about governance, accountability and the explain-",
            f"ability of {_WORDS[i - 1]} models, plus a third-party dependency.",
            f"Page {i}"]


def test_clean_text_rejoins_hyphenated_breaks_and_collapses_whitespace():
    assert clean_text("explain-\nability   of\nmodels") == "explainability of models"
    assert clean_text("third-party and  well-known") == "third-party and well-known"   # real hyphens survive


def test_repeated_header_and_footer_are_removed_but_body_is_kept():
    pages = ["\n".join(_page(i)) for i in range(1, 9)]
    cleaned = strip_repeated_lines(pages)
    assert all("NIST AI 100-1" not in p and "Page " not in p for p in cleaned)
    assert "Unique body sentence number 3" in cleaned[2]


def test_long_lines_are_never_treated_as_boilerplate():
    long_line = "Body text that differs between pages only by one digit, here: {}. " + "filler " * 12
    pages = ["\n".join(["Running header 1", long_line.format(i)]) for i in range(1, 9)]
    cleaned = strip_repeated_lines(pages)
    assert all("Running header" not in p and "Body text" in p for p in cleaned)


def test_short_documents_are_left_untouched():
    pages = ["\n".join(_page(i)) for i in range(1, 4)]            # 3 pages < min_pages
    assert strip_repeated_lines(pages) == pages


def test_parse_pdf_end_to_end(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_multipage_pdf([_page(i) for i in range(1, 9)]))
    pages = parse_pdf(str(pdf))
    assert [p["page"] for p in pages] == list(range(1, 9))
    assert all(p["source"] == "doc.pdf" for p in pages)
    assert "explainability of echo models" in pages[4]["text"]
    assert "third-party" in pages[4]["text"]
    assert "NIST AI 100-1" not in pages[4]["text"] and "Page 5" not in pages[4]["text"]
    assert "\n" not in pages[4]["text"]
