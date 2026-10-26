"""Tiny dependency-free PDF builder so tests don't need reportlab or binary fixtures."""


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(lines: list[str]) -> bytes:
    """Build a valid single-page text PDF. An empty list gives a blank page (no extractable text)."""
    return make_multipage_pdf([lines])


def make_multipage_pdf(pages: list[list[str]]) -> bytes:
    """Build a valid text PDF with one page per entry of `pages` (each entry is a list of text lines)."""
    n = len(pages)
    page_ids = [3 + 2 * i for i in range(n)]       # page objects: 3, 5, 7, ...; their content streams: 4, 6, 8, ...
    font_id = 3 + 2 * n
    objects = ["<< /Type /Catalog /Pages 2 0 R >>",
               f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] /Count {n} >>"]
    for pid, lines in zip(page_ids, pages):
        stream = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(f"({_esc(line)}) Tj T*" for line in lines) + " ET"
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {pid + 1} 0 R "
                       f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return out


# A document whose facts don't exist anywhere in the bundled sample corpus.
ROCKET_PDF_LINES = [
    "Project Bluebird Launch Runbook",
    "The Bluebird rocket engine test window opens at 04:30 UTC every Tuesday.",
    "Deployment freezes for Bluebird begin at 6 PM on Fridays.",
]
