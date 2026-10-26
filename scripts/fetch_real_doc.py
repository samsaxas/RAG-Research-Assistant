"""
Download the real-world document used by this project into data/real_docs/.

Document: NIST AI 100-1, "Artificial Intelligence Risk Management Framework (AI RMF 1.0)", January 2023.
48 PDF pages. Published by the U.S. National Institute of Standards and Technology (a U.S. government work),
free of charge: https://doi.org/10.6028/NIST.AI.100-1

Usage:
    python scripts/fetch_real_doc.py                 # download + verify
    python scripts/fetch_real_doc.py --force         # re-download even if the file exists
    python scripts/fetch_real_doc.py --file ~/Downloads/NIST.AI.100-1.pdf   # use a copy you downloaded by hand

Only the standard library is needed to download; pypdf (already a project dependency) is used to count pages.
"""
import argparse
import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_DIR = os.path.join(ROOT, "data", "real_docs")
DEST = os.path.join(DEST_DIR, "NIST.AI.100-1.pdf")
URLS = [
    "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
    "https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf",
]
MIN_BYTES = 500_000          # the real file is ~1.8 MB; an HTML error page would be far smaller
PAGE_RANGE = (40, 60)        # sanity bounds: the real document has 48 pages


def verify(path: str) -> int:
    """Raise ValueError if `path` doesn't look like the expected document; return its page count."""
    with open(path, "rb") as fh:
        if fh.read(5) != b"%PDF-":
            raise ValueError("not a PDF (missing %PDF- header) — the server probably returned an HTML page")
    if os.path.getsize(path) < MIN_BYTES:
        raise ValueError(f"file is only {os.path.getsize(path)} bytes — too small to be the full document")
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  (pypdf not installed — skipping the page-count check)")
        return -1
    n_pages = len(PdfReader(path).pages)
    if not PAGE_RANGE[0] <= n_pages <= PAGE_RANGE[1]:
        raise ValueError(f"PDF has {n_pages} pages, expected about 48")
    return n_pages


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (rag-research-analyst fetch script)"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download again even if the file already exists")
    ap.add_argument("--file", help="copy a PDF you downloaded manually instead of using the network")
    ap.add_argument("--url", help="download from this URL instead of the built-in ones")
    args = ap.parse_args()

    os.makedirs(DEST_DIR, exist_ok=True)
    if os.path.exists(DEST) and not args.force and not args.file:
        try:
            print(f"Already present: {DEST} ({verify(DEST)} pages). Use --force to re-download.")
            return 0
        except ValueError as exc:
            print(f"Existing file is not valid ({exc}); downloading again.")

    tmp = DEST + ".part"
    if args.file:
        shutil.copyfile(args.file, tmp)
        sources = [f"file {args.file}"]
    else:
        sources = [args.url] if args.url else URLS
        for url in sources:
            print(f"Downloading {url} ...")
            try:
                download(url, tmp)
                break
            except (urllib.error.URLError, OSError) as exc:
                print(f"  failed: {exc}")
        else:
            print("\nCould not download the document automatically (offline, proxy, or the site blocked the request)."
                  "\nDownload it in a browser from https://doi.org/10.6028/NIST.AI.100-1 (the 'PDF' link), then run:"
                  f"\n    python scripts/fetch_real_doc.py --file <path-to-downloaded.pdf>\n"
                  f"or save it as {DEST}")
            if os.path.exists(tmp):
                os.remove(tmp)
            return 1

    try:
        n_pages = verify(tmp)
    except ValueError as exc:
        os.remove(tmp)
        print(f"Rejected the downloaded file: {exc}")
        return 1
    os.replace(tmp, DEST)
    print(f"Saved {DEST}\n  pages: {n_pages}   size: {os.path.getsize(DEST):,} bytes   sha256: {sha256(DEST)}")
    print("Next:  python eval/validate_qa.py   then   python eval/evaluate.py --corpus real --backend huggingface")
    return 0


if __name__ == "__main__":
    sys.exit(main())
