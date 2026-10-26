"""
Chunking: splits parsed pages into overlapping, roughly fixed-size chunks.
Overlap prevents a fact from being cut across a chunk boundary and lost to
retrieval; source/page metadata is carried through so every chunk can be
cited back to its origin.
"""


def chunk_pages(pages: list[dict], chunk_size: int = 700, overlap: int = 100) -> list[dict]:
    """
    chunk_size / overlap are in characters, not tokens — simple and
    dependency-free. For production, swap for a token-based splitter
    (e.g. LangChain's RecursiveCharacterTextSplitter) if you want chunk
    boundaries to respect sentence structure more precisely.
    """
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "chunk_id": f"{page['source']}_p{page['page']}_c{chunk_idx}",
                "text": chunk_text,
            })
            start += chunk_size - overlap
            chunk_idx += 1
    return chunks


if __name__ == "__main__":
    from parse import parse_directory
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(os.path.dirname(here), "data", "sample_docs")
    pages = parse_directory(docs_dir)
    chunks = chunk_pages(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    print(chunks[0])
