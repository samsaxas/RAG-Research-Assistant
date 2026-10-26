"""
Embeddings + FAISS vector store.

Two backends:
  - "huggingface" (production): sentence-transformers (all-MiniLM-L6-v2).
    This is what the resume line refers to. It requires downloading the model
    from the HuggingFace Hub on first use, so it needs real internet access.
  - "tfidf" (local fallback, for testing/demo only): scikit-learn TF-IDF,
    no external downloads required. This is NOT what should be cited as
    "HuggingFace embeddings" — it exists purely so this pipeline's chunking,
    indexing, and retrieval logic can be verified end-to-end without network
    access to the HF Hub.

Both backends are indexed into the same FAISS index so the rest of the
pipeline (retrieval, evaluation) doesn't care which one is active.
"""
import numpy as np
import faiss


class TfidfBackend:
    """Local, dependency-light fallback — TF-IDF + truncated SVD to get
    fixed-length dense vectors FAISS can index."""

    name = "tfidf (local fallback — NOT the production HuggingFace backend)"

    def __init__(self, dim: int = 128):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self._fitted = False
        self.dim = dim

    def fit(self, texts: list[str]):
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        if n_comp < self.dim:
            from sklearn.decomposition import TruncatedSVD
            self.svd = TruncatedSVD(n_components=max(n_comp, 2), random_state=42)
        vecs = self.svd.fit_transform(tfidf)
        self._fitted = True
        return self._normalize(vecs)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() on the corpus before encode()")
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        return self._normalize(vecs)

    @staticmethod
    def _normalize(vecs):
        vecs = vecs.astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms


class HuggingFaceBackend:
    """Production backend — requires network access to huggingface.co."""

    name = "sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # local import — only needed for this backend
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def fit(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")


class VectorStore:
    def __init__(self, backend):
        self.backend = backend
        self.index = None
        self.chunks = []

    def build(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        vecs = self.backend.fit(texts)
        self.index = faiss.IndexFlatIP(vecs.shape[1])  # cosine sim via normalized dot product
        self.index.add(vecs)
        return self

    def search(self, query: str, k: int = 4) -> list[dict]:
        qvec = self.backend.encode([query])
        scores, idxs = self.index.search(qvec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            c = dict(self.chunks[idx])
            c["score"] = float(score)
            results.append(c)
        return results


def get_backend(kind: str = "tfidf"):
    if kind == "huggingface":
        return HuggingFaceBackend()
    return TfidfBackend()
