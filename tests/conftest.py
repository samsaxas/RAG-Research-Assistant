import os

import pytest
from fastapi.testclient import TestClient

from api.main import app

SAMPLE_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_docs")

_ENV_TO_CLEAR = ("GEMINI_API_KEY", "GEMINI_MODEL", "API_KEY", "MAX_UPLOAD_MB", "DOCS_DIR")


@pytest.fixture
def client_factory(tmp_path, monkeypatch):
    """Returns make(**env) -> TestClient. Each client gets its own empty upload folder, the TF-IDF backend
    (no model downloads), and no API keys unless the test asks for them. Settings are read at app startup,
    so the environment must be prepared before the client is entered — which is what this does."""
    clients = []

    def make(**env):
        monkeypatch.setenv("EMBEDDING_BACKEND", "tfidf")
        monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
        for name in _ENV_TO_CLEAR:
            monkeypatch.delenv(name, raising=False)
        # Tests assert on the 3 tiny sample PDFs, so pin them even if the real document has been downloaded.
        monkeypatch.setenv("DOCS_DIR", SAMPLE_DOCS)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        tc = TestClient(app)
        tc.__enter__()  # runs the lifespan: builds the index
        clients.append(tc)
        return tc

    yield make
    for tc in clients:
        tc.__exit__(None, None, None)


@pytest.fixture
def client(client_factory):
    return client_factory()
