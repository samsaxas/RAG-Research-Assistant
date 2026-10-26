"""End-to-end API tests using FastAPI's TestClient (no network, no model downloads, no API keys)."""
import pytest

import generate
from api.service import InvalidDocument, UnsupportedFileType, sanitize_filename
from tests.helpers import ROCKET_PDF_LINES, make_pdf

HANDBOOK_Q = "How many days of paid annual leave do full-time employees get?"


def _upload(client, name="bluebird.pdf", data=None, headers=None):
    data = make_pdf(ROCKET_PDF_LINES) if data is None else data
    return client.post("/upload", files={"file": (name, data, "application/pdf")}, headers=headers or {})


# ----------------------------------------------------------------------------- health / documents
def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "tfidf"
    assert body["documents"] == 3 and body["chunks"] > 0
    assert body["llm_configured"] is False


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_lists_bundled_sample_documents(client):
    docs = client.get("/documents").json()
    assert {d["filename"] for d in docs} == {"company_handbook.pdf", "product_release_notes.pdf",
                                             "security_whitepaper.pdf"}
    assert all(d["origin"] == "sample" for d in docs)


# ----------------------------------------------------------------------------- /ask
def test_ask_returns_sources_and_marks_answer_as_not_generated_without_key(client):
    r = client.post("/ask", json={"question": HANDBOOK_Q, "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["generated"] is False
    assert len(body["sources"]) == 3
    assert body["sources"][0]["source"] == "company_handbook.pdf"
    assert {"source", "page", "score", "text"} <= set(body["sources"][0])


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "   "},
    {"question": "x" * 1001},
    {"question": "ok", "k": 0},
    {"question": "ok", "k": 11},
    {},
])
def test_ask_rejects_invalid_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_uses_llm_when_key_is_configured(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = {}

    def fake_generate(question, chunks, model_name=None):
        seen["n_chunks"] = len(chunks)
        return "18 days [company_handbook.pdf, p.1]"

    monkeypatch.setattr(generate, "generate_answer", fake_generate)
    body = client.post("/ask", json={"question": HANDBOOK_Q, "k": 2}).json()
    assert body["generated"] is True
    assert body["answer"] == "18 days [company_handbook.pdf, p.1]"
    assert seen["n_chunks"] == 2


def test_ask_maps_llm_failure_to_502_without_leaking_details(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def boom(*args, **kwargs):
        raise RuntimeError("upstream exploded: secret-internal-detail")

    monkeypatch.setattr(generate, "generate_answer", boom)
    r = client.post("/ask", json={"question": HANDBOOK_Q})
    assert r.status_code == 502
    assert "secret-internal-detail" not in r.text


# ----------------------------------------------------------------------------- /upload
def test_upload_makes_document_searchable(client):
    r = _upload(client)
    assert r.status_code == 201
    body = r.json()
    assert body == {"filename": "bluebird.pdf", "pages": 1, "replaced": False,
                    "total_documents": 4, "total_chunks": body["total_chunks"]}

    assert "bluebird.pdf" in {d["filename"] for d in client.get("/documents").json()}
    top = client.post("/ask", json={"question": "When do deployment freezes begin?", "k": 1}).json()
    assert top["sources"][0]["source"] == "bluebird.pdf"
    assert "6 PM on Fridays" in top["sources"][0]["text"]


def test_reuploading_same_name_replaces_the_document(client):
    assert _upload(client).json()["replaced"] is False
    second = _upload(client, data=make_pdf(["Completely different content about gardening."]))
    assert second.status_code == 201 and second.json()["replaced"] is True
    assert second.json()["total_documents"] == 4  # replaced, not duplicated


def test_upload_rejects_non_pdf_extension(client):
    r = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_upload_rejects_fake_pdf(client):
    assert _upload(client, data=b"this is not a pdf").status_code == 400


def test_upload_rejects_truncated_pdf(client):
    assert _upload(client, data=b"%PDF-1.4\nnot really").status_code == 400


def test_upload_rejects_empty_file(client):
    assert _upload(client, data=b"").status_code == 400


def test_upload_rejects_pdf_without_text(client):
    r = _upload(client, name="blank.pdf", data=make_pdf([]))
    assert r.status_code == 422
    assert "blank.pdf" not in {d["filename"] for d in client.get("/documents").json()}


def test_upload_rejects_oversized_file(client_factory):
    client = client_factory(MAX_UPLOAD_MB="1")
    r = _upload(client, data=b"%PDF-1.4\n" + b"0" * (1024 * 1024))
    assert r.status_code == 413


def test_upload_cannot_overwrite_bundled_sample(client):
    assert _upload(client, name="company_handbook.pdf").status_code == 409


def test_upload_sanitises_hostile_filenames(client):
    r = _upload(client, name="../../etc/evil name!.pdf")
    assert r.status_code == 201
    assert r.json()["filename"] == "evil name.pdf"


def test_failed_upload_leaves_index_unchanged(client):
    before = client.get("/health").json()
    _upload(client, data=b"%PDF-1.4\ngarbage")
    assert client.get("/health").json()["chunks"] == before["chunks"]


# ----------------------------------------------------------------------------- DELETE
def test_delete_uploaded_document(client):
    _upload(client)
    assert client.delete("/documents/bluebird.pdf").status_code == 204
    assert "bluebird.pdf" not in {d["filename"] for d in client.get("/documents").json()}


def test_delete_unknown_and_protected_documents(client):
    assert client.delete("/documents/nope.pdf").status_code == 404
    assert client.delete("/documents/company_handbook.pdf").status_code == 403


def test_uploads_survive_a_restart(client_factory):
    first = client_factory()
    assert _upload(first).status_code == 201
    second = client_factory()  # a fresh app process pointed at the same upload folder
    assert "bluebird.pdf" in {d["filename"] for d in second.get("/documents").json()}
    top = second.post("/ask", json={"question": "When do deployment freezes begin?", "k": 1}).json()
    assert top["sources"][0]["source"] == "bluebird.pdf"


def test_service_works_with_no_documents_then_first_upload(client_factory, tmp_path):
    empty = tmp_path / "empty_docs"
    empty.mkdir()
    client = client_factory(DOCS_DIR=str(empty))
    assert client.get("/health").json()["documents"] == 0
    assert client.post("/ask", json={"question": "anything?"}).status_code == 409
    assert _upload(client).status_code == 201
    top = client.post("/ask", json={"question": "When does the engine test window open?"}).json()
    assert top["sources"][0]["source"] == "bluebird.pdf"


# ----------------------------------------------------------------------------- API key
def test_api_key_protects_everything_except_health(client_factory):
    client = client_factory(API_KEY="s3cret")
    assert client.get("/health").status_code == 200
    for method, url, kwargs in [
        ("get", "/documents", {}),
        ("post", "/ask", {"json": {"question": "hi"}}),
        ("post", "/upload", {"files": {"file": ("a.pdf", make_pdf(["x"]), "application/pdf")}}),
        ("delete", "/documents/a.pdf", {}),
    ]:
        assert getattr(client, method)(url, **kwargs).status_code == 401, url
        assert getattr(client, method)(url, headers={"X-API-Key": "wrong"}, **kwargs).status_code == 401, url

    ok = client.post("/ask", json={"question": HANDBOOK_Q}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200


# ----------------------------------------------------------------------------- unit: filenames
@pytest.mark.parametrize("raw, expected", [
    ("report.pdf", "report.pdf"),
    ("My Report (final).PDF", "My Report _final.pdf"),
    ("C:\\Users\\me\\doc.pdf", "doc.pdf"),
    ("../../../etc/passwd.pdf", "passwd.pdf"),
])
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize("raw, error", [
    ("notes.txt", UnsupportedFileType),
    ("", UnsupportedFileType),
    (None, UnsupportedFileType),
    ("....pdf", InvalidDocument),
])
def test_sanitize_filename_rejects(raw, error):
    with pytest.raises(error):
        sanitize_filename(raw)
