"""Tests for the Gemini wrapper using a fake client — verifies our code, never touches the network."""
import pytest

import generate

CHUNKS = [{"source": "a.pdf", "page": 1, "text": "Fact."}]


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    def __init__(self, text="  An answer [a.pdf, p.1]  "):
        self.calls = []
        self._text = text
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeResponse(self._text)


@pytest.fixture
def fake_client(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(generate, "_get_client", lambda api_key, timeout: fake)
    return fake


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        generate.generate_answer("q?", CHUNKS)


def test_sends_grounding_prompt_and_strips_answer(monkeypatch, fake_client):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    answer = generate.generate_answer("What is the fact?", CHUNKS)
    call = fake_client.calls[0]
    assert answer == "An answer [a.pdf, p.1]"
    assert call["model"] == generate.DEFAULT_MODEL
    assert call["config"].system_instruction == generate.GROUNDING_SYSTEM_PROMPT
    assert "[a.pdf, p.1]" in call["contents"] and "What is the fact?" in call["contents"]


def test_model_is_configurable_via_env(monkeypatch, fake_client):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "some-other-model")
    generate.generate_answer("q?", CHUNKS)
    assert fake_client.calls[0]["model"] == "some-other-model"


def test_empty_response_is_an_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(generate, "_get_client", lambda api_key, timeout: _FakeClient(text=None))
    with pytest.raises(RuntimeError, match="empty response"):
        generate.generate_answer("q?", CHUNKS)
