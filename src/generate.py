"""
Generation layer — calls the Gemini API with retrieved chunks as context,
using a system prompt that instructs the model to answer only from the
provided context (the "grounding guardrail" from the resume bullet).

Requires: pip install google-genai, and a GEMINI_API_KEY env var.

Configuration (environment variables):
  GEMINI_API_KEY          required for generation
  GEMINI_MODEL            model id (default below). Gemini model ids are retired
                          regularly, so this is an env var and not hard-coded —
                          check https://ai.google.dev/gemini-api/docs/models
  GEMINI_TIMEOUT_SECONDS  per-request timeout (default 30). Without a timeout a
                          stuck upstream call would hang a web worker forever.

Note: this module has not been exercised against the live Gemini API in the
build environment (no access to generativelanguage.googleapis.com there) —
verify it once with your own key (see README, "Verify generation").
"""
import os
from functools import lru_cache

GROUNDING_SYSTEM_PROMPT = """You are a research assistant answering questions strictly from the provided document excerpts.

Rules:
1. Only use information explicitly present in the CONTEXT below. Do not use outside knowledge.
2. If the answer is not present in the CONTEXT, respond exactly: "I don't know based on the provided documents."
3. When you do answer, cite the source and page for every claim, like this: [source.pdf, p.3].
4. Do not speculate, infer beyond what's stated, or blend information from your own training data with the context.
"""

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_SECONDS = 30.0


def get_model_name() -> str:
    return os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(f"[{c['source']}, p.{c['page']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"


@lru_cache(maxsize=2)
def _get_client(api_key: str, timeout_seconds: float):
    """One client per (key, timeout) — avoids rebuilding the HTTP client on every request."""
    from google import genai  # local import — only needed when actually calling the API
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),  # SDK expects ms
    )


def generate_answer(question: str, retrieved_chunks: list[dict], model_name: str | None = None) -> str:
    """Calls Gemini with the grounding guardrail. Requires GEMINI_API_KEY."""
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a key from https://aistudio.google.com/apikey "
            "and set it as an environment variable before running generation."
        )
    timeout = float(os.environ.get("GEMINI_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS)
    client = _get_client(api_key, timeout)
    response = client.models.generate_content(
        model=model_name or get_model_name(),
        contents=build_prompt(question, retrieved_chunks),
        config=types.GenerateContentConfig(system_instruction=GROUNDING_SYSTEM_PROMPT),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response (it may have been blocked by a safety filter).")
    return text
