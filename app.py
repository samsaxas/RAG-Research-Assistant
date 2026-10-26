"""
Streamlit chat UI for the RAG-Based Multi-Document Research Analyst.
Run with: streamlit run app.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
from corpus import REAL_DIR, SAMPLE_DIR, has_pdfs
from rag_pipeline import ask, build_index

st.set_page_config(page_title="Multi-Document Research Analyst", page_icon="📄", layout="centered")
st.title("📄 Multi-Document Research Analyst")
st.caption("RAG pipeline: PDF parsing → cleaning → chunking → FAISS retrieval → Gemini generation with a grounding "
           "guardrail. Citations like [file.pdf, p.N] use the page number shown by a PDF viewer.")

real_available = has_pdfs(REAL_DIR)
corpus_options = (["Real document (NIST AI RMF 1.0, 48 pages)"] if real_available else []) + \
                 ["Sample PDFs (3 tiny synthetic files)"]
corpus_choice = st.sidebar.radio("Corpus", corpus_options)
if not real_available:
    st.sidebar.info("The real 48-page document isn't downloaded yet. Run  `python scripts/fetch_real_doc.py`  "
                    "and restart the app.")
docs_dir = REAL_DIR if corpus_choice.startswith("Real") else SAMPLE_DIR

backend_kind = st.sidebar.radio(
    "Embedding backend",
    ["tfidf", "huggingface"],
    help="tfidf = local fallback (works without internet). "
         "huggingface = production backend (sentence-transformers), needs internet access to download the model.",
)
has_key = bool(os.environ.get("GEMINI_API_KEY"))
if not has_key:
    st.sidebar.warning("GEMINI_API_KEY not set — answers will show retrieved context only, not a generated response. "
                       "Set the env var and restart to enable real generation.")

st.sidebar.markdown("**Indexed documents:**")
for f in sorted(os.listdir(docs_dir)):
    if f.lower().endswith(".pdf"):
        st.sidebar.markdown(f"- {f}")


@st.cache_resource
def get_store(docs_dir, backend_kind):
    return build_index(docs_dir, backend_kind=backend_kind)


store = get_store(docs_dir, backend_kind)
st.sidebar.caption(f"{len(store.chunks)} chunks indexed")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about the indexed documents...")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("Retrieved context"):
            for r in turn["retrieved"]:
                st.markdown(f"**[{r['source']}, p.{r['page']}]** (score: {r['score']:.3f})")
                st.text(r["text"][:300])

if question:
    result = ask(question, store, use_llm=True)
    st.session_state.history.append(result)
    st.rerun()
