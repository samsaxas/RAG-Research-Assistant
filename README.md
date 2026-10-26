# RAG-Based Multi-Document Research Analyst

![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)
<!-- ^ replace <your-username>/<your-repo> with your GitHub path once you've pushed (Step 6) -->

A retrieval-augmented generation (RAG) system that answers questions across multiple PDF documents, grounded
strictly in retrieved context and cited by source and page — served as a **production-style REST API**
(FastAPI), containerised with **Docker**, tested and built by **GitHub Actions CI**, and deployable to
**Render, Google Cloud Run or AWS**. It also ships with a Streamlit chat UI and an evaluation harness for
retrieval quality and citation accuracy.

**The corpus is a real document:** the 48-page **NIST AI Risk Management Framework (AI RMF 1.0, NIST AI 100-1)** —
see [Quick start: the real document](#quick-start-the-real-document). Three tiny synthetic PDFs are still bundled
so tests and CI stay fast and offline.

**Contents**
0. [Quick start: the real document](#quick-start-the-real-document)
1. [Architecture](#1-architecture)
2. [The RAG pipeline](#2-the-rag-pipeline)
3. [Embedding backends](#3-embedding-backends--read-this-before-citing-numbers)
4. [Evaluation](#4-evaluation-evalevaluatepy)
5. [The REST API](#5-the-rest-api) — endpoints, examples, configuration
6. [**Step-by-step: from zip to a deployed API**](#6-step-by-step-from-zip-to-a-deployed-api)
7. [Verify generation with a live Gemini key](#7-verify-generation-with-a-live-gemini-key)
8. [Repo structure](#8-repo-structure)
9. [Design decisions & scaling notes](#9-design-decisions--scaling-notes)
10. [Troubleshooting](#10-troubleshooting)
11. [What is verified vs. what needs your own run](#11-what-is-verified-vs-what-needs-your-own-run)
12. [Changes from the original project](#12-changes-from-the-original-project)

---

## Quick start: the real document

The project is evaluated on **one real, long document** instead of made-up PDFs: *NIST AI 100-1, Artificial
Intelligence Risk Management Framework (AI RMF 1.0)*, January 2023 — 48 PDF pages of definitions, prose and long
tables of numbered controls (GOVERN 1.1 … MANAGE 4.3), published free of charge by the U.S. National Institute of
Standards and Technology (<https://doi.org/10.6028/NIST.AI.100-1>). See `data/real_docs/README.md`.

```bash
pip install -r requirements.txt                     # or requirements-dev.txt for the light, TF-IDF-only setup
python scripts/fetch_real_doc.py                    # 1. download + verify the PDF into data/real_docs/
python eval/validate_qa.py                          # 2. check all 36 question labels against *your* copy of the PDF
python eval/evaluate.py --corpus real --backend huggingface      # 3. retrieval metrics (add --backend tfidf as a baseline)
export GEMINI_API_KEY=...                           #    optional: also measures answers, citations, refusals, latency
python eval/evaluate.py --corpus real --backend huggingface --gen-limit 36 --sleep 4
python src/rag_pipeline.py "What are the four functions of the AI RMF Core?" --corpus real
uvicorn api.main:app --port 8000                    # the API / Docker image index the real PDF automatically
streamlit run app.py                                # chat UI with a corpus switch
```

If the download is blocked on your network, save the PDF from the DOI link in a browser and run
`python scripts/fetch_real_doc.py --file path/to/it.pdf`.
As soon as `data/real_docs/` contains a PDF, the API, Docker image, CLI and Streamlit app index it instead of the
three samples (override with `DOCS_DIR`). The tests always pin the samples, so they behave the same either way.

---

## 1. Architecture

```
                         ┌──────────────────────────── FastAPI service (api/) ────────────────────────────┐
 client ── HTTPS ──────► │  POST /upload   GET /documents   DELETE /documents/{name}   POST /ask   /health │
 (curl, Swagger UI,      └──────────────┬───────────────────────────────────────────────┬────────────────┘
  your web/mobile app)                  │ add / remove PDF                              │ question
                                        ▼                                               ▼
                         ┌──────────────────────── RAG pipeline (src/) ─────────────────────────────────┐
                         │ parse.py → chunking.py → embed_store.py (FAISS) → retrieve top-k → generate.py │
                         │  PDF→text    overlapping     TF-IDF / MiniLM        cosine sim     Gemini with │
                         │  +metadata   chunks          embeddings                            guardrail   │
                         └────────────────────────────────────────────────────────────────────────────────┘
```

The API layer (`api/`) is a thin wrapper: it owns the in-memory index, validates uploads, and calls the
same `src/` modules the CLI, the Streamlit app and the evaluation harness use.

## 2. The RAG pipeline

```
PDFs -> parse.py (per-page text extraction, source/page metadata kept; running headers/footers
                  removed, hyphenated line breaks re-joined, whitespace normalised)
     -> chunking.py (overlapping chunks, ~700 chars, 100 char overlap)
     -> embed_store.py (embeddings -> FAISS index)
     -> rag_pipeline.py (top-k retrieval -> Gemini generation)
     -> generate.py (grounding guardrail: "answer only from context,
                      say I don't know otherwise, cite source+page")
```

## 3. Embedding backends — read this before citing numbers

This repo supports two backends in `src/embed_store.py`:

- **`huggingface`** (production — `sentence-transformers/all-MiniLM-L6-v2`):
  this is what should be cited as "HuggingFace embeddings." Requires
  internet access to the HuggingFace Hub to download the model on first run
  (the Docker image downloads it at *build* time, so containers don't need to).
- **`tfidf`** (local fallback): scikit-learn TF-IDF + SVD, no downloads
  required. This exists **only** to verify the parsing/chunking/indexing/
  retrieval logic works end-to-end in network-restricted environments (and to
  keep CI and free-tier deployments light). It is explicitly labeled in code
  and output as "NOT the production HuggingFace backend" — do not cite results
  from this backend as HuggingFace-based.

Choose the backend with `--backend huggingface` on the CLI, the Streamlit sidebar, or the
`EMBEDDING_BACKEND` environment variable for the API.

## 4. Evaluation (`eval/evaluate.py`)

Retrieval and generation are evaluated **separately** because they fail separately (the right chunk may not be
retrieved; or it is retrieved and the model still answers badly).

**Questions.** `eval/qa_real.json` holds **36 hand-written questions** about the NIST document: 32 answerable
(definitions, the four functions, bias categories, specific numbered controls such as GOVERN 1.6 / MAP 3.5 /
MEASURE 2.11, appendix details) and 4 deliberately unanswerable ones (to test refusals). Each answerable question
is labelled with a short **evidence phrase** copied from the document instead of a page number, so a label stays
valid whatever page numbering the PDF uses. A retrieved chunk counts as correct if it comes from the expected file
and contains that phrase. Run `python eval/validate_qa.py` once after downloading: it proves every label exists in
your copy and flags any that don't. `eval/qa_set.json` is the original page-labelled set for the 3 sample PDFs.

**1. Retrieval (offline, no key).**

| Metric | Meaning |
|---|---|
| `Hit@1/3/5/10` | share of questions whose evidence chunk is in the top-k. With one evidence passage per question this equals **Recall@k** |
| `MRR@10` | mean of 1/rank of the first correct chunk (0 if not in the top 10) — rewards ranking it first |
| retrieval latency | mean / p95 milliseconds per query (query embedding + FAISS search) |

**2. Generation (needs `GEMINI_API_KEY`).**

| Metric | Meaning |
|---|---|
| `answer_correctness` | the answer contains the expected key terms (e.g. all four of govern / map / measure / manage) |
| `citation_accuracy` | the answer cites the right source file |
| `citation_validity` | every cited `[file, p.N]` really was among the retrieved chunks (catches invented citations) |
| `correct_refusal_rate` | unanswerable questions are refused ("I don't know …") instead of hallucinated |
| end-to-end latency | mean / p95 seconds for retrieval + Gemini |

Results are written to `outputs/eval_<corpus>_<backend>.json` (per-question ranks and every generated answer),
so you can inspect each miss. Useful flags: `--k`, `--chunk-size`, `--overlap` (to compare settings),
`--gen-limit N` and `--sleep S` (to stay inside free-tier Gemini quotas), `--min-hit-rate X` (exit non-zero if
`Hit@k` drops below X — handy as a regression gate).

**Turning this into resume numbers.** Every `XX` in the resume bullet maps to a field in the results file:
documents and chunks → the first line printed (`1 document → 48 pages → N chunks`); chunk size / overlap →
`chunk_size` / `overlap`; embedding model → `backend`; MRR and Recall@3 → `mrr_at_10`, `hit_rate_at_3`;
latency → `end_to_end_latency_s_mean`; faithfulness → `citation_validity` / `citation_accuracy` (name the metric
you actually measured). Quote only numbers from your own run.

*Caveats worth stating honestly:* 36 questions is a small test set, so a difference of one question moves a rate by
about 3 points; questions were written by the project author; the keyword-based correctness check can mark a
correct paraphrase as wrong (read the per-question log before trusting the headline). The sample-corpus score
(`outputs/eval_sample_tfidf.json`, 100% Hit@4 on only 6 chunks) is a plumbing check, not evidence of quality.

---

## 5. The REST API

Interactive documentation (Swagger UI) is served at **`/docs`** — you can upload a PDF and ask questions
from the browser without writing any code.

### Endpoints

| Method & path | Purpose | Success | Common errors |
|---|---|---|---|
| `GET /health` | Liveness/readiness probe: backend, document/chunk counts, whether an LLM key is configured. **Never requires the API key.** | `200` | – |
| `GET /documents` | List indexed PDFs (`origin`: `sample` = bundled, `uploaded`). | `200` | `401` |
| `POST /upload` | Upload a text-based PDF (multipart field `file`). It is parsed, chunked and indexed; searchable as soon as the response returns. Re-uploading the same file name replaces it. | `201` | `400` not a valid/readable PDF · `401` · `409` name clashes with a bundled sample · `413` too large · `415` not `.pdf` · `422` no extractable text (scanned PDF) |
| `DELETE /documents/{filename}` | Remove an uploaded PDF from the index. | `204` | `401` · `403` bundled sample · `404` |
| `POST /ask` | Ask a question. JSON body: `{"question": "...", "k": 4}` (`k` = chunks to retrieve, 1–10, default 4). | `200` | `401` · `409` nothing indexed · `422` invalid body · `502` LLM call failed |

`/ask` response:

```json
{
  "question": "How many days of paid annual leave do full-time employees get?",
  "answer": "Full-time employees accrue 18 days of paid annual leave per year [company_handbook.pdf, p.1].",
  "generated": true,
  "sources": [
    {"source": "company_handbook.pdf", "page": 1, "score": 0.83, "text": "Northbridge Analytics — Employee Handbook ..."}
  ]
}
```

`generated` is `false` when no `GEMINI_API_KEY` is configured: the API then returns a placeholder `answer`
plus the retrieved `sources`, rather than pretending an LLM answered.

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | *(unset)* | Enables LLM answers. Get one at <https://aistudio.google.com/apikey>. **Treat as a secret.** |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model id. Google retires model ids regularly (the old `gemini-1.5-flash` default isn't on Google's current model list, and `gemini-2.0-flash` was shut down in June 2026) — check <https://ai.google.dev/gemini-api/docs/models> if `/ask` starts returning `502`. |
| `GEMINI_TIMEOUT_SECONDS` | `30` | Per-request timeout for the Gemini call. |
| `EMBEDDING_BACKEND` | `tfidf` (`huggingface` in the Docker image) | `tfidf` or `huggingface` — see [section 3](#3-embedding-backends--read-this-before-citing-numbers). |
| `API_KEY` | *(unset)* | If set, every endpoint except `/health` requires the header `X-API-Key: <value>`. **Set this on any public deployment** — otherwise anyone can upload files and spend your Gemini quota. |
| `MAX_UPLOAD_MB` | `10` | Reject uploads larger than this. |
| `UPLOAD_DIR` | `data/uploads` (`/app/data/uploads` in Docker) | Where uploaded PDFs are stored. |
| `DOCS_DIR` | `data/sample_docs` | Bundled read-only corpus indexed at startup. |
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `PORT` | `8080` in Docker | Port the container listens on (Render/Cloud Run set this for you). |

A template lives in `.env.example`.

---

## 6. Step-by-step: from zip to a deployed API

> **Windows users:** use **Git Bash** or **WSL** to copy the commands below as-is. In PowerShell, activate the
> venv with `.venv\Scripts\Activate.ps1`, set variables with `$env:GEMINI_API_KEY="..."`, and use `curl.exe`
> (plain `curl` is an alias for something else) — or just use the Swagger UI at `/docs`.

### Step 0 — Prerequisites

- Python 3.11+ and `git`
- A free Gemini API key from <https://aistudio.google.com/apikey> (optional for local testing — without it
  `/ask` returns the retrieved chunks only)
- **Docker Desktop** (for Steps 5 and 7)
- A GitHub account (Step 6), and an account on Render or Google Cloud (Step 7)

### Step 1 — Unzip and set up a virtual environment

```bash
unzip rag_research_analyst.zip && cd rag_research_analyst
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
```

Install **one** of these:

```bash
pip install -r requirements.txt      # everything: API + Streamlit UI + HuggingFace embeddings (installs PyTorch, ~GBs)
# or, lightweight (API + tests only; use the TF-IDF backend):
pip install -r requirements-dev.txt
```

### Step 2 — Run the API locally

```bash
export EMBEDDING_BACKEND=tfidf                # use "huggingface" if you installed requirements.txt
export GEMINI_API_KEY="your-key-here"         # optional
uvicorn api.main:app --reload --port 8000
```

Startup logs a line like `Index ready: 48 pages, N chunks, backend=tfidf` when the real document has been
downloaded (see the quick start), or `3 pages, 6 chunks` with only the three bundled samples. The bundled corpus is
indexed automatically.

### Step 3 — Try it

Open **<http://localhost:8000/docs>** and use *Try it out* on each endpoint, or use curl:

```bash
curl http://localhost:8000/health

# ask about the bundled corpus (real NIST document if downloaded; otherwise ask the sample question below)
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What are the four functions of the AI RMF Core?", "k": 4}'
# sample-corpus question: "How many days of paid annual leave do full-time employees get?"

# upload your own text-based PDF, then ask about it
curl -X POST http://localhost:8000/upload -F "file=@/path/to/your_paper.pdf"
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What is the main conclusion of the paper?"}'

curl http://localhost:8000/documents
curl -X DELETE http://localhost:8000/documents/your_paper.pdf
```

If you set `API_KEY`, add `-H "X-API-Key: $API_KEY"` to each call (in Swagger UI click **Authorize**).

### Step 4 — Run the tests and linter

```bash
pytest          # 66 tests (+2 that run once the real PDF is downloaded): pipeline, parsing, eval maths, API, auth, uploads, Gemini wrapper
ruff check .    # lint
```

The tests run fully offline: they use the TF-IDF backend and a fake Gemini client, so they need no API key,
no network and no model downloads. They cover: valid and invalid uploads (fake/corrupt/empty/oversized/
scanned PDFs, hostile file names), replace-on-reupload, protection of sample docs, delete, `/ask` validation,
LLM success and LLM failure (`502` without leaking internals), API-key enforcement, and index persistence
across restarts, plus header/footer stripping, text normalisation, the evaluation metrics, and the schema of the
real-document question file. The two tests that need the real PDF (page count, every label present) skip themselves
until you run `python scripts/fetch_real_doc.py`. **CI runs the offline tests.**

### Step 5 — Build and run the Docker image locally

```bash
# Production image: CPU-only PyTorch + the MiniLM model baked in (first build takes several minutes)
docker build -t rag-research-analyst .
docker run --rm -p 8080:8080 -e GEMINI_API_KEY="$GEMINI_API_KEY" rag-research-analyst

# Light image (TF-IDF backend, ~300 MB) — same code, no PyTorch:
docker build --build-arg EMBEDDING_BACKEND=tfidf -t rag-research-analyst:light .
docker run --rm -p 8080:8080 -e GEMINI_API_KEY="$GEMINI_API_KEY" rag-research-analyst:light
```

Then `curl http://localhost:8080/health` (or open <http://localhost:8080/docs>).
You can also pass a whole env file: `docker run --rm -p 8080:8080 --env-file .env rag-research-analyst`
(copy `.env.example` to `.env` first; the file must not contain quotes around values).

What the `Dockerfile` does, and why:
- installs dependencies **before** copying code so rebuilds after a code change are fast (layer caching);
- installs **CPU-only** PyTorch (~200 MB instead of a ~2 GB CUDA build) for the HuggingFace backend;
- downloads the embedding model **at build time**, so containers start quickly and never depend on
  huggingface.co at runtime;
- runs as a **non-root** user, with a `HEALTHCHECK` hitting `/health`;
- listens on `$PORT` (default 8080), which is how Render and Cloud Run tell the container where to listen;
- starts **one** uvicorn worker (see [Design decisions](#9-design-decisions--scaling-notes)).

### Step 6 — Push to GitHub and watch CI run

```bash
git init
git add .
git status                            # confirm .env is NOT listed (it is git-ignored)
git commit -m "RAG research analyst: FastAPI service, Docker, CI"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git   # create the empty repo on github.com first
git push -u origin main
```

Open the repo's **Actions** tab. `.github/workflows/ci.yml` runs on **every push and pull request**:

| Job | What it does |
|---|---|
| **Lint + tests** | installs `requirements-dev.txt` → `ruff check .` → `pytest` |
| **Docker build + smoke test** *(runs after tests pass)* | builds the image, starts the container, calls `/health` and `/ask`, and checks the answer's sources include `company_handbook.pdf` |

Then replace `<your-username>/<your-repo>` in the badge at the top of this README. If GitHub prints a
warning about action versions being outdated, bump the `@v4` / `@v5` versions in `ci.yml`.

> Never put your Gemini key in the repo or in `ci.yml`. CI doesn't need it. On the deployment platforms
> below you enter it as a *secret / environment variable*.

### Step 7 — Deploy

Pick **one**. All three run the **same Docker image** from Step 5.

| | Render | Google Cloud Run | AWS |
|---|---|---|---|
| Effort | easiest, fully in the browser | a few `gcloud` commands | most setup |
| Backend to use | `tfidf` (free plan has 512 MB RAM) | `huggingface` (give it 2 GiB) | `huggingface` (≥ 2 GB memory) |
| Auto-deploy on `git push` | yes (built in) | add a Cloud Build trigger (optional) | add a pipeline (optional) |

#### 7A — Render (free tier, easiest)

1. Complete Step 6 (the repo must be on GitHub; `render.yaml` in the repo root defines the service).
2. On <https://dashboard.render.com>: **New + → Blueprint** → connect GitHub → select your repo.
3. Render reads `render.yaml`. When prompted, paste your `GEMINI_API_KEY`. (`API_KEY` is generated for you.)
4. Click **Apply** and wait for the Docker build to finish. You get a URL like
   `https://rag-research-analyst-xxxx.onrender.com`.
5. Open the service → **Environment** tab → copy the generated `API_KEY` value.
6. Verify (Step 8).

If the Render build installs PyTorch anyway (the `EMBEDDING_BACKEND` build argument may not reach the Docker
build), the service still runs correctly on `tfidf` — the build is just slower and the image larger.

Notes: the free plan sleeps after a period of inactivity, so the first request afterwards is slow (cold
start), and its disk is ephemeral — uploads disappear on restart/redeploy. Check Render's current free-tier
limits on their pricing page. With `autoDeploy: true`, every push to your default branch redeploys.

#### 7B — Google Cloud Run

Cloud Run needs a Google Cloud project with billing enabled (it scales to zero, and has a free tier — check
current pricing). `asia-south1` is Mumbai; pick any region you like.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
                       artifactregistry.googleapis.com secretmanager.googleapis.com

# 1) Store secrets in Secret Manager (never in the image or the repo)
printf '%s' "YOUR_GEMINI_KEY" | gcloud secrets create gemini-api-key --data-file=-
openssl rand -hex 24 | tr -d '\n' | gcloud secrets create rag-api-key --data-file=-

# 2) Allow Cloud Run's runtime service account to read them
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')
for s in gemini-api-key rag-api-key; do
  gcloud secrets add-iam-policy-binding $s \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
done

# 3) Build (Cloud Build uses the Dockerfile) and deploy in one command, from the repo root
gcloud run deploy rag-research-analyst \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi --cpu 1 --cpu-boost \
  --max-instances 1 \
  --timeout 120 \
  --set-env-vars EMBEDDING_BACKEND=huggingface \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest,API_KEY=rag-api-key:latest

# 4) Get your URL and your API key
gcloud run services describe rag-research-analyst --region asia-south1 --format='value(status.url)'
gcloud secrets versions access latest --secret=rag-api-key
```

`--allow-unauthenticated` only means Google won't require a Google login; **your app-level `API_KEY`
still protects `/ask`, `/upload` and `/documents`**. `--max-instances 1` matters — see the scaling notes.
The first build takes several minutes (PyTorch + model download). Then verify (Step 8).

To redeploy after a code change, re-run the `gcloud run deploy ...` command, or set up a Cloud Build
trigger on your GitHub repo so pushes to `main` deploy automatically.

#### 7C — AWS (outline — not tested here)

The image is portable, so any AWS container service works. The generic recipe (Amazon ECS on Fargate, or
App Runner if it's available on your account):

```bash
aws ecr create-repository --repository-name rag-research-analyst --region ap-south-1
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com

# --platform matters if you build on an Apple-Silicon Mac
docker build --platform linux/amd64 -t rag-research-analyst .
docker tag  rag-research-analyst:latest <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/rag-research-analyst:latest
docker push <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/rag-research-analyst:latest
```

Then create a service from that image with: container port **8080**, health check path **`/health`**,
**≥ 2 GB memory**, **one** running task/instance, `EMBEDDING_BACKEND=huggingface`, and `GEMINI_API_KEY` /
`API_KEY` injected from **AWS Secrets Manager**. Follow the AWS console wizard or docs for the rest.

### Step 8 — Verify the deployment

```bash
URL="https://<your-service-url>"      # from Step 7
API_KEY="<your key>"

curl "$URL/health"
# -> {"status":"ok","backend":"...","documents":1,"chunks":N,"llm_configured":true,"model":"..."}
#    (documents 1 = the real NIST PDF was baked into the image; 3 / 6 chunks = the sample PDFs only)
#    llm_configured must be true, otherwise your GEMINI_API_KEY secret didn't reach the container.

curl -X POST "$URL/ask" -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
     -d '{"question": "How many days of paid annual leave do full-time employees get?"}'
# -> "generated": true and an answer citing [file.pdf, p.N]  (use the NIST question when the real PDF is in the image)

curl -X POST "$URL/upload" -H "X-API-Key: $API_KEY" -F "file=@your_paper.pdf"
curl -s "$URL/documents" -H "X-API-Key: $API_KEY"
curl -i "$URL/ask" -X POST -H "Content-Type: application/json" -d '{"question":"x"}'   # no key -> 401
```

### Step 9 — Hardening checklist (before sharing the URL publicly)

- [ ] `API_KEY` is set (Step 8 shows `401` without it)
- [ ] Your Gemini key is in the platform's secret store, not in git
- [ ] Set a budget/quota alert on your Gemini and cloud accounts
- [ ] If you ever leaked a key, revoke it in Google AI Studio and create a new one
- [ ] Add your live URL and the CI badge to your resume / GitHub README

---

## 7. Verify generation with a live Gemini key

The generation code (`src/generate.py`) was tested with a fake client but **not** against the live Gemini API
(no network access in the build environment). Verify it once with your own key:

```bash
export GEMINI_API_KEY="your-key-here"
python src/rag_pipeline.py "What are the four functions of the AI RMF Core?" --backend huggingface --corpus real
```

You should see an answer naming GOVERN, MAP, MEASURE and MANAGE with a citation like `[NIST.AI.100-1.pdf, p.N]`. If you get a `404`/"model not found"
error, set `GEMINI_MODEL` to a current model id from <https://ai.google.dev/gemini-api/docs/models>.
Then produce your own citation-accuracy / hallucination figures:

```bash
python eval/evaluate.py --corpus real --backend huggingface --sleep 4   # writes outputs/eval_real_huggingface.json
```

## 8. Repo structure

```
rag_research_analyst/
├── api/                        # FastAPI service
│   ├── main.py                 #   routes, auth dependency, error mapping
│   ├── service.py              #   thread-safe index owner: upload / delete / ask
│   ├── schemas.py              #   request/response models (validated by pydantic)
│   └── config.py               #   settings from environment variables
├── src/                        # the RAG pipeline
│   ├── parse.py                #   PDF -> per-page text + metadata; strips headers/footers, fixes hyphenation
│   ├── corpus.py               #   which PDFs to index by default (real document if present, else samples)
│   ├── textutil.py             #   text normalisation shared by the parser, evaluator and label validator
│   ├── chunking.py             #   overlapping chunking
│   ├── embed_store.py          #   HuggingFace / TF-IDF embeddings + FAISS
│   ├── generate.py             #   Gemini call + grounding guardrail prompt
│   └── rag_pipeline.py         #   ties retrieval + generation together (CLI)
├── tests/                      # pytest suite (runs offline)
│   ├── helpers.py              #   dependency-free PDF builder for tests
│   ├── conftest.py             #   client fixtures with isolated upload dirs
│   ├── test_pipeline.py        #   parse / chunk / retrieval / prompt
│   ├── test_generate.py        #   Gemini wrapper (fake client)
│   ├── test_parse_clean.py     #   header/footer stripping, hyphenation, multi-page parsing
│   ├── test_textutil.py        #   normalisation + keyword matching
│   ├── test_eval.py            #   Hit@k / MRR / citation parsing maths, CLI smoke test
│   ├── test_real_corpus.py     #   question-file schema; real-PDF checks (skip until downloaded)
│   └── test_api.py             #   endpoints, validation, auth, persistence
├── eval/
│   ├── qa_real.json            #   36 questions on the real NIST document (32 answerable + 4 unanswerable)
│   ├── qa_set.json             #   16 page-labelled questions on the 3 sample PDFs
│   ├── evaluate.py             #   retrieval + generation evaluation harness
│   └── validate_qa.py          #   checks every label exists in your PDF
├── scripts/
│   └── fetch_real_doc.py       #   downloads + verifies NIST.AI.100-1.pdf
├── data/
│   ├── real_docs/              #   the real 48-page document goes here (README.md explains; PDF is downloaded)
│   ├── make_sample_docs.py     #   generates 3 sample PDFs with known facts
│   └── sample_docs/            #   bundled synthetic corpus for tests/CI
├── outputs/eval_<corpus>_<backend>.json   # run output (eval_sample_tfidf.json shipped as an example)
├── .github/workflows/ci.yml    # CI: lint + tests + Docker build/smoke test
├── Dockerfile                  # production image
├── .dockerignore  .gitignore  .env.example
├── render.yaml                 # Render Blueprint
├── app.py                      # Streamlit chat UI (streamlit run app.py)
├── requirements.txt            # everything, for local development
├── requirements-api.txt        # API runtime (what the Docker image installs)
├── requirements-hf.txt         # HuggingFace embeddings (PyTorch)
├── requirements-dev.txt        # what CI installs: API + pytest + ruff
└── pytest.ini  ruff.toml
```

The original entry points still work: `python src/rag_pipeline.py "..."`, `python eval/evaluate.py` (also the old
`python eval/evaluate.py huggingface` form), and `streamlit run app.py` (Streamlit UI: `pip install -r requirements.txt` first).

## 9. Design decisions & scaling notes

These are worth being able to explain in an interview.

- **Immutable index, atomic swap.** An upload builds a *new* index in memory and swaps it in with a single
  assignment, after the PDF has been validated. In-flight `/ask` requests keep using the old, consistent
  index; a failed upload can never leave the index and the files on disk disagreeing. Writers are serialised
  with a lock.
- **Validation at the boundary.** Uploads are checked for extension, `%PDF` header, size, readability and
  extractable text; file names are sanitised (no path traversal); staging happens in a private temp
  directory. Sample documents can't be overwritten or deleted.
- **Honest degradation.** No Gemini key → `generated: false` plus sources. Gemini failure → `502` with a
  generic message (details go to server logs, not to the client). Gemini calls have a timeout so a stuck
  upstream call can't hang a worker.
- **Config in the environment**, secrets in the platform's secret store — the image itself contains no secrets.
- **One worker, one instance — on purpose.** The FAISS index lives in process memory. With several workers or
  instances, an upload handled by one wouldn't be visible to the others. That's why the deploy commands use
  `--workers 1` / `--max-instances 1`. It is fine for a demo or a small internal tool; to scale out, move the
  state out of the process.
- **Uploads are ephemeral.** Container disks are wiped on restart/redeploy (Cloud Run's is in-memory). The
  bundled samples always come back; uploaded PDFs don't.
- **Every upload re-indexes the whole corpus.** Fine for tens of documents. With the HuggingFace backend this
  re-embeds every chunk each time.

**If you wanted to take this further** (good "next steps" to mention): store PDFs in object storage
(GCS/S3) and vectors in a managed vector DB (pgvector, Qdrant, Pinecone) so the service becomes stateless
and can scale horizontally; embed only new chunks (incremental `index.add`); move large uploads to a
background job queue; add rate limiting and real per-user auth (OAuth/JWT); add CORS if a browser frontend
calls it; add OCR for scanned PDFs; export metrics (latency, retrieval scores, refusal rate) to monitor
answer quality over time; add a Cloud Build/GitHub Actions deploy step for continuous delivery.

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Startup error `No module named 'sentence_transformers'` | `EMBEDDING_BACKEND=huggingface` but PyTorch/sentence-transformers aren't installed. `pip install -r requirements-hf.txt`, or use `EMBEDDING_BACKEND=tfidf`. |
| `/ask` returns `502` | Read the server logs. Usual causes: invalid/expired `GEMINI_API_KEY`, retired model id (set `GEMINI_MODEL`), quota exceeded, or no outbound network. |
| `/ask` returns `"generated": false` | `GEMINI_API_KEY` isn't reaching the process. Check `/health` → `llm_configured`. |
| `401 Missing or invalid X-API-Key` | `API_KEY` is set on the server; send `-H "X-API-Key: ..."` (Swagger UI: **Authorize**). |
| Upload `422 No extractable text` | The PDF is a scan/image. OCR isn't included; use a text-based PDF. |
| Upload `413` | File exceeds `MAX_UPLOAD_MB`. Raise it (and your platform's request-size limit, if any). |
| Render deploy crashes / restarts | Out of memory: use `EMBEDDING_BACKEND=tfidf` on the 512 MB free plan (`render.yaml` already does). |
| Cloud Run "container failed to start and listen on PORT" | Check the logs (`gcloud run services logs read rag-research-analyst --region asia-south1`); with the HuggingFace backend give it `--memory 2Gi`. |
| Docker build is slow / huge | Use `--build-arg EMBEDDING_BACKEND=tfidf` for the light image. |
| CI **Docker** job fails | Open the failed step; the last step prints the container logs. |
| Uploaded files vanished | Expected on ephemeral disks after a restart/redeploy — see the scaling notes. |
| `curl` behaves oddly on Windows PowerShell | Use `curl.exe`, Git Bash, or the Swagger UI at `/docs`. |

## 11. What is verified vs. what needs your own run

**Verified in the build environment**
- 66 automated tests pass, 2 skip (they need the real PDF), and `ruff` is clean: pipeline, parsing, eval maths, API,
  upload validation, auth, restart persistence, Gemini wrapper with a fake client.
- The API was started with `uvicorn` and exercised end-to-end with curl (health, ask, upload → ask over the
  new document → documents → delete, error codes, Swagger UI, OpenAPI schema).
- The container layout, `CMD` and `HEALTHCHECK` command were exercised outside Docker (copying only what the
  Dockerfile copies).
- The real Gemini SDK client is constructed and its request is issued; the resulting failure (the sandbox
  blocks the Gemini host) is turned into a clean `502`.
- Parsing, chunking, indexing and retrieval on the sample corpus: `outputs/eval_sample_tfidf.json` is a real,
  reproducible run (TF-IDF backend).
- The real-document code path (header/footer removal, hyphen re-joining, label validation, evaluation, API and
  Streamlit corpus switching) was exercised on a throwaway 48-page stand-in PDF with running headers and footers.

**Not verified here — please check on your machine / on first push**
- **The real NIST PDF itself.** The build environment could not reach nist.gov, so the PDF is not in the zip and the
  numbers on it have **not** been measured. The 36 questions were written from the document's text, but run
  `python eval/validate_qa.py` first: it proves each label exists in your copy and reports any that don't. Then run
  the evaluation yourself — nothing in this repo claims a retrieval or citation score on the real document.
- **`docker build`** — Docker wasn't available in the build environment. The CI "Docker build + smoke test"
  job builds the image and hits `/health` and `/ask` on your first push, so you'll see immediately if
  anything needs a tweak.
- **The HuggingFace backend** (no access to huggingface.co) and **live Gemini generation** (Step 7 above).
- **The cloud deployments** (Render, Cloud Run, AWS) — the steps follow each platform's standard flow, but
  platforms change; if a flag or field has been renamed, the platform's error message will point to it.

The citation-accuracy and hallucination-reduction figures still have to come from your own run of
`eval/evaluate.py` with your own Gemini key — do not quote numbers you haven't measured.

## 12. Changes from the original project

**Latest round — real-world corpus**
- **New:** `data/real_docs/` + `scripts/fetch_real_doc.py` (NIST AI 100-1, 48 pages), `eval/qa_real.json` (36
  phrase-labelled questions), `eval/validate_qa.py`, `src/corpus.py`, `src/textutil.py`, and tests for each.
- **`src/parse.py`:** removes running headers/footers, re-joins hyphenated line breaks and collapses whitespace
  (essential for real PDFs; no effect on the 1-page sample files).
- **`eval/evaluate.py`:** rewritten — Hit@1/3/5/10, MRR@10, retrieval and end-to-end latency, citation accuracy and
  citation *validity*, per-question logs, `--corpus/--k/--chunk-size/--overlap/--gen-limit/--sleep/--min-hit-rate`.
  The old `python eval/evaluate.py huggingface` form still works. Results now go to
  `outputs/eval_<corpus>_<backend>.json`.
- **API / Docker / Streamlit:** index `data/real_docs/` automatically when it contains a PDF (else the samples);
  `/documents` reports bundled files as `origin: "sample"` for compatibility even when they are the real document.
  Tests pin the sample corpus so they are unaffected.

**Earlier round — production packaging**

- **New:** `api/` (FastAPI service with `/health`, `/documents`, `/upload`, `/ask`, `DELETE /documents/{name}`,
  optional API-key auth), `tests/`, `Dockerfile`, `.dockerignore`, `render.yaml`, `.github/workflows/ci.yml`,
  `pytest.ini`, `ruff.toml`, `.env.example`, `.gitignore`.
- **`src/generate.py`:** migrated from the deprecated `google-generativeai` package to `google-genai`, added a
  request timeout, and made the model configurable via `GEMINI_MODEL` (default `gemini-3.8-flash`, a stable model
  on Google's model list when this was written). The old hard-coded default `gemini-1.5-flash` is no longer on
  that list, so a deployed service relying on it would likely fail. The grounding prompt is unchanged.
- **`src/chunk.py` → `src/chunking.py`:** `chunk` is also the name of a Python standard-library module (removed
  in 3.13), so importing it can be shadowed depending on import order. Imports updated in `rag_pipeline.py`
  and `eval/evaluate.py`; the chunking logic is unchanged.
- **Requirements** split into `requirements-api.txt` / `-hf.txt` / `-dev.txt`; `requirements.txt` still
  installs everything for local development.
- Removed a stray `{data/` directory (from a mistyped `mkdir`) and `__pycache__`; removed an unused import.
