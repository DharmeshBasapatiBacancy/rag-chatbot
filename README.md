# Document RAG Chatbot (Python + LangChain + Gemini)

A chatbot that answers questions **only** from a document you upload (PDF or
DOCX), remembers the conversation, and says **"I don't have that information."**
when the answer isn't in the document.

Built for a learning task. Base demo document: a fictional sports league
rulebook (`data/league_rules.md`, also generated as PDF/DOCX).

## What it does (the pipeline)
1. **Parses** a document — PDF, DOCX, TXT, or MD — with structure awareness:
   extracts body text **and tables** (rendered to Markdown so numbers keep their
   row/column headers), de-scrambles **two-column** layouts, strips running
   headers/footers, and tags every block with its **chapter/section breadcrumb**
   and page (`parsing.py`).
2. **Chunks** narrative text into overlapping pieces and keeps tables whole; each
   chunk is prefixed with its breadcrumb so the retriever/LLM know which
   section/region it belongs to.
3. **Embeds** chunks with a **local** sentence-transformers model
   (`all-MiniLM-L6-v2`) into an in-memory vector store — no API quota or rate
   limit, works offline.
4. **Retrieves** with a **hybrid** pipeline: BM25 (exact keywords/numbers) +
   dense MMR (meaning), fused by an `EnsembleRetriever`, then re-ranked by a
   local **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) for precision. Candidate
   pool scales with document size.
5. Sends **chunks + chat history + question** to the LLM (`gemini-3.1-flash-lite`)
   with a concise-answer, section-aware grounding prompt.
6. **Reads charts & figures.** When a question references a figure/chart
   (e.g. *"Figure 3.1 chart D"*), the relevant PDF page is rendered to an image
   (PyMuPDF) and sent to the multimodal model, which reads values straight off
   the bars/lines — facts that live only in the graphic, not the text layer.
   A figure-label→page map (built at index time, picking the real chart page over
   the list-of-figures) locates the right page. Vision runs **only** for visual
   questions, so normal queries stay fast and quota-free.
7. **Answers only from the document** — otherwise replies exactly
   `"I don't have that information."` (temperature=0 + a strict grounding prompt).

These upgrades target the three document shapes in `RAG-Hands-On-Use-Cases`:
plain narrative (GPO report), long structured report (World Bank GEP — keep
*East Asia* separate from *Sub-Saharan Africa*), and short visual docs where
facts live in tables/charts/footnotes (iMac environmental report).

## Project layout
```
app.py            # Streamlit chat UI; shows breadcrumb+page sources per answer
parsing.py        # Structure-aware parsing: tables, 2-column, heading breadcrumbs
rag.py            # Chunk, local-embed, hybrid retrieve + rerank, grounded answer
check_model.py    # Verifies pinned models are LIVE via REAL API calls (run first!)
verify_rag.py     # Full end-to-end test (in-doc, refusal, follow-up)
verify_followup.py# Minimal memory + refusal test (few API calls)
data/             # Demo rulebook + fixture generator (md/pdf/docx)
requirements.txt
.env              # GOOGLE_API_KEY + pinned model names (git-ignored)
```

## Embeddings run locally (no quota)
Embeddings use a local sentence-transformers model, **not** the Gemini embedding
API. A large PDF produces thousands of chunks, and the Gemini free tier caps
embeddings at ~100/minute — so Gemini embeddings 429-fail on real documents.
Local embeddings have no such limit and keep Gemini usage to answer-generation
only. Tunable via `.env`: `EMBED_MODEL`, `RERANK_MODEL`, and `USE_RERANKER=0` to
disable the cross-encoder for lower latency. First upload downloads the models
(~90 MB embedder + ~80 MB reranker) once.

## Setup

This project uses **Python 3.11** via [`uv`](https://docs.astral.sh/uv/)
(the system Python 3.8 is too old for modern LangChain).

```bash
# 1. Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Create the environment and install deps
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# 3. Add your key to .env
echo 'GOOGLE_API_KEY=your_key_here' > .env
echo 'GEN_MODEL=gemini-3.1-flash-lite' >> .env
echo 'EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2' >> .env
```

Get a free key at https://aistudio.google.com.

## Run

```bash
# ALWAYS verify models are live first (they get retired without notice)
.venv/bin/python check_model.py

# Launch the chatbot (run.sh sets the env vars that keep it stable)
./run.sh
```

`run.sh` exports `JE_ARROW_MALLOC_CONF=background_thread:false` — the key fix: it
disables the pyarrow/jemalloc **background thread** that otherwise segfaults and
kills the Streamlit server with no traceback (seen crashing mid-upload while
indexing). Arrow's jemalloc is built with the `je_arrow_` prefix, so it reads
`JE_ARROW_MALLOC_CONF`, **not** `MALLOC_CONF`. It also sets
`ARROW_DEFAULT_MEMORY_POOL=system` and OpenMP guards for torch/pymupdf. The
watcher is disabled in `.streamlit/config.toml`.

Then open the browser tab, upload a PDF or DOCX in the sidebar, and ask away.

To run the automated tests:
```bash
.venv/bin/python verify_followup.py   # light (few calls)
.venv/bin/python verify_rag.py        # full suite (uses more quota)
```

## Notes on Gemini models (important)
- **Pin the model and verify it's live.** `check_model.py` makes a *real*
  generate + embed call, because `list_models()` can still list a model that is
  actually closed to new users (returns 404 on use). We hit this: `gemini-2.0-flash`
  and `gemini-2.5-flash` were unusable for a fresh key; `gemini-3.6-flash` and
  `gemini-3.1-flash-lite` work.
- **Free-tier quota is small.** `gemini-3.6-flash` is ~20 requests/day on the
  free tier; `gemini-3.1-flash-lite` has more generous limits, so it's the
  default here. Switch `GEN_MODEL` in `.env` to trade quality vs. quota.
- **Use only public/dummy documents** — free-tier content may be used to
  improve the provider's models.

## How the anti-hallucination guardrail works
- `temperature=0` (no creative drift).
- A system prompt that forbids outside knowledge and mandates the exact refusal
  sentence when the context doesn't contain the answer.
- Retrieved chunks are injected as `<context>`; if retrieval is empty, the model
  is instructed to refuse.
- A history-aware retriever rewrites follow-up questions (e.g. resolving "that
  player") into standalone queries before retrieval.

## Deploying to Streamlit Community Cloud (share with your team)

1. Push this repo to GitHub (public or private — Community Cloud supports both).
2. Go to https://share.streamlit.io → **Create app** → **Deploy from GitHub**.
3. Select your repo/branch, set **Main file path** to `app.py`.
4. Under **Advanced settings**, set **Python version** to **3.11**.
5. Open **Secrets** and paste (with your real key):
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   GEN_MODEL = "gemini-3.1-flash-lite"
   EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   ```
6. Click **Deploy**. You'll get a public `*.streamlit.app` URL to share.

Notes:
- Never commit `.env` or `.streamlit/secrets.toml` — both are git-ignored.
  Cloud uses the dashboard **Secrets**; local uses `.env`. The app reads both.
- Uploaded documents and the vector store live only in memory per session —
  nothing is persisted server-side.
- The free Gemini tier is rate-limited (see above); a shared team app can hit
  those limits faster, so consider a paid key for heavier use.
