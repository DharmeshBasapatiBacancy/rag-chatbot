# Document RAG Chatbot (Python + LangChain + Gemini)

A chatbot that answers questions **only** from a document you upload (PDF or
DOCX), remembers the conversation, and says **"I don't have that information."**
when the answer isn't in the document.

Built for a learning task. Base demo document: a fictional sports league
rulebook (`data/league_rules.md`, also generated as PDF/DOCX).

## What it does (the six requirements)
1. **Loads** a document — PDF, DOCX, TXT, or MD.
2. **Splits** it into overlapping chunks (`RecursiveCharacterTextSplitter`).
3. **Embeds** the chunks (`gemini-embedding-001`) and stores them in a Chroma
   vector store.
4. **Retrieves** the most relevant chunks per question (similarity search).
5. Sends **chunks + chat history + question** to the LLM (`gemini-3.1-flash-lite`).
6. **Answers only from the document** — otherwise replies exactly
   `"I don't have that information."` (temperature=0 + a strict grounding prompt).

## Project layout
```
app.py            # Streamlit chat UI with PDF/DOCX upload
rag.py            # RAG core: load, split, embed, retrieve, answer + guardrail
check_model.py    # Verifies pinned models are LIVE via REAL API calls (run first!)
verify_rag.py     # Full end-to-end test (in-doc, refusal, follow-up)
verify_followup.py# Minimal memory + refusal test (few API calls)
data/             # Demo rulebook + fixture generator (md/pdf/docx)
requirements.txt
.env              # GOOGLE_API_KEY + pinned model names (git-ignored)
```

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
echo 'EMBED_MODEL=gemini-embedding-001' >> .env
```

Get a free key at https://aistudio.google.com.

## Run

```bash
# ALWAYS verify models are live first (they get retired without notice)
.venv/bin/python check_model.py

# Launch the chatbot
.venv/bin/streamlit run app.py
```

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
   EMBED_MODEL = "gemini-embedding-001"
   ```
6. Click **Deploy**. You'll get a public `*.streamlit.app` URL to share.

Notes:
- Never commit `.env` or `.streamlit/secrets.toml` — both are git-ignored.
  Cloud uses the dashboard **Secrets**; local uses `.env`. The app reads both.
- Uploaded documents and the vector store live only in memory per session —
  nothing is persisted server-side.
- The free Gemini tier is rate-limited (see above); a shared team app can hit
  those limits faster, so consider a paid key for heavier use.
