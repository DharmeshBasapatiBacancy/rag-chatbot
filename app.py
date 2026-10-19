"""Docullence - a minimalist, document-grounded RAG chatbot.

Upload a PDF or DOCX, then ask questions. Docullence answers ONLY from the
uploaded document and says "I don't have that information." otherwise.

Run:  streamlit run app.py
"""
import os

# Native-library safety.
# 1) pyarrow (pulled in by Streamlit) bundles jemalloc, whose background thread
#    segfaults intermittently (crash signature: "jemalloc_bg_thd ... segfault in
#    libarrow.so"), killing the server with no Python traceback - sometimes
#    mid-upload. Using the system allocator removes that thread. NOTE: pyarrow is
#    imported before this file runs, so this is ALSO exported in the launch
#    command; kept here as a safety net.
# 2) torch (sentence-transformers / cross-encoder) and MuPDF (pymupdf) use
#    OpenMP; duplicate OpenMP init inside Streamlit's worker thread can abort the
#    process. Pin threads and allow duplicate OMP libs.
# Must run BEFORE heavy imports (i.e. before `from rag import ...`).
os.environ.setdefault("JE_ARROW_MALLOC_CONF", "background_thread:false")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import tempfile

import streamlit as st
from dotenv import load_dotenv

# Config loading works in BOTH environments:
#  - Local dev: reads .env (git-ignored).
#  - Streamlit Community Cloud: reads secrets set in the app dashboard.
# We copy secrets into os.environ so rag.py picks them up identically in both
# places. This must run BEFORE importing rag.
load_dotenv()
try:
    for _k in ("GOOGLE_API_KEY", "GEN_MODEL", "EMBED_MODEL"):
        if _k not in os.environ and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets configured (e.g. pure local run) -> rely on .env

from langchain_core.messages import AIMessage, HumanMessage

from rag import build_index, build_chain, ask

APP_NAME = "DOCULLENCE"
TAGLINE = "document intelligence · answers grounded in your file"

st.set_page_config(page_title="Docullence · Document Intelligence", page_icon="🟢")

# ---------------------------------------------------------------------------
# Minimalist black & green theme
# ---------------------------------------------------------------------------
THEME_CSS = """
<style>
:root{
  --bg:#0A0E0C; --panel:#101613; --panel-2:#0C120F;
  --border:#1E2A24; --text:#DCE7E1; --muted:#7A8C83;
  --accent:#1FD68A; --accent-2:#2AF0A0; --accent-dim:rgba(31,214,138,.10);
  --mono:ui-monospace,"SF Mono","JetBrains Mono","Fira Code",Menlo,Consolas,monospace;
}
.stApp{background:var(--bg);color:var(--text);}
/* strip Streamlit chrome for a minimal canvas */
#MainMenu,header[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stDecoration"],[data-testid="stStatusWidget"],footer{display:none!important;}
.block-container{max-width:840px;padding-top:2.2rem;padding-bottom:7rem;}

/* ---- brand header ---- */
.vir-brand{display:flex;align-items:center;gap:.6rem;}
.vir-logo{color:var(--accent);font-size:1.5rem;line-height:1;
  text-shadow:0 0 12px rgba(31,214,138,.55);}
.vir-name{font-family:var(--mono);font-weight:700;letter-spacing:.30em;
  font-size:1.5rem;color:var(--text);}
.vir-tag{color:var(--muted);font-size:.82rem;letter-spacing:.03em;margin:.35rem 0 0 2.1rem;}
.vir-rule{height:1px;background:linear-gradient(90deg,var(--accent),transparent);
  margin:1.1rem 0 1.5rem;opacity:.55;}

/* ---- empty state ---- */
.vir-empty{color:var(--muted);font-family:var(--mono);font-size:.9rem;
  border:1px dashed var(--border);border-radius:12px;padding:2.4rem 1.5rem;
  text-align:center;background:var(--panel-2);}
.vir-empty b{color:var(--accent);font-weight:600;}

/* ---- chat messages: bubbles with side alignment + icon badges ---- */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"]{display:none!important;}
[data-testid="stChatMessage"]{width:fit-content;max-width:84%;align-items:flex-start;
  gap:.55rem;padding:.35rem 1.1rem;border-radius:14px;margin-bottom:.75rem;}
[data-testid="stChatMessageContent"]{color:var(--text);min-width:0;}
/* icon badge shared */
[data-testid="stChatMessage"]::before{flex:0 0 auto;width:28px;height:28px;
  display:flex;align-items:center;justify-content:center;border-radius:50%;
  font-family:var(--mono);margin-top:.3rem;}

/* assistant -> left */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){
  margin-right:auto;background:var(--panel);border:1px solid var(--border);
  border-left:2px solid var(--accent);}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])::before{
  content:"◆";color:var(--accent);background:var(--accent-dim);font-size:.82rem;
  border:1px solid rgba(31,214,138,.4);box-shadow:0 0 10px rgba(31,214,138,.28);}

/* user -> right */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  margin-left:auto;flex-direction:row-reverse;background:var(--accent-dim);
  border:1px solid rgba(31,214,138,.22);}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])::before{
  content:"\\203A";color:var(--muted);background:transparent;font-size:1.05rem;
  border:1px solid var(--border);}

/* ---- chat input ---- */
[data-testid="stChatInput"]{background:var(--panel)!important;border:1px solid var(--border)!important;
  border-radius:12px!important;}
[data-testid="stChatInput"]:focus-within{border-color:var(--accent)!important;
  box-shadow:0 0 0 1px var(--accent),0 0 18px rgba(31,214,138,.15)!important;}
/* gentle breathing glow to draw attention while the box is empty + idle;
   :placeholder-shown stops matching the instant the user types */
@keyframes virPulse{
  0%,100%{box-shadow:0 0 0 1px rgba(31,214,138,.14);}
  50%{box-shadow:0 0 0 1px rgba(31,214,138,.55),0 0 20px rgba(31,214,138,.22);}
}
[data-testid="stChatInput"]:has(textarea:placeholder-shown):not(:focus-within){
  animation:virPulse 2.4s ease-in-out infinite;}
@media (prefers-reduced-motion:reduce){
  [data-testid="stChatInput"]:has(textarea:placeholder-shown):not(:focus-within){
    animation:none;border-color:rgba(31,214,138,.4)!important;}}
[data-testid="stChatInput"] textarea{color:var(--text)!important;}
[data-testid="stChatInput"] textarea::placeholder{color:var(--muted)!important;}
/* send button: dark arrow on the green fill; muted when disabled (empty input) */
[data-testid="stChatInputSubmitButton"]{color:#07100B!important;}
[data-testid="stChatInputSubmitButton"] svg{color:#07100B!important;fill:#07100B!important;}
[data-testid="stChatInputSubmitButton"]:disabled,
[data-testid="stChatInputSubmitButton"]:disabled svg{
  color:var(--muted)!important;fill:var(--muted)!important;}

/* ---- sidebar ---- */
[data-testid="stSidebar"]{background:var(--panel-2);border-right:1px solid var(--border);}
.vir-side-label{font-family:var(--mono);font-size:.72rem;letter-spacing:.22em;
  color:var(--muted);text-transform:uppercase;margin:.2rem 0 .5rem;}
.vir-chip{display:inline-block;font-family:var(--mono);font-size:.78rem;color:var(--accent);
  background:var(--accent-dim);border:1px solid rgba(31,214,138,.3);
  border-radius:6px;padding:.25rem .6rem;word-break:break-all;}
.vir-meta{color:var(--muted);font-family:var(--mono);font-size:.72rem;margin-top:.2rem;}

/* file uploader */
[data-testid="stFileUploaderDropzone"]{background:var(--panel);border:1px dashed var(--border);
  color:var(--muted);}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent);}
[data-testid="stBaseButton-secondary"],.stButton>button{background:transparent;
  color:var(--accent);border:1px solid rgba(31,214,138,.4);border-radius:8px;}
[data-testid="stBaseButton-secondary"]:hover,.stButton>button:hover{
  border-color:var(--accent);background:var(--accent-dim);color:var(--accent-2);}

/* ---- alerts ---- */
[data-testid="stAlert"]{background:var(--panel);border:1px solid var(--border);border-radius:10px;}

/* ---- sources expander ---- */
[data-testid="stExpander"]{border:1px solid var(--border)!important;border-radius:10px;
  background:var(--panel-2);}
[data-testid="stExpander"] summary{color:var(--muted);font-family:var(--mono);font-size:.8rem;}
[data-testid="stExpander"] summary:hover{color:var(--accent);}
.vir-src-head{font-family:var(--mono);font-size:.76rem;color:var(--accent);margin:.3rem 0 .1rem;}

/* links + misc */
a,a:visited{color:var(--accent)!important;}
hr{border-color:var(--border);}
::-webkit-scrollbar{width:8px;height:8px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:var(--accent);}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

if not os.getenv("GOOGLE_API_KEY"):
    st.error(
        "GOOGLE_API_KEY is not set. Locally: add it to .env. "
        "On Streamlit Cloud: add it under App settings -> Secrets."
    )
    st.stop()

# ---- Brand header ----------------------------------------------------------
st.markdown(
    f"""
    <div class="vir-brand">
      <span class="vir-logo">◆</span>
      <span class="vir-name">{APP_NAME}</span>
    </div>
    <div class="vir-tag">{TAGLINE}</div>
    <div class="vir-rule"></div>
    """,
    unsafe_allow_html=True,
)

# ---- Session state ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of (role, text) for display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # LangChain messages for the LLM
if "chain" not in st.session_state:
    st.session_state.chain = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ---- Sidebar: upload + indexing -------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div class="vir-brand"><span class="vir-logo">◆</span>'
        f'<span class="vir-name" style="font-size:1.1rem">{APP_NAME}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="vir-side-label">Document</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded is not None and uploaded.name != st.session_state.doc_name:
        with st.spinner(f"Indexing '{uploaded.name}' — parsing, embedding locally…"):
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            try:
                index = build_index([tmp_path])
                n_chunks = index["n_chunks"]
                st.session_state.chain = build_chain(index)
                st.session_state.doc_name = uploaded.name
                # New document -> reset the conversation.
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success(f"Ready · {n_chunks} searchable chunks")
            except Exception as e:  # surface loader/index errors to the user
                st.error(f"Could not process file: {e}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.doc_name:
        st.markdown('<div class="vir-side-label">Active</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="vir-chip">◆ {st.session_state.doc_name}</span>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="vir-meta">model · {os.getenv("GEN_MODEL")}<br>'
        f'embeddings · local · temp 0</div>',
        unsafe_allow_html=True,
    )

# ---- Chat history display --------------------------------------------------
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

# ---- Empty state -----------------------------------------------------------
if st.session_state.chain is None and not st.session_state.messages:
    st.markdown(
        '<div class="vir-empty">Upload a <b>PDF</b> or <b>DOCX</b> in the sidebar, '
        'then ask anything.<br>Docullence answers <b>only</b> from that document.</div>',
        unsafe_allow_html=True,
    )

# ---- Chat input ------------------------------------------------------------
prompt = st.chat_input("Ask your document…")
if prompt:
    if st.session_state.chain is None:
        st.warning("Upload a PDF or DOCX first.")
        st.stop()

    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document…"):
            answer, sources = ask(
                st.session_state.chain, prompt, st.session_state.chat_history
            )
        st.markdown(answer)
        if sources:
            with st.expander(f"Sources · {len(sources)} retrieved chunks"):
                for i, doc in enumerate(sources, 1):
                    md = doc.metadata
                    page = md.get("page")
                    bits = []
                    if md.get("breadcrumb"):
                        bits.append(md["breadcrumb"])
                    if isinstance(page, int):
                        bits.append(f"page {page + 1}")
                    if md.get("kind") == "table":
                        bits.append("table")
                    tag = "  ·  ".join(bits) or f"chunk {i}"
                    st.markdown(f'<div class="vir-src-head">[{i}] {tag}</div>',
                                unsafe_allow_html=True)
                    st.text(doc.page_content[:600])

    # Persist turn for display + LLM history.
    st.session_state.messages.append(("assistant", answer))
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=answer))
