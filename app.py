"""Streamlit chat UI for the document-grounded RAG chatbot.

Upload a PDF or DOCX, then ask questions. The bot answers ONLY from the
uploaded document and says "I don't have that information." otherwise.

Run:  streamlit run app.py
"""
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

# Config loading works in BOTH environments:
#  - Local dev: reads .env (git-ignored).
#  - Streamlit Community Cloud: reads secrets set in the app dashboard.
# We copy secrets into os.environ so langchain-google-genai and rag.py pick
# them up identically in both places. This must run BEFORE importing rag.
load_dotenv()
try:
    for _k in ("GOOGLE_API_KEY", "GEN_MODEL", "EMBED_MODEL"):
        if _k not in os.environ and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets configured (e.g. pure local run) -> rely on .env

from langchain_core.messages import AIMessage, HumanMessage

from rag import build_vectorstore, build_chain, ask

if not os.getenv("GOOGLE_API_KEY"):
    st.error(
        "GOOGLE_API_KEY is not set. Locally: add it to .env. "
        "On Streamlit Cloud: add it under App settings -> Secrets."
    )
    st.stop()

st.set_page_config(page_title="Document RAG Chatbot", page_icon="📄")
st.title("📄 Document RAG Chatbot")
st.caption("Upload a PDF or DOCX. I answer **only** from that document.")

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
    st.header("1. Upload a document")
    uploaded = st.file_uploader(
        "PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=False
    )

    if uploaded is not None and uploaded.name != st.session_state.doc_name:
        with st.spinner(f"Reading and indexing '{uploaded.name}'..."):
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            try:
                vectorstore, n_chunks = build_vectorstore([tmp_path])
                st.session_state.chain = build_chain(vectorstore)
                st.session_state.doc_name = uploaded.name
                # New document -> reset the conversation.
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success(f"Indexed '{uploaded.name}' into {n_chunks} chunks.")
            except Exception as e:  # surface loader/index errors to the user
                st.error(f"Could not process file: {e}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.doc_name:
        st.info(f"Active document: **{st.session_state.doc_name}**")
    st.divider()
    st.caption(f"Model: {os.getenv('GEN_MODEL')}  ·  temp=0")

# ---- Chat history display --------------------------------------------------
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

# ---- Chat input ------------------------------------------------------------
prompt = st.chat_input("Ask a question about the document...")
if prompt:
    if st.session_state.chain is None:
        st.warning("Please upload a PDF or DOCX first.")
        st.stop()

    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask(
                st.session_state.chain, prompt, st.session_state.chat_history
            )
        st.markdown(answer)
        if sources:
            with st.expander("Sources used (retrieved chunks)"):
                for i, doc in enumerate(sources, 1):
                    page = doc.metadata.get("page")
                    tag = f" (page {page + 1})" if isinstance(page, int) else ""
                    st.markdown(f"**Chunk {i}{tag}**")
                    st.text(doc.page_content[:500])

    # Persist turn for display + LLM history.
    st.session_state.messages.append(("assistant", answer))
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=answer))
