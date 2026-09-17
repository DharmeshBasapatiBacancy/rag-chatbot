"""RAG core: load a document (PDF / DOCX / TXT / MD), split, embed, store,
retrieve, and answer STRICTLY from the document.

The anti-hallucination guardrail lives in ANSWER_PROMPT + temperature=0:
the model may only use retrieved context and must refuse otherwise.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

REFUSAL = "I don't have that information."

# ---- Guardrail prompt: answer ONLY from context, else refuse ---------------
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions about a specific "
     "document the user uploaded.\n"
     "Follow these rules with NO exceptions:\n"
     "1. Answer ONLY using the information in the <context> below.\n"
     "2. If the answer is not clearly contained in the context, reply with "
     f"EXACTLY this sentence and nothing else: \"{REFUSAL}\"\n"
     "3. Never use outside knowledge, and never guess or infer beyond the "
     "context.\n"
     "4. If the context is empty, reply with the exact refusal sentence.\n\n"
     "<context>\n{context}\n</context>"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ---- Prompt that rewrites a follow-up into a standalone question ------------
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, rephrase the "
     "question into a standalone question that can be understood without the "
     "chat history. Do NOT answer it, just reformulate it if needed and "
     "otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def _load_file(path: str):
    """Load one file into LangChain Documents based on its extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".docx":
        loader = Docx2txtLoader(path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext} (use PDF, DOCX, TXT, MD)")
    return loader.load()


def build_vectorstore(paths):
    """Load -> split -> embed -> store. Returns (vectorstore, n_chunks).

    Uses a fresh in-memory vector store so each upload starts clean and the
    bot only ever answers from the currently uploaded document(s).
    """
    docs = []
    for p in paths:
        docs.extend(_load_file(p))

    if not docs:
        raise ValueError("No text could be extracted from the uploaded file(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model=f"models/{EMBED_MODEL}")
    vectorstore = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    return vectorstore, len(chunks)


def build_chain(vectorstore, k=4):
    """Build the history-aware retrieval + strictly-grounded answer chain."""
    llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0)
    # Don't request more chunks than exist (small documents).
    try:
        available = len(vectorstore.store)
        k = max(1, min(k, available))
    except Exception:
        pass
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZE_PROMPT
    )
    answer_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    return create_retrieval_chain(history_aware_retriever, answer_chain)


def ask(chain, question, chat_history):
    """Run one turn. Returns (answer_text, source_documents)."""
    result = chain.invoke({"input": question, "chat_history": chat_history})
    return result["answer"], result.get("context", [])
