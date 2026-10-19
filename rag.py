"""RAG core: parse a document (PDF / DOCX / TXT / MD) into structure-aware
blocks, chunk them, embed locally, retrieve with a hybrid keyword+dense +
cross-encoder pipeline, and answer STRICTLY from the document.

Design goals (see RAG-Hands-On-Use-Cases brief):
  * Plain-text accuracy  - exact facts out of narrative prose (GPO report).
  * Length & structure    - tell "East Asia" from "Sub-Saharan Africa" in a
                            200-page report (World Bank GEP) via breadcrumbs.
  * Visual / scattered    - pull numbers out of tables & footnotes (iMac PER),
                            and READ CHARTS: figure questions render the page and
                            send it to the multimodal model (see ask()).

The anti-hallucination guardrail lives in GROUNDING_SYSTEM + temperature=0: the
model may only use retrieved context (and the provided page images) and must
refuse otherwise.

Embeddings run LOCALLY (sentence-transformers) so there is no API quota/rate
limit; Gemini is used only to write the answers.
"""
import base64
import os
import re

import pymupdf as fitz  # renders PDF pages to images for chart/figure vision
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

from parsing import load_document

load_dotenv()

GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = os.getenv("USE_RERANKER", "1").lower() not in ("0", "false", "no")

REFUSAL = "I don't have that information."

# ---- Guardrail system prompt: answer ONLY from context, else refuse --------
# Built as a plain string (not a ChatPromptTemplate) so we can append rendered
# figure images to the human turn for chart/visual questions (see ask()).
GROUNDING_SYSTEM = (
    "You are Docullence, a precise assistant that answers questions about ONE "
    "document the user uploaded. Each context block is prefixed with its section "
    "breadcrumb in square brackets, e.g. [CHAPTER 2 > East Asia and Pacific].\n"
    "Follow these rules with NO exceptions:\n"
    "1. Answer ONLY using the information in the <context> below (and any page "
    "images provided).\n"
    "2. Lead with the direct answer in the first sentence, stating any number or "
    "fact EXACTLY as written, including units. Then, when the context supports "
    "it, elaborate: give the supporting figures, the breakdown, relevant "
    "comparisons, and any caveats the document itself states. Use short "
    "paragraphs, a compact Markdown table, or bullets when that makes the answer "
    "clearer. Stay grounded in the context - never pad with generic filler. Do "
    "NOT repeat the bracketed [breadcrumb] tags from the context in your answer.\n"
    "3. The breadcrumbs are navigation hints for keeping facts straight: never "
    "mix figures from a DIFFERENT region or entity (e.g. keep East Asia "
    "separate from Sub-Saharan Africa).\n"
    "4. Phrases in the question like 'according to the executive summary', 'in "
    "chapter 3', or 'the report says' describe what the user is looking for - "
    "they are NOT a requirement that the context be labeled that way. If the "
    "requested fact appears anywhere in the context, answer it. Refuse ONLY "
    "when the fact itself is missing, never because a heading is worded "
    f"differently. When refusing, reply with EXACTLY: \"{REFUSAL}\"\n"
    "5. Never use outside knowledge, and never guess or infer beyond the "
    "context. If the context is empty and no image is provided, reply with the "
    "exact refusal sentence.\n"
    "6. Each user turn carries its OWN <context>. Answer the LATEST question from "
    "the context in that same turn - earlier turns were about other questions, so "
    "never refuse just because the topic changed from the previous turn."
)

# Appended when a cropped figure image accompanies the question.
VISION_ADDENDUM = (
    "\n\nYou are ALSO shown a cropped, high-resolution image of the figure/chart "
    "the question is about. Read the values DIRECTLY off it. Work in this order:\n"
    "1. LEGEND FIRST: read the legend and map each series name to its colour "
    "(e.g. 'Moderate risk = orange'). Then locate the EXACT series the question "
    "names - match its name to its colour. Do not confuse adjacent series/colours.\n"
    "2. For a STACKED bar, read the y-axis value at the BOTTOM edge and the TOP "
    "edge of that one coloured band against the gridlines; its value = top - "
    "bottom. Do this per bar; do NOT eyeball the whole coloured region as one "
    "number, and do not include neighbouring bands.\n"
    "3. Lead with the value the question asks for, then elaborate: give the other "
    "series/components in the same chart as a short breakdown (a compact table is "
    "ideal), note which figure/panel it is read from, and flag that chart-read "
    "numbers are approximate.\n"
    "4. For a change/difference question, report, in order: the value in each "
    "period; the ABSOLUTE change in percentage points; and the RELATIVE change "
    "in percent = (new - old) / old x 100.\n"
    "5. Values read off unlabelled bars are ESTIMATES - say they are approximate.\n"
    "6. Use ONLY the chart and context; refuse if the answer is genuinely not shown."
)

# Appended when a figure is shown as a BACKUP to full text context: the answer
# might be printed in the text OR appear only in the chart. Used for quantitative
# questions whose answer (a share/percentage) is typically only drawn, not typed.
VISION_ASSIST_ADDENDUM = (
    "\n\nYou are ALSO shown an image of a figure/chart from the document that is "
    "likely about this question. The exact value may be printed in the <context> "
    "text OR shown only in the chart. If the context states the number, use it "
    "verbatim. Otherwise READ it off the chart: identify the legend/series the "
    "question names, read the relevant bar/segment against the gridlines, and give "
    "the value - noting it is approximate because it was read from the chart. "
    "Prefer the figure over loosely related prose: if the context discusses a "
    "RELATED but different metric, do not substitute it for the value asked."
)

# Figure references in a question, e.g. "Figure 3.1", "fig 3", "exhibit 2.4".
FIG_LABEL_RE = re.compile(r"(?:figure|fig\.?|exhibit)\s*(\d+(?:\.\d+)?)", re.I)
# A specific panel within a figure, e.g. "chart D", "panel B", "CHART D".
PANEL_RE = re.compile(r"\b(?:chart|panel|plot|graph)\s+([A-Fa-f])\b", re.I)
# Words that signal the user is asking about something visual.
FIG_WORDS_RE = re.compile(
    r"\b(figure|fig\.?|chart|graph|plot|diagram|exhibit|panel|infographic|"
    r"bar\s*chart|axis|legend|curve)\b", re.I
)

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


# ---- Cached heavy models (load once per process, reuse across uploads) ------
_EMBEDDINGS = None
_CROSS_ENCODER = None


def get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _EMBEDDINGS


def get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
    return _CROSS_ENCODER


# ---- Chunking --------------------------------------------------------------
def _prefix_breadcrumb(doc):
    """Prepend the section breadcrumb so both the embedder and the LLM can see
    which chapter/section/region a chunk belongs to."""
    bc = doc.metadata.get("breadcrumb")
    page = doc.metadata.get("page")
    tag = bc or ""
    if isinstance(page, int) and doc.metadata.get("source", "").endswith(".pdf"):
        tag = (tag + f" (p.{page + 1})").strip()
    if tag:
        doc.page_content = f"[{tag}]\n{doc.page_content}"
    return doc


def _chunk(docs):
    """Split narrative text into overlapping chunks; keep tables whole (they are
    only split if very large, and then on row boundaries to preserve headers)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    out = []
    for d in docs:
        if d.metadata.get("kind") == "table":
            if len(d.page_content) <= 2200:
                out.append(_prefix_breadcrumb(d))
            else:
                out.extend(_prefix_breadcrumb(p) for p in _split_table(d))
        else:
            for piece in splitter.split_documents([d]):
                out.append(_prefix_breadcrumb(piece))
    return out


def _split_table(doc):
    """Split a very large Markdown table into row-batches, repeating the header
    row so every piece stays self-describing."""
    from langchain_core.documents import Document

    lines = doc.page_content.split("\n")
    if len(lines) < 3:
        return [doc]
    header, sep, rows = lines[0], lines[1], lines[2:]
    pieces, batch = [], []
    for row in rows:
        batch.append(row)
        if sum(len(x) for x in batch) > 1600:
            pieces.append("\n".join([header, sep] + batch))
            batch = []
    if batch:
        pieces.append("\n".join([header, sep] + batch))
    return [Document(page_content=p, metadata=dict(doc.metadata)) for p in pieces]


def _build_figure_map(pdf_bytes):
    """Map each figure label ("3.1", "2.4", ...) to the page index that holds the
    actual chart. A label appears several times (list-of-figures, in-text
    references, and the figure itself); the real figure page is the candidate
    with the most vector drawings, so we pick that one."""
    figure_map, figure_pages = {}, set()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        label_pages = {}
        for i in range(len(doc)):
            text = doc[i].get_text()
            for m in FIG_LABEL_RE.finditer(text):
                label_pages.setdefault(m.group(1), []).append(i)
        for label, pages in label_pages.items():
            best = pages[0] if len(pages) == 1 else max(
                pages, key=lambda p: len(doc[p].get_drawings())
            )
            figure_map[label] = best
            figure_pages.add(best)
    finally:
        doc.close()
    return figure_map, figure_pages


def build_index(paths):
    """Load -> parse -> chunk -> embed. Returns a dict with the vector store, the
    chunk list (reused for BM25), and - for a single PDF - the raw bytes plus a
    figure->page map so chart/figure questions can be answered with vision."""
    raw = []
    for p in paths:
        raw.extend(load_document(p))
    if not raw:
        raise ValueError("No text could be extracted from the uploaded file(s).")

    chunks = _chunk(raw)
    if not chunks:
        raise ValueError("No content to index after chunking.")

    vectorstore = InMemoryVectorStore.from_documents(chunks, get_embeddings())
    index = {"vectorstore": vectorstore, "chunks": chunks, "n_chunks": len(chunks),
             "pdf_bytes": None, "figure_map": {}, "figure_pages": set()}

    pdfs = [p for p in paths if p.lower().endswith(".pdf")]
    if len(pdfs) == 1:
        try:
            with open(pdfs[0], "rb") as f:
                pdf_bytes = f.read()
            index["pdf_bytes"] = pdf_bytes
            index["figure_map"], index["figure_pages"] = _build_figure_map(pdf_bytes)
        except Exception:
            # Chart vision is a bonus; never let it break indexing of the text.
            index["pdf_bytes"] = None
            index["figure_map"], index["figure_pages"] = {}, set()
    return index


# ---- Retrieval + answer chain ----------------------------------------------
def _build_retriever(index, k=6):
    """Hybrid retrieval: BM25 (exact keywords/numbers) + dense MMR (meaning),
    fused by an EnsembleRetriever, then re-ranked by a cross-encoder for
    precision. Candidate pool scales with document size."""
    vectorstore = index["vectorstore"]
    chunks = index["chunks"]
    n = len(chunks)
    cand = min(24, max(8, n // 40))  # per-retriever candidate pool

    dense = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": cand, "fetch_k": cand * 3, "lambda_mult": 0.5},
    )
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = cand

    ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])

    if USE_RERANKER:
        reranker = CrossEncoderReranker(model=get_cross_encoder(), top_n=k)
        return ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=ensemble
        )
    return ensemble


def build_chain(index, k=6):
    """Assemble the pieces ask() needs: a history-aware hybrid retriever, the LLM,
    and the index (for the figure map + PDF bytes used by chart vision)."""
    llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0)
    retriever = create_history_aware_retriever(
        llm, _build_retriever(index, k=k), CONTEXTUALIZE_PROMPT
    )
    return {"retriever": retriever, "llm": llm, "index": index}


# ---- Chart / figure vision -------------------------------------------------
def _target_figure_pages(question, docs, index):
    """Decide which PDF page image(s), if any, to show the model. Prefers an
    explicitly named figure ("Figure 3.1"); otherwise, if the question sounds
    visual, falls back to figure pages among the retrieved chunks."""
    if not index.get("pdf_bytes"):
        return []
    fmap, fpages = index.get("figure_map", {}), index.get("figure_pages", set())

    pages = [fmap[m.group(1)] for m in FIG_LABEL_RE.finditer(question)
             if m.group(1) in fmap]

    if not pages and FIG_WORDS_RE.search(question):
        pages = [d.metadata.get("page") for d in docs
                 if isinstance(d.metadata.get("page"), int)
                 and d.metadata.get("page") in fpages]
        if not pages:  # no known figure page retrieved -> try the top chunk's page
            for d in docs:
                if isinstance(d.metadata.get("page"), int):
                    pages = [d.metadata["page"]]
                    break

    seen = []
    for p in pages:
        if p not in seen:
            seen.append(p)
    return seen[:2]  # cap: at most two page images per question


# Short stoplist so question->figure-page matching keys on content words only.
_STOP = {
    "what", "which", "whats", "does", "did", "about", "report", "according",
    "there", "their", "theirs", "from", "with", "into", "have", "has", "that",
    "this", "will", "would", "could", "should", "these", "those", "share",
    "much", "many", "percent", "percentage", "value", "values", "number",
    "shown", "show", "the", "for", "and", "are", "was", "were", "how",
}


def _question_keywords(question):
    """Content words from the question, used to score which figure page a data
    question is really about (e.g. 'energy exporters commodity nontax revenue')."""
    toks = re.findall(r"[a-z]{4,}", question.lower())
    return {t for t in toks if t not in _STOP}


# A question that asks for a quantity - the kind whose answer is often ONLY in a
# chart (a share, percentage, rate, amount) rather than printed in the prose.
_QUANT_RE = re.compile(
    r"(%|\bshares?\b|\bpercent|\bproportion|\bhow much\b|\bhow many\b|\brates?\b|"
    r"\bratio\b|\brevenues?\b|\bgdp\b|\baverage\b|\bgrowth\b|\bper\s*cent)", re.I)


def _scored_figure_pages(question, docs, index):
    """Rank figure pages by how many distinct question keywords appear in the
    page's text (its caption/axis labels/legend). Returns [(matched, page, label)]
    sorted best-first, where `matched` is the count of distinct keywords hit."""
    pdf_bytes = index.get("pdf_bytes")
    fpages = index.get("figure_pages", set())
    kw = _question_keywords(question)
    if not pdf_bytes or not fpages or not kw:
        return []

    page_to_label = {}
    for label, pg in index.get("figure_map", {}).items():
        page_to_label.setdefault(pg, label)  # one representative label per page
    retrieved = {d.metadata.get("page") for d in docs
                 if isinstance(d.metadata.get("page"), int)}

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        scored = []
        for p in fpages:
            if not (0 <= p < len(doc)):
                continue
            text = doc[p].get_text().lower()
            matched = sum(1 for w in kw if w in text)
            if matched:
                # Tie-break toward pages the retriever already surfaced.
                rank = matched + (0.5 if p in retrieved else 0)
                scored.append((rank, matched, p, page_to_label.get(p)))
    finally:
        doc.close()

    scored.sort(reverse=True)
    return [(matched, p, label) for _r, matched, p, label in scored]


def _fallback_figure_pages(question, docs, index):
    """Last resort when a text answer refused: the fact may live in a chart the
    question never named. Returns up to two [(page, label)] best-matching pages."""
    return [(p, label) for _m, p, label in _scored_figure_pages(question, docs, index)[:2]]


def _assist_figure_pages(question, docs, index):
    """Proactive chart-assist: for a QUANTITATIVE question with no figure keyword,
    if a figure page strongly matches the question's keywords the answer is likely
    read off that chart (shares/percentages are rarely printed in prose). Returns
    [(page, label)] only on a strong, specific match to avoid false positives."""
    if not _QUANT_RE.search(question):
        return []
    kw = _question_keywords(question)
    scored = _scored_figure_pages(question, docs, index)
    if not scored:
        return []
    matched, page, label = scored[0]
    strong = matched >= 3 and matched >= 0.5 * len(kw)
    return [(page, label)] if strong else []


def _crop_png(page, rect, zoom):
    pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(zoom, zoom))
    return base64.b64encode(pix.tobytes("png")).decode()


def _best_panel(page, question):
    """In a multi-panel figure, pick the panel the question is really about by
    matching its caption/title text (each starts with 'A.', 'B.', ...) against the
    question keywords. Returns the letter only on a clear, specific win."""
    kw = _question_keywords(question)
    if not kw:
        return None
    buckets = {}
    for b in page.get_text("dict").get("blocks", []):
        for ln in b.get("lines", []):
            txt = "".join(s["text"] for s in ln.get("spans", [])).strip()
            m = re.match(r"^([A-Fa-f])[\.\)]\s", txt)
            if m:
                buckets.setdefault(m.group(1).upper(), []).append(txt.lower())
    scored = sorted(
        ((sum(1 for w in kw if w in " ".join(v)), L) for L, v in buckets.items()),
        reverse=True,
    )
    if not scored:
        return None
    top = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0
    return top[1] if top[0] >= 2 and (top[0] >= 3 or top[0] > second) else None


def _panel_rect(page, letter):
    """Bounding box of a single panel (e.g. "D. Risk of debt distress...") so we
    can crop and zoom into just that chart - a full-page render is too small for
    the model to resolve stacked-bar segment boundaries, which causes misreads.

    A page has several spans starting with the letter (the panel TITLE and the
    small footnote "D. Sample covers..."). The panel title is set in a larger
    font, so pick the largest-font match and anchor the crop on it."""
    letter = letter.upper()
    pat = re.compile(rf"^{letter}[\.\)]\s")
    best = None  # (font_size, Rect)
    for b in page.get_text("dict").get("blocks", []):
        for ln in b.get("lines", []):
            for s in ln.get("spans", []):
                if pat.match(s["text"].strip() + " "):
                    size = s.get("size", 0)
                    if best is None or size > best[0]:
                        best = (size, fitz.Rect(s["bbox"]))
    if best is None:
        return None
    r = best[1]
    W, H = page.rect.width, page.rect.height
    # Keep the box tight to the panel's own chart: a wider box pulls in the
    # neighbouring text column / footnotes, which distracts the model and caused
    # mis-reads. These proportions isolate one chart in a multi-panel figure.
    return fitz.Rect(
        max(page.rect.x0, r.x0 - 16),
        max(page.rect.y0, r.y0 - 12),
        min(page.rect.x1 - 2, r.x0 + W * 0.27),
        min(page.rect.y1 - 2, r.y0 + H * 0.19),
    )


def _figure_rect(page, label):
    """Bounding box of a whole figure, anchored on its "Figure X.Y" caption."""
    pat = re.compile(rf"(?i)figure\s*{re.escape(label)}\b")
    for b in page.get_text("dict").get("blocks", []):
        for ln in b.get("lines", []):
            txt = "".join(s["text"] for s in ln.get("spans", []))
            if pat.search(txt):
                r = fitz.Rect(ln["bbox"])
                H = page.rect.height
                return fitz.Rect(page.rect.x0 + 4, max(page.rect.y0, r.y0 - 6),
                                 page.rect.x1 - 4, min(page.rect.y1, r.y0 + H * 0.62))
    return None


def _figure_images(pdf_bytes, pages, question, force_label=None, auto_panel=False):
    """Render the figure for a visual question. Prefers a tight, high-zoom crop of
    the named panel/figure (accurate); falls back to a higher-DPI full page.
    force_label crops to that figure when the question named none; auto_panel lets
    us detect the right panel of a multi-panel figure from the question keywords."""
    if not pages:
        return []
    pm = PANEL_RE.search(question)
    letter = pm.group(1) if pm else None
    labels = FIG_LABEL_RE.findall(question)
    label = labels[0] if labels else force_label

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        p = pages[0]
        if not (0 <= p < len(doc)):
            return []
        page = doc[p]
        if letter is None and auto_panel:
            letter = _best_panel(page, question)
        rect = (_panel_rect(page, letter) if letter else None)
        if rect is None and label:
            rect = _figure_rect(page, label)
        if rect is not None:
            return [_crop_png(page, rect, 5.0 if letter else 3.0)]
        # No locatable crop -> full page at a higher DPI than the old 160.
        return [_crop_png(page, None, 230 / 72)]
    finally:
        doc.close()


def _generate(llm, context, chat_history, question, images, assist=False):
    # Keep the context in the CURRENT user turn (not the system message) so that,
    # across a multi-turn chat, the model always binds the context to the latest
    # question - otherwise prior turns sit between a system-held context and the
    # new question and the model wrongly refuses on a topic change.
    addendum = (VISION_ASSIST_ADDENDUM if assist else VISION_ADDENDUM) if images else ""
    system = GROUNDING_SYSTEM + addendum
    turn = (f"<context>\n{context or '(no text retrieved)'}\n</context>\n\n"
            f"Question: {question}")
    messages = [SystemMessage(content=system), *chat_history]
    if images:
        content = [{"type": "text", "text": turn}]
        content += [{"type": "image_url",
                     "image_url": f"data:image/png;base64,{b}"} for b in images]
        messages.append(HumanMessage(content=content))
    else:
        messages.append(HumanMessage(content=turn))
    return llm.invoke(messages).content


def _is_refusal(answer):
    return (answer or "").strip().lower().startswith(REFUSAL.lower()[:20])


def ask(chain, question, chat_history):
    """Run one turn. Retrieves text context, and for chart/figure questions also
    renders the relevant page(s) so the model can read the visual. Returns
    (answer_text, source_documents)."""
    retriever, llm, index = chain["retriever"], chain["llm"], chain["index"]
    docs = retriever.invoke({"input": question, "chat_history": chat_history})

    # Mode 1 - figure-primary: the question names a figure or is explicitly visual,
    # so the CHART is the source of truth. Full text context alongside the image
    # biases the chart read, so send only a little context then.
    pages = _target_figure_pages(question, docs, index)
    images = _figure_images(index["pdf_bytes"], pages, question) if pages else []
    assist = False

    # Mode 2 - figure-assist: a quantitative question (a share/percentage) whose
    # answer is typically only drawn, not typed. Show the best-matching chart as a
    # BACKUP next to the FULL text, so the model reads the exact value off it.
    if not images and index.get("pdf_bytes"):
        ap = _assist_figure_pages(question, docs, index)
        if ap:
            pages = [p for p, _l in ap]
            images = _figure_images(index["pdf_bytes"], pages, question,
                                    force_label=ap[0][1], auto_panel=True)
            assist = bool(images)

    context = (_format_context(docs[:2], max_chars=500) if (images and not assist)
               else _format_context(docs))
    answer = _generate(llm, context, chat_history, question, images, assist=assist)

    # Mode 3 - refusal fallback: still refused with no image tried, yet the fact
    # may live in a chart. Find the best-matching figure page and re-ask with it.
    if not images and _is_refusal(answer) and index.get("pdf_bytes"):
        fb = _fallback_figure_pages(question, docs, index)
        if fb:
            pages = [p for p, _l in fb]
            images = _figure_images(index["pdf_bytes"], pages, question,
                                    force_label=fb[0][1], auto_panel=True)
            if images:
                answer = _generate(llm, _format_context(docs), chat_history,
                                   question, images, assist=True)

    sources = list(docs)
    for p in pages:  # show the user which page image was consulted
        sources.append(Document(
            page_content="(chart/figure read directly from the rendered page image)",
            metadata={"breadcrumb": "Figure page", "page": p, "kind": "figure"},
        ))
    return answer, sources


def _format_context(docs, max_chars=None):
    if not docs:
        return ""
    parts = [d.page_content for d in docs]
    if max_chars:
        parts = [p[:max_chars] for p in parts]
    return "\n\n".join(parts)
