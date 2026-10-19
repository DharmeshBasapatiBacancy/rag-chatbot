"""Structure-aware document parsing: body text + tables + heading breadcrumbs.

Why this exists
---------------
A naive PDF-to-text pass (e.g. PyPDFLoader) throws away two things that matter
for real-world RAG:

1. **Tables** - power-consumption tables, carbon breakdowns, statistics. Flattened
   to a stream of numbers, a figure loses its row/column header and becomes
   un-retrievable ("346" means nothing without "iMac emissions / manufacturing").
2. **Section structure** - in a 200-page report, a chunk about "East Asia" and a
   chunk about "Sub-Saharan Africa" look almost identical to an embedder unless
   each one carries its chapter/section label.

This module extracts both. Every returned Document carries ``page``, ``chapter``,
``section`` and a ``breadcrumb`` in its metadata; tables are rendered to Markdown
so numbers stay bound to their headers; ``kind`` is ``"text"`` or ``"table"``.
"""
import re
from collections import Counter
from pathlib import Path
from statistics import median

import pdfplumber
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------
def _clean_cell(v):
    if v is None:
        return ""
    return " ".join(str(v).split())


def _has_alpha(s):
    return any(c.isalpha() for c in s)


def table_to_markdown(rows):
    """Render a list-of-rows table to GitHub-flavoured Markdown, or "" if the
    grid is too sparse / unlabeled to be trustworthy.

    pdfplumber happily returns "tables" that are really chart decorations or
    number grids whose label column got split off. Those add noise (orphaned
    numbers) rather than signal, so we drop any grid that is mostly empty or has
    no textual labels - the surrounding page text still carries those facts.
    """
    rows = [[_clean_cell(c) for c in r] for r in rows if r]
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    if width < 2:
        return ""
    rows = [r + [""] * (width - len(r)) for r in rows]

    cells = [c for r in rows for c in r]
    fill = sum(bool(c) for c in cells) / len(cells)
    has_labels = any(_has_alpha(c) for c in rows[0]) or any(
        _has_alpha(r[0]) for r in rows
    )
    if fill < 0.5 or not has_labels:
        return ""

    header, body = rows[0], rows[1:]
    header = [h or f"col{i+1}" for i, h in enumerate(header)]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF heading detection
# ---------------------------------------------------------------------------
_BOLD_HINTS = ("bold", "black", "semibold", "heavy", "medium")
_CHAPTER_RE = re.compile(r"^(chapter|part|section|appendix)\b", re.IGNORECASE)


def _group_lines(words):
    """Group words (already belonging to one reading column) into visual lines."""
    words = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    lines, cur, cur_top = [], [], None
    for w in words:
        top = round(w["top"])
        if cur_top is None or abs(top - cur_top) <= 3:
            cur.append(w)
            cur_top = top if cur_top is None else cur_top
        else:
            lines.append(cur)
            cur, cur_top = [w], top
    if cur:
        lines.append(cur)

    out = []
    for ln in lines:
        text = " ".join(w["text"] for w in ln).strip()
        if not text:
            continue
        size = median([round(float(w.get("size", 0)), 1) for w in ln])
        fonts = " ".join(str(w.get("fontname", "")) for w in ln).lower()
        is_bold = any(h in fonts for h in _BOLD_HINTS)
        out.append((text, size, is_bold, ln[0]["top"]))
    return out


def _page_lines(page):
    """Return [(text, size, is_bold, top)] in correct reading order, handling
    two-column layouts (common in reports like the World Bank GEP). Reading a
    two-column page straight across interleaves the columns and scrambles every
    sentence, so we detect the split and read each column top-to-bottom."""
    # NB: no use_text_flow - it merges words across the column gap into wide
    # tokens, which hides the gutter we rely on to detect columns.
    words = page.extract_words(extra_attrs=["size", "fontname"])
    if not words:
        return []

    split = _column_split(page, words)
    if split is not None:
        left = [w for w in words if (w["x0"] + w["x1"]) / 2 < split]
        right = [w for w in words if (w["x0"] + w["x1"]) / 2 >= split]
        return _group_lines(left) + _group_lines(right)
    return _group_lines(words)


def _column_split(page, words):
    """Return the x of a clean two-column gutter, or None if the page is a single
    column. A true gutter is a vertical line in the page's central third that
    almost no word crosses, with substantial text on both sides."""
    n = len(words)
    if n < 40:
        return None
    width = float(page.width)
    x0 = float(page.bbox[0])
    best_x, best_cross = None, None
    for i in range(40, 61):
        x = x0 + width * i / 100.0
        cross = sum(1 for w in words if w["x0"] < x - 1 and w["x1"] > x + 1)
        if best_cross is None or cross < best_cross:
            best_x, best_cross = x, cross
    left = sum(1 for w in words if (w["x0"] + w["x1"]) / 2 < best_x)
    right = n - left
    if best_cross <= 0.02 * n and min(left, right) >= 0.25 * n:
        return best_x
    return None


def _looks_titleish(text):
    """A heading starts with a capital/number and is not a sentence fragment
    (no mid-line period, which is the tell-tale of captured body prose)."""
    if not (text[:1].isupper() or text[:1].isdigit()):
        return False
    core = text[:-1] if text[-1:] in ".:" else text
    return ". " not in core


def _is_heading(text, size, is_bold, body_size):
    """A line is a heading if it is short, title-like, and visually emphasised -
    clearly bigger than body text, or bold and at least slightly bigger. Region
    and sub-section titles (e.g. "East Asia and Pacific") are set larger/bold;
    inline bold lead-ins inside paragraphs are rejected by the title check."""
    words = text.split()
    if not (1 <= len(words) <= 14):
        return False
    if text.endswith((",", ";", ":")):
        return False
    letters = sum(c.isalpha() for c in text)
    if letters < max(2, len(text) // 2):
        return False
    if not _looks_titleish(text):
        return False
    bigger = size >= body_size * 1.15
    bold_heading = is_bold and size >= body_size * 1.08 and len(words) <= 8
    return bigger or bold_heading


def _parse_pdf(path):
    docs = []
    with pdfplumber.open(path) as pdf:
        # --- Pass 1: learn the body font size + repeated header/footer lines ---
        size_counter = Counter()
        line_freq = Counter()
        pages_lines = [_page_lines(page) for page in pdf.pages]
        for lines in pages_lines:
            for text, size, _bold, _top in lines:
                size_counter[size] += len(text)
                line_freq[text] += 1
        body_size = size_counter.most_common(1)[0][0] if size_counter else 10.0
        n_pages = len(pdf.pages)
        # Lines that repeat on many pages are running headers/footers, not content.
        boilerplate = {
            t for t, c in line_freq.items()
            if c >= max(4, n_pages * 0.5) and len(t) < 120
        }
        # The largest heading font size marks chapter-level titles.
        htier = Counter()
        for lines in pages_lines:
            for text, size, bold, _top in lines:
                if text not in boilerplate and _is_heading(text, size, bold, body_size):
                    htier[size] += 1
        top_size = max(htier) if htier else body_size * 2
        chapter_cut = top_size * 0.98  # this size or larger == chapter

        # --- Pass 2: build documents, tracking chapter/section state -----------
        chapter, section = None, None
        for pno, (page, lines) in enumerate(zip(pdf.pages, pages_lines)):
            start_chapter, start_section = chapter, section
            kept_lines = []
            for text, size, bold, _top in lines:
                if text in boilerplate:
                    continue
                if _is_heading(text, size, bold, body_size):
                    if _CHAPTER_RE.match(text) or size >= chapter_cut:
                        chapter, section = text, None
                    else:
                        section = text
                kept_lines.append(text)

            body = "\n".join(kept_lines).strip()
            meta = _meta(path, pno, start_chapter, start_section)
            if body:
                docs.append(Document(page_content=body, metadata={**meta, "kind": "text"}))

            # Tables on this page -> their own Documents (kept whole).
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for tno, tbl in enumerate(tables):
                md = table_to_markdown(tbl)
                if md:
                    docs.append(Document(
                        page_content=md,
                        metadata={**meta, "kind": "table", "table_index": tno},
                    ))
    return docs


# ---------------------------------------------------------------------------
# DOCX + plain text
# ---------------------------------------------------------------------------
def _parse_docx(path):
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(path)
    docs = []
    chapter, section = None, None
    buf = []

    def flush():
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata=_meta(path, 0, chapter, section, kind="text"),
            ))
        buf = []

    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            para = Paragraph(child, d)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "").lower() if para.style else ""
            if style.startswith("title") or style.startswith("heading 1"):
                flush()
                chapter, section = text, None
            elif style.startswith("heading"):
                flush()
                section = text
            else:
                buf.append(text)
        elif tag == "tbl":
            flush()
            tbl = Table(child, d)
            rows = [[c.text for c in r.cells] for r in tbl.rows]
            md = table_to_markdown(rows)
            if md:
                docs.append(Document(
                    page_content=md,
                    metadata=_meta(path, 0, chapter, section, kind="table"),
                ))
    flush()
    return docs


def _parse_text(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata=_meta(path, 0, None, None, kind="text"))]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _meta(path, page, chapter, section, kind="text"):
    bc = " › ".join(x for x in (chapter, section) if x)
    return {
        "source": Path(path).name,
        "page": page,
        "chapter": chapter,
        "section": section,
        "breadcrumb": bc,
        "kind": kind,
    }


def load_document(path):
    """Parse one file into structure-aware Documents (text + table blocks)."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext == ".docx":
        return _parse_docx(path)
    if ext in (".txt", ".md"):
        return _parse_text(path)
    raise ValueError(f"Unsupported file type: {ext} (use PDF, DOCX, TXT, MD)")
