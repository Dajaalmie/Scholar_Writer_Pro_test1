import io
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from openai import OpenAI
import pytesseract
from pdf2image import convert_from_bytes

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Scholar Writer Pro",
    page_icon="📚",
    layout="wide",
)

# Change this if your Tesseract path is different
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .hero-box {
        padding: 1.2rem 1.4rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(37,99,235,0.16), rgba(168,85,247,0.14));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }

    .section-card {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.8rem;
    }

    .metric-box {
        padding: 0.75rem 0.9rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
    }

    .big-title {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .sub-title {
        color: #cbd5e1;
        font-size: 0.98rem;
    }

    .muted {
        color: #94a3b8;
        font-size: 0.92rem;
    }

    .small-box {
        padding: 0.7rem 0.9rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================
if "papers_raw" not in st.session_state:
    st.session_state.papers_raw = []

if "papers_display" not in st.session_state:
    st.session_state.papers_display = []

if "pdf_docs" not in st.session_state:
    st.session_state.pdf_docs = []

if "last_generated_text" not in st.session_state:
    st.session_state.last_generated_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome. Search papers in Scholar Search, add PDFs or article links in PDF Library, "
                "then generate notes in Write-up Bot."
            ),
        }
    ]


# ============================================================
# HELPERS
# ============================================================
def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def extract_year(summary: str) -> Optional[int]:
    if not summary:
        return None
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", summary)
    return int(years[-1]) if years else None


def find_pdf_link(item: Dict[str, Any]) -> str:
    resources = item.get("resources", [])
    for r in resources:
        if isinstance(r, dict) and r.get("link"):
            return r["link"]
    return ""


# ============================================================
# QUERY SCOPE LOGIC
# ============================================================
def build_scope_queries(base_query: str, scope: str) -> List[str]:
    q = base_query.strip()

    if scope == "Nigeria":
        return [
            q,
            f"{q} Nigeria",
            f"{q} Nigerian",
            f"{q} \"Kwara State\"",
            f"{q} Lagos",
            f"{q} Ilorin",
        ]

    if scope == "African":
        return [
            q,
            f"{q} Africa",
            f"{q} African",
            f"{q} Ghana",
            f"{q} Kenya",
            f"{q} Nigeria",
            f"{q} \"South Africa\"",
        ]

    return [q]


# ============================================================
# SCHOLAR SEARCH
# ============================================================
def scholar_search_page(
    api_key: str,
    query: str,
    start_year: int,
    end_year: int,
    num_results: int = 20,
    start_offset: int = 0,
) -> List[Dict[str, Any]]:
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": api_key,
        "as_ylo": start_year,
        "as_yhi": end_year,
        "num": min(num_results, 20),
        "start": start_offset,
        "hl": "en",
        "as_vis": "0",
        "filter": "1",
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    organic = data.get("organic_results", [])
    papers = []

    for item in organic:
        publication_info = item.get("publication_info", {})
        summary = publication_info.get("summary", "")

        authors = []
        if "authors" in publication_info:
            for author in publication_info["authors"]:
                name = author.get("name")
                if name:
                    authors.append(name)

        papers.append(
            {
                "title": item.get("title", "No title"),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "summary": summary,
                "authors": authors,
                "year": extract_year(summary),
                "cited_by_count": safe_get(item, "inline_links", "cited_by", "total", default=0),
                "cited_by_link": safe_get(item, "inline_links", "cited_by", "link", default=""),
                "pdf_link": find_pdf_link(item),
                "result_id": item.get("result_id", ""),
            }
        )

    return papers


def scholar_search_multi(
    api_key: str,
    query: str,
    start_year: int,
    end_year: int,
    total_results: int,
) -> List[Dict[str, Any]]:
    results = []
    remaining = total_results
    offset = 0

    while remaining > 0:
        batch = min(remaining, 20)
        page = scholar_search_page(
            api_key=api_key,
            query=query,
            start_year=start_year,
            end_year=end_year,
            num_results=batch,
            start_offset=offset,
        )
        if not page:
            break
        results.extend(page)
        offset += batch
        remaining -= batch

    return results


def scholar_search_multi_query(
    api_key: str,
    queries: List[str],
    start_year: int,
    end_year: int,
    total_results_per_query: int,
) -> List[Dict[str, Any]]:
    all_results = []

    for q in queries:
        try:
            results = scholar_search_multi(
                api_key=api_key,
                query=q,
                start_year=start_year,
                end_year=end_year,
                total_results=total_results_per_query,
            )
            all_results.extend(results)
        except Exception:
            continue

    seen = set()
    deduped = []
    for p in all_results:
        key = (
            p.get("title", "").strip().lower(),
            p.get("year"),
            p.get("link", "").strip().lower(),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return deduped


def rank_papers(papers: List[Dict[str, Any]], display_limit: int) -> List[Dict[str, Any]]:
    ranked = []

    for p in papers:
        score = 0

        cited = int(p.get("cited_by_count", 0) or 0)
        score += min(cited // 20, 10)

        year = p.get("year")
        if isinstance(year, int):
            if year >= 2020:
                score += 3
            elif year >= 2015:
                score += 2
            elif year >= 2010:
                score += 1

        enriched = dict(p)
        enriched["score"] = score
        ranked.append(enriched)

    ranked.sort(
        key=lambda x: (
            x.get("score", 0),
            x.get("cited_by_count", 0),
            x.get("year") or 0,
        ),
        reverse=True,
    )

    return ranked[:display_limit]


# ============================================================
# PDF / URL INGESTION
# ============================================================
def extract_text_from_pdf_bytes_smart(pdf_bytes: bytes) -> Tuple[str, int]:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            texts.append(txt)

        full_text = clean_text("\n\n".join(texts))
        page_count = len(reader.pages)

        if len(full_text.strip()) >= 500:
            return full_text, page_count

    except Exception:
        pass

    images = convert_from_bytes(pdf_bytes)
    ocr_texts = []

    for img in images:
        text = pytesseract.image_to_string(img)
        ocr_texts.append(text)

    full_ocr_text = clean_text("\n\n".join(ocr_texts))
    return full_ocr_text, len(images)


def extract_text_from_uploaded_pdf(uploaded_file) -> Tuple[str, int]:
    raw = uploaded_file.read()
    return extract_text_from_pdf_bytes_smart(raw)


def fetch_pdf_bytes_from_url(url: str, timeout: int = 60) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()
    if "pdf" not in content_type and not url.lower().endswith(".pdf"):
        if not resp.content.startswith(b"%PDF"):
            raise ValueError("The URL did not return a PDF file.")
    return resp.content


def add_pdf_doc(source_name: str, text: str, page_count: int, source_url: str = ""):
    if not text.strip():
        return
    st.session_state.pdf_docs.append(
        {
            "source_name": source_name,
            "source_url": source_url,
            "page_count": page_count,
            "text": text,
            "word_count": word_count(text),
        }
    )


def add_web_doc(source_name: str, text: str, source_url: str = ""):
    if not text.strip():
        return

    st.session_state.pdf_docs.append(
        {
            "source_name": source_name,
            "source_url": source_url,
            "page_count": 1,
            "text": text,
            "word_count": word_count(text),
        }
    )


def is_pdf_url(url: str) -> bool:
    return url.lower().endswith(".pdf")


def fetch_url_content(url: str, timeout: int = 60) -> Tuple[bytes, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").lower()
    return resp.content, content_type


def extract_text_from_html(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.extract()

    text_parts = []

    article = soup.find("article")
    if article:
        text = article.get_text("\n", strip=True)
        if text:
            text_parts.append(text)

    main = soup.find("main")
    if main:
        text = main.get_text("\n", strip=True)
        if text:
            text_parts.append(text)

    body = soup.find("body")
    if body:
        text = body.get_text("\n", strip=True)
        if text:
            text_parts.append(text)

    full_text = "\n\n".join(text_parts)
    full_text = clean_text(full_text)

    lines = []
    seen = set()
    for line in full_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line not in seen:
            seen.add(line)
            lines.append(line)

    return clean_text("\n".join(lines))


def find_pdf_link_in_html(page_url: str, html_bytes: bytes) -> Optional[str]:
    soup = BeautifulSoup(html_bytes, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full_url = urljoin(page_url, href)

        if ".pdf" in href.lower() or ".pdf" in full_url.lower():
            return full_url

        link_text = a.get_text(" ", strip=True).lower()
        if any(term in link_text for term in ["pdf", "download", "full text", "fulltext"]):
            return full_url

    for meta in soup.find_all("meta"):
        content = meta.get("content", "")
        if content and ".pdf" in content.lower():
            return urljoin(page_url, content)

    return None


def ingest_any_url(url: str) -> Tuple[str, str, int]:
    raw, content_type = fetch_url_content(url)

    if "pdf" in content_type or is_pdf_url(url) or raw.startswith(b"%PDF"):
        text, page_count = extract_text_from_pdf_bytes_smart(raw)
        source_name = url.split("/")[-1] or "remote_pdf.pdf"
        return source_name, text, page_count

    html_text = extract_text_from_html(raw)

    if len(html_text.strip()) < 800:
        pdf_link = find_pdf_link_in_html(url, raw)
        if pdf_link:
            try:
                pdf_bytes, _ = fetch_url_content(pdf_link)
                text, page_count = extract_text_from_pdf_bytes_smart(pdf_bytes)
                source_name = pdf_link.split("/")[-1] or "linked_pdf.pdf"
                return source_name, text, page_count
            except Exception:
                pass

    soup = BeautifulSoup(raw, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url

    return title, html_text, 1


# ============================================================
# PDF CHUNKING
# ============================================================
def chunk_text(text: str, chunk_words: int = 350, overlap_words: int = 60) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_words]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks


def build_pdf_chunk_index(pdf_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_chunks = []
    for doc in pdf_docs:
        chunks = chunk_text(doc["text"])
        for i, ch in enumerate(chunks, start=1):
            all_chunks.append(
                {
                    "source_name": doc["source_name"],
                    "source_url": doc.get("source_url", ""),
                    "chunk_id": i,
                    "text": ch,
                }
            )
    return all_chunks


def score_chunk(query: str, chunk_text_value: str) -> int:
    q_terms = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", query.lower())
    ctext = chunk_text_value.lower()
    score = 0
    for term in q_terms:
        score += ctext.count(term)
    return score


def retrieve_relevant_chunks(query: str, pdf_docs: List[Dict[str, Any]], top_k: int = 12) -> List[Dict[str, Any]]:
    chunks = build_pdf_chunk_index(pdf_docs)
    if not chunks:
        return []

    scored = []
    for ch in chunks:
        s = score_chunk(query, ch["text"])
        if s > 0:
            scored.append((s, ch))

    if not scored:
        return chunks[:top_k]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


# ============================================================
# CONTEXT BUILDERS
# ============================================================
def compact_scholar_record(p: Dict[str, Any], idx: int) -> str:
    authors = ", ".join(p.get("authors", [])[:6]) if p.get("authors") else "Unknown authors"
    return (
        f"[Scholar {idx}]\n"
        f"Title: {p.get('title', '')}\n"
        f"Authors: {authors}\n"
        f"Year: {p.get('year', 'n.d.')}\n"
        f"Cited by: {p.get('cited_by_count', 0)}\n"
        f"Score: {p.get('score', 0)}\n"
        f"Snippet: {p.get('snippet', '')}\n"
        f"Summary: {p.get('summary', '')}\n"
        f"Link: {p.get('link', '')}\n"
    )


def build_scholar_context(papers: List[Dict[str, Any]], limit_chars: int = 14000) -> str:
    out = []
    used = 0
    for i, p in enumerate(papers, start=1):
        block = compact_scholar_record(p, i) + "\n"
        if used + len(block) > limit_chars:
            break
        out.append(block)
        used += len(block)
    return "\n".join(out)


def build_pdf_context(query: str, pdf_docs: List[Dict[str, Any]], top_k: int = 12, limit_chars: int = 24000) -> str:
    chunks = retrieve_relevant_chunks(query, pdf_docs, top_k=top_k)
    out = []
    used = 0
    for idx, ch in enumerate(chunks, start=1):
        block = (
            f"[Source Chunk {idx} | Source: {ch['source_name']} | Part: {ch['chunk_id']}]\n"
            f"{ch['text']}\n"
        )
        if used + len(block) > limit_chars:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)


# ============================================================
# GENERATION
# ============================================================
def build_instruction_block(
    request_text: str,
    min_words: int,
    paragraph_length: str,
    citation_style: str,
    writing_tone: str,
    strict_mode: bool,
) -> str:
    strict_rules = ""
    if strict_mode:
        strict_rules = """
STRICT MODE:
- Do not summarize unless asked.
- Meet or exceed minimum words.
- Follow the user's structure closely.
- Do not invent references or citations.
"""

    return f"""
You are an academic writing assistant.

User request:
{request_text}

Settings:
- Minimum words: {min_words}
- Paragraph length: {paragraph_length}
- Citation style: {citation_style}
- Writing tone: {writing_tone}

Core rules:
1. Prefer uploaded/source evidence when available.
2. Use scholar metadata only as support.
3. Do not invent references.
4. If evidence is incomplete, say so briefly.
5. Write clearly and naturally.
6. Use paragraph style requested.
7. Keep the response academically useful.

{strict_rules}
"""


def generate_writeup(
    openai_api_key: str,
    model_name: str,
    request_text: str,
    papers: List[Dict[str, Any]],
    pdf_docs: List[Dict[str, Any]],
    min_words: int,
    paragraph_length: str,
    citation_style: str,
    writing_tone: str,
    strict_mode: bool,
) -> str:
    client = OpenAI(api_key=openai_api_key)

    scholar_context = build_scholar_context(papers)
    pdf_context = build_pdf_context(request_text, pdf_docs)

    developer_prompt = build_instruction_block(
        request_text=request_text,
        min_words=min_words,
        paragraph_length=paragraph_length,
        citation_style=citation_style,
        writing_tone=writing_tone,
        strict_mode=strict_mode,
    )

    user_prompt = f"""
Scholar records:
{scholar_context}

Relevant source evidence:
{pdf_context}

Now write:
{request_text}
"""

    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text


def answer_followup(
    openai_api_key: str,
    model_name: str,
    user_query: str,
    papers: List[Dict[str, Any]],
    pdf_docs: List[Dict[str, Any]],
    last_generated_text: str,
    paragraph_length: str,
    min_words: int,
    citation_style: str,
    writing_tone: str,
    strict_mode: bool,
) -> str:
    client = OpenAI(api_key=openai_api_key)

    scholar_context = build_scholar_context(papers, limit_chars=10000)
    pdf_context = build_pdf_context(user_query, pdf_docs, top_k=10, limit_chars=18000)

    strict_note = "Follow instructions strictly." if strict_mode else "Follow instructions normally."

    developer_prompt = f"""
You are an academic writing chatbot.

Settings:
- Paragraph length: {paragraph_length}
- Minimum words: {min_words}
- Citation style: {citation_style}
- Writing tone: {writing_tone}

Rules:
1. Prefer source evidence.
2. Do not invent references.
3. Be direct and useful.
4. {strict_note}
"""

    user_prompt = f"""
Scholar records:
{scholar_context}

Relevant source evidence:
{pdf_context}

Previous output:
{last_generated_text}

User follow-up:
{user_query}
"""

    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="big-title">Scholar Writer Pro</div>
        <div class="sub-title">
            Search papers, add PDFs or direct article links, and generate academic notes.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Scope works by expanding the search topic with Nigeria/Africa-related terms. "
    "The library now accepts PDFs and direct article/journal links."
)


# ============================================================
# METRICS
# ============================================================
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f'<div class="metric-box">Raw Papers<br><strong>{len(st.session_state.papers_raw)}</strong></div>',
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        f'<div class="metric-box">Displayed Papers<br><strong>{len(st.session_state.papers_display)}</strong></div>',
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        f'<div class="metric-box">Stored Sources<br><strong>{len(st.session_state.pdf_docs)}</strong></div>',
        unsafe_allow_html=True
    )
with m4:
    st.markdown(
        f'<div class="metric-box">Generated Text<br><strong>{"Yes" if st.session_state.last_generated_text else "No"}</strong></div>',
        unsafe_allow_html=True
    )


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["Scholar Search", "PDF Library", "Write-up Bot"])


# ============================================================
# TAB 1: SCHOLAR SEARCH
# ============================================================
with tab1:
    st.subheader("Scholar Search")

    c1, c2 = st.columns([1.4, 0.6])

    with c1:
        search_query = st.text_input(
            "Search topic / keywords",
            value="effects of academic performance",
            key="search_query",
        )

    with c2:
        serpapi_key = st.text_input(
            "SerpApi Key",
            type="password",
            value=os.getenv("SERPAPI_KEY", ""),
            key="serpapi_key",
        )

    c3, c4, c5, c6 = st.columns(4)

    with c3:
        year_range = st.slider(
            "Year range",
            min_value=1990,
            max_value=2030,
            value=(2018, 2025),
            step=1,
            key="year_range",
        )

    with c4:
        author_scope = st.selectbox(
            "Scope",
            ["International", "African", "Nigeria"],
            index=2,
            key="author_scope",
        )

    with c5:
        total_results = st.slider(
            "Results to fetch per query",
            min_value=20,
            max_value=100,
            value=40,
            step=10,
            key="total_results",
            help="This is how many results to fetch for each search variation.",
        )

    with c6:
        display_limit = st.slider(
            "Results to display",
            min_value=10,
            max_value=100,
            value=30,
            step=10,
            key="display_limit",
        )

    search_btn = st.button("Search Papers", use_container_width=True, type="primary")

    start_year, end_year = year_range

    if search_btn:
        if not serpapi_key.strip():
            st.error("Enter your SerpApi key.")
        elif not search_query.strip():
            st.error("Enter a search query.")
        else:
            with st.spinner("Searching Scholar..."):
                try:
                    queries = build_scope_queries(search_query.strip(), author_scope)

                    raw = scholar_search_multi_query(
                        api_key=serpapi_key.strip(),
                        queries=queries,
                        start_year=start_year,
                        end_year=end_year,
                        total_results_per_query=total_results,
                    )

                    ranked = rank_papers(raw, display_limit=display_limit)

                    st.session_state.papers_raw = raw
                    st.session_state.papers_display = ranked

                    st.success(
                        f"Fetched {len(raw)} merged paper(s) from scope-based search. "
                        f"Showing top {len(ranked)} result(s)."
                    )

                except Exception as e:
                    st.error(f"Scholar search failed: {e}")

    if not st.session_state.papers_display:
        st.info("No papers loaded yet.")
    else:
        st.markdown("### Results")

        for i, p in enumerate(st.session_state.papers_display, start=1):
            authors = ", ".join(p.get("authors", [])[:6]) if p.get("authors") else "Unknown authors"
            year = p["year"] if p["year"] else "n.d."
            cited = p.get("cited_by_count", 0)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(f"#### {i}. {p['title']}")
            st.write(f"**Authors:** {authors}")
            st.write(f"**Year:** {year} | **Cited by:** {cited}")

            if p.get("snippet"):
                st.write(p["snippet"])

            a, b, c, d = st.columns(4)
            with a:
                if p.get("link"):
                    st.link_button("Open Result", p["link"], use_container_width=True)
            with b:
                if p.get("pdf_link"):
                    st.link_button("Open PDF", p["pdf_link"], use_container_width=True)
            with c:
                if p.get("cited_by_link"):
                    st.link_button("Cited By", p["cited_by_link"], use_container_width=True)
            with d:
                if p.get("pdf_link"):
                    if st.button(f"Add PDF {i}", key=f"add_pdf_{i}"):
                        try:
                            pdf_bytes = fetch_pdf_bytes_from_url(p["pdf_link"])
                            text, page_count = extract_text_from_pdf_bytes_smart(pdf_bytes)
                            add_pdf_doc(p["title"], text, page_count, p["pdf_link"])
                            st.success(f"Added PDF for: {p['title']}")
                        except Exception as e:
                            st.error(f"Could not add PDF: {e}")

            st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 2: PDF LIBRARY
# ============================================================
with tab2:
    st.subheader("PDF Library")

    left, right = st.columns([1, 1])

    with left:
        uploaded_pdfs = st.file_uploader(
            "Upload PDF papers",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploaded_pdfs",
        )

        if st.button("Ingest Uploaded PDFs", use_container_width=True):
            if not uploaded_pdfs:
                st.warning("Upload at least one PDF.")
            else:
                added = 0
                for file in uploaded_pdfs:
                    try:
                        text, page_count = extract_text_from_uploaded_pdf(file)
                        if text.strip():
                            add_pdf_doc(file.name, text, page_count)
                            added += 1
                    except Exception as e:
                        st.error(f"Failed on {file.name}: {e}")
                st.success(f"Ingested {added} PDF(s).")

    with right:
        st.markdown("### Add Links")

        pdf_url = st.text_input(
            "Paste direct PDF URL",
            value="",
            key="pdf_url",
        )

        article_url = st.text_input(
            "Paste direct article/journal URL",
            value="",
            key="article_url",
            help="Works for journal pages, article pages, and many publisher links.",
        )

        multi_links = st.text_area(
            "Paste multiple links (one per line)",
            value="",
            key="multi_links",
            height=130,
            help="You can mix PDF links and normal article links here.",
        )

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button("Ingest PDF URL", use_container_width=True):
                if not pdf_url.strip():
                    st.warning("Paste a PDF URL first.")
                else:
                    try:
                        pdf_bytes = fetch_pdf_bytes_from_url(pdf_url.strip())
                        text, page_count = extract_text_from_pdf_bytes_smart(pdf_bytes)
                        filename = pdf_url.strip().split("/")[-1] or "remote_pdf.pdf"
                        add_pdf_doc(filename, text, page_count, source_url=pdf_url.strip())
                        st.success("PDF from URL added successfully.")
                    except Exception as e:
                        st.error(f"Failed to ingest PDF URL: {e}")

        with col_b:
            if st.button("Ingest Article URL", use_container_width=True):
                if not article_url.strip():
                    st.warning("Paste an article URL first.")
                else:
                    try:
                        source_name, text, page_count = ingest_any_url(article_url.strip())
                        if text.strip():
                            if page_count == 1 and not article_url.strip().lower().endswith(".pdf"):
                                add_web_doc(source_name, text, source_url=article_url.strip())
                            else:
                                add_pdf_doc(source_name, text, page_count, source_url=article_url.strip())
                            st.success("Article URL added successfully.")
                        else:
                            st.error("No readable text was extracted from that article link.")
                    except Exception as e:
                        st.error(f"Failed to ingest article URL: {e}")

        with col_c:
            if st.button("Ingest All Links", use_container_width=True, type="primary"):
                if not multi_links.strip():
                    st.warning("Paste at least one link.")
                else:
                    links = [x.strip() for x in multi_links.splitlines() if x.strip()]
                    added = 0
                    failed = 0

                    for link in links:
                        try:
                            source_name, text, page_count = ingest_any_url(link)
                            if text.strip():
                                if page_count == 1 and not link.lower().endswith(".pdf"):
                                    add_web_doc(source_name, text, source_url=link)
                                else:
                                    add_pdf_doc(source_name, text, page_count, source_url=link)
                                added += 1
                            else:
                                failed += 1
                        except Exception as e:
                            failed += 1
                            st.error(f"Failed on {link}: {e}")

                    st.success(f"Added {added} link(s). Failed: {failed}.")

    st.markdown("---")
    st.markdown("### Stored Sources")

    if not st.session_state.pdf_docs:
        st.info("No sources in your library yet.")
    else:
        for i, doc in enumerate(st.session_state.pdf_docs, start=1):
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(f"#### {i}. {doc['source_name']}")
            st.write(f"**Pages:** {doc['page_count']} | **Words:** {doc['word_count']}")

            if doc.get("source_url"):
                st.link_button("Open Source URL", doc["source_url"])

            preview = doc["text"][:1200] + ("..." if len(doc["text"]) > 1200 else "")
            st.text_area(
                f"Preview {i}",
                preview,
                height=180,
                key=f"preview_{i}",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "Download Source Library JSON",
            data=json.dumps(st.session_state.pdf_docs, indent=2, ensure_ascii=False),
            file_name="source_library.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# TAB 3: WRITE-UP BOT
# ============================================================
with tab3:
    st.subheader("Write-up Bot")

    s1, s2 = st.columns([1, 1])

    with s1:
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            key="openai_key",
        )

    with s2:
        model_name = st.selectbox(
            "Model",
            ["gpt-5", "gpt-4.1", "gpt-4o-mini"],
            index=1,
            key="model_name",
        )

    w1, w2, w3, w4 = st.columns(4)

    with w1:
        paragraph_length = st.selectbox(
            "Paragraph length",
            ["Short", "Medium", "Long"],
            index=1,
            key="paragraph_length",
        )

    with w2:
        min_words = st.number_input(
            "Minimum words",
            min_value=300,
            max_value=10000,
            value=1500,
            step=100,
            key="min_words",
        )

    with w3:
        citation_style = st.selectbox(
            "Citation style",
            ["APA 7th", "MLA 9th", "Chicago", "Harvard"],
            index=0,
            key="citation_style",
        )

    with w4:
        writing_tone = st.selectbox(
            "Writing tone",
            ["Formal academic", "Simple academic", "Concise explanatory", "Critical analytical", "Nigerian thesis style"],
            index=0,
            key="writing_tone",
        )

    strict_mode = st.checkbox("Strict mode", value=True, key="strict_mode")

    writing_request = st.text_area(
        "What should the bot write?",
        value="Write notes on the effects of academic performance in Nigeria.",
        height=160,
        key="writing_request",
    )

    generate_btn = st.button("Generate Notes", use_container_width=True, type="primary")

    if generate_btn:
        if not openai_key.strip():
            st.error("Enter your OpenAI API key.")
        elif not writing_request.strip():
            st.error("Enter what you want to write.")
        elif not st.session_state.papers_display and not st.session_state.pdf_docs:
            st.error("Load papers or sources first.")
        else:
            with st.spinner("Generating..."):
                try:
                    generated = generate_writeup(
                        openai_api_key=openai_key.strip(),
                        model_name=model_name,
                        request_text=writing_request.strip(),
                        papers=st.session_state.papers_display,
                        pdf_docs=st.session_state.pdf_docs,
                        min_words=int(min_words),
                        paragraph_length=paragraph_length,
                        citation_style=citation_style,
                        writing_tone=writing_tone,
                        strict_mode=strict_mode,
                    )
                    st.session_state.last_generated_text = generated
                    st.success("Generation completed.")
                except Exception as e:
                    err = str(e)
                    if "insufficient_quota" in err or "429" in err:
                        st.error("Generation failed because your OpenAI API quota is exhausted or billing is inactive.")
                    else:
                        st.error(f"Generation failed: {e}")

    if st.session_state.last_generated_text:
        st.markdown("### Generated Output")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write(st.session_state.last_generated_text)
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "Download Output as TXT",
            data=st.session_state.last_generated_text,
            file_name="scholar_writer_output.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### Follow-up Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    followup = st.chat_input("Ask follow-up questions about your papers, sources, or generated notes...")
    if followup:
        st.session_state.messages.append({"role": "user", "content": followup})

        if not openai_key.strip():
            reply = "Enter your OpenAI API key first."
        elif not st.session_state.papers_display and not st.session_state.pdf_docs:
            reply = "Please search papers or ingest sources first."
        else:
            try:
                reply = answer_followup(
                    openai_api_key=openai_key.strip(),
                    model_name=model_name,
                    user_query=followup,
                    papers=st.session_state.papers_display,
                    pdf_docs=st.session_state.pdf_docs,
                    last_generated_text=st.session_state.last_generated_text,
                    paragraph_length=paragraph_length,
                    min_words=int(min_words),
                    citation_style=citation_style,
                    writing_tone=writing_tone,
                    strict_mode=strict_mode,
                )
            except Exception as e:
                err = str(e)
                if "insufficient_quota" in err or "429" in err:
                    reply = "I could not answer because your OpenAI API quota is exhausted or billing is inactive."
                else:
                    reply = f"Follow-up failed: {e}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)