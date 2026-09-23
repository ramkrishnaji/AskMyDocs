"""
app.py

Streamlit UI for AskmyDocs.
Upload a PDF -> Upload -> Chunking -> Embedding -> Retrieval & Answer ->
chat with citations back to the source page.
"""

import os
import time
import tempfile
import shutil

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from create_db import ingest_pdf

load_dotenv()

st.set_page_config(page_title="AskmyDocs — AI Document Assistant", page_icon="✦", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --bg: #0d0d18;
  --panel: #12121e;
  --card: #1a1a2b;
  --line: #262638;
  --text: #ecebf5;
  --muted: #9a99b0;
  --accent: #7c6cf0;
  --accent-soft: #a596ff;
}

html, body, .stApp, .stMarkdown, p, label, span, button, input, textarea {
  font-family: 'DM Sans', sans-serif;
}
.stApp { background: var(--bg); color: var(--text); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 3rem; max-width: 1100px; }

/* sidebar */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
.brand { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2rem; color: var(--accent-soft); margin: 0; }
.brand-sub { font-size: .65rem; letter-spacing: .18em; color: var(--muted); margin: .2rem 0 1.2rem; text-transform: uppercase; }
.model-badge { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: .6rem .8rem; font-size: .78rem; margin-bottom: 1rem; }
.dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#3ddc97; margin-right:.5rem; }
.side-label { font-size: .65rem; letter-spacing: .16em; color: var(--muted); text-transform: uppercase; margin: 1rem 0 .4rem; }
.stack { font-size: .7rem; color: var(--muted); text-align: center; margin-top: 1.5rem; }

[data-testid="stFileUploader"] section { background: var(--card); border: 1px dashed var(--accent); border-radius: 12px; }
[data-testid="stFileUploader"] button { background: var(--card); border: 1px solid var(--line); color: var(--text); }
[data-testid="stStatusWidget"], [data-testid="stExpander"], details {
  background: var(--card) !important; border: 1px solid var(--line) !important; border-radius: 10px !important;
}

section[data-testid="stSidebar"] .stButton > button {
  width: 100%; background: var(--accent); color: #fff; border: 0; border-radius: 10px;
  padding: .7rem 1rem; font-weight: 600; box-shadow: 0 6px 24px rgba(124,108,240,.35);
}
section[data-testid="stSidebar"] .stButton > button:hover { background: #8c7df5; color: #fff; }
[data-testid="stToggle"] [role="checkbox"][aria-checked="true"] { background: var(--accent); }

/* hero */
.eyebrow { font-size: .7rem; letter-spacing: .18em; color: var(--accent-soft); text-transform: uppercase; }
.eyebrow::before { content: ""; display:inline-block; width:18px; height:2px; background: var(--accent); margin-right:.6rem; vertical-align:middle; }
.hero { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 3.6rem; line-height: 1.02; margin: .6rem 0 1rem; color: #fff; }
.hero span { color: var(--accent-soft); }
.hero-sub { color: var(--muted); max-width: 560px; line-height: 1.6; }
.pill { display:inline-flex; align-items:center; gap:.5rem; border:1px solid var(--line); border-radius:999px;
  padding:.35rem .9rem; font-size:.75rem; color: var(--muted); margin:1.2rem 0 1.5rem; }
.pill.on { color: var(--text); border-color: var(--accent); }
.pill .d { width:6px; height:6px; border-radius:50%; background: var(--muted); }
.pill.on .d { background:#3ddc97; }
.rule { border-top: 1px solid var(--line); margin-bottom: 1.5rem; }

/* empty state */
.empty { text-align:center; padding: 5rem 0 3rem; }
.empty .icon { width:62px; height:62px; margin:0 auto 1.2rem; border-radius:16px; background: var(--card);
  border:1px solid var(--line); display:flex; align-items:center; justify-content:center; font-size:1.6rem; color:#fff; }
.empty h3 { font-family:'Syne',sans-serif; font-weight:800; font-size:1.4rem; margin:0 0 .6rem; }
.empty p { color: var(--muted); font-size:.9rem; max-width:340px; margin:0 auto 1.4rem; }
.chip { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:.35rem .9rem;
  font-size:.75rem; color: var(--muted); margin:0 .25rem; }

/* chat */
[data-testid="stChatMessage"] { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.2rem; }
[data-testid="stChatInput"] { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; }
[data-testid="stChatInput"] button { background: var(--accent); color: #fff; }

/* processing steps */
.steps { display:flex; gap:.5rem; margin:.8rem 0; }
.step { flex:1; text-align:center; font-size:.7rem; padding:.6rem .2rem; border:1px solid var(--line);
  border-radius:10px; background: var(--card); color: var(--muted); }
.step .i { display:block; font-size:1.1rem; margin-bottom:.2rem; }
.step.done { border-color:#1f5c45; color: var(--text); background:#13241f; }
.step.active { border-color: var(--accent); color: var(--text); }
.step.active .i { animation: pulse 1s infinite; }
@keyframes pulse { 50% { opacity:.3; } }
.bar { height:24px; border-radius:999px; color:#fff; font-size:.72rem; display:flex; align-items:center; padding:0 .9rem;
  background: linear-gradient(90deg,#7c6cf0,#a596ff,#7c6cf0); background-size:200% 100%; animation: slide 1.4s linear infinite; }
@keyframes slide { to { background-position:-200% 0; } }
</style>
""",
    unsafe_allow_html=True,
)


def get_secret(name):
    # Works both locally (.env) and on Streamlit Cloud (st.secrets)
    return os.getenv(name) or st.secrets.get(name, None)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "messages": [],
    "vector_store": None,
    "current_file_name": None,
    "n_chunks": 0,
    "chroma_dir": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.chroma_dir is None:
    st.session_state.chroma_dir = tempfile.mkdtemp(prefix="chroma_")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", api_key=get_secret("GROQ_API_KEY"), temperature=0)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


def is_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg


def with_retry(fn, tries=4):
    """Retry on 429 rate limits with a growing pause."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if is_rate_limit(e) and i < tries - 1:
                time.sleep(2 * (i + 1))
                continue
            raise


def answer_question(vector_store, query: str):
    """Retrieve relevant chunks, generate an answer, return (answer, sources)."""
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    try:
        docs = with_retry(lambda: retriever.invoke(query))
        context = "\n\n".join(doc.page_content for doc in docs)
        final_prompt = prompt.invoke({"context": context, "question": query})
        response = with_retry(lambda: get_llm().invoke(final_prompt))
    except Exception as e:
        if is_rate_limit(e):
            return ("Groq is still rate-limiting after several retries. Wait a few seconds and try again.", [])
        return (f"Something went wrong while generating the answer: {e}", [])

    # source citations (page numbers, deduplicated, in order of relevance)
    sources = []
    seen_pages = set()
    for doc in docs:
        page = doc.metadata.get("page", None)
        page_label = f"Page {page + 1}" if page is not None else "Unknown page"
        if page_label not in seen_pages:
            seen_pages.add(page_label)
            snippet = doc.page_content[:180].strip().replace("\n", " ")
            sources.append({"label": page_label, "snippet": snippet + "..."})

    return response.content, sources


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['label']}** — {s['snippet']}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<p class="brand">✦ AskmyDocs</p><p class="brand-sub">AI Document Intelligence</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="model-badge"><span class="dot"></span>llama-3.3-70b · Groq · RAG pipeline</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="side-label">Document</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")

    def tiles(states):
        labels = [("Reading", "📄"), ("Chunking", "✂️"), ("Embedding", "🧠"), ("Indexing", "💾")]
        html = "".join(
            f'<div class="step {st_}"><span class="i">{"✓" if st_ == "done" else ic}</span>{name}</div>'
            for (name, ic), st_ in zip(labels, states)
        )
        return f'<div class="steps">{html}</div>'

    is_new_file = uploaded_file is not None and st.session_state.current_file_name != uploaded_file.name

    if is_new_file:
        if not get_secret("MISTRAL_API_KEY") or not get_secret("GROQ_API_KEY"):
            st.error("Add MISTRAL_API_KEY and GROQ_API_KEY to your .env or Streamlit secrets.")
        elif st.button("⚡ Process Document", key="process_btn"):
            steps_box = st.empty()
            bar_box = st.empty()

            def show(states, label):
                steps_box.markdown(tiles(states), unsafe_allow_html=True)
                bar_box.markdown(f'<div class="bar">{label}</div>', unsafe_allow_html=True)

            tmp_dir = tempfile.mkdtemp(prefix="upload_")
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            try:
                show(["active", "pending", "pending", "pending"], "Reading PDF…")
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if os.path.getsize(tmp_path) == 0:
                    raise ValueError("The uploaded file is empty.")
                time.sleep(0.4)

                show(["done", "active", "pending", "pending"], "Chunking…")
                time.sleep(0.4)

                show(["done", "done", "active", "pending"], "Building index…")
                vector_store, n_chunks = with_retry(
                    lambda: ingest_pdf(
                        tmp_path,
                        persist_dir=st.session_state.chroma_dir,
                        api_key=get_secret("MISTRAL_API_KEY"),
                    ),
                    tries=3,
                )
                if n_chunks == 0:
                    raise ValueError(
                        "No extractable text was found in this PDF (it may be scanned/image-only)."
                    )

                show(["done", "done", "done", "done"], "Ready")
                st.session_state.vector_store = vector_store
                st.session_state.current_file_name = uploaded_file.name
                st.session_state.n_chunks = n_chunks
                st.session_state.messages = []
                time.sleep(0.8)
                st.rerun()

            except Exception as e:
                steps_box.empty()
                bar_box.empty()
                st.error(f"Could not process this PDF: {e}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    st.markdown('<p class="side-label">Options</p>', unsafe_allow_html=True)
    show_sources = st.toggle("Show source excerpts", value=True)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<p class="stack">LangChain · ChromaDB · Groq · Mistral · Streamlit</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
ready = st.session_state.vector_store is not None

if ready:
    pill = (
        f'<div class="pill on"><span class="d"></span>'
        f'{st.session_state.current_file_name} · {st.session_state.n_chunks} chunks</div>'
    )
else:
    pill = '<div class="pill"><span class="d"></span>No document</div>'

st.markdown(
    f"""
<div class="eyebrow">AI Document Intelligence</div>
<div class="hero">Ask your <span>documents</span><br>anything.</div>
<div class="hero-sub">Upload a PDF and have a natural conversation with its contents — answers cite the source page.</div>
{pill}
<div class="rule"></div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
if not ready and not st.session_state.messages:
    st.markdown(
        """
<div class="empty">
  <div class="icon">✦</div>
  <h3>Ready when you are</h3>
  <p>Upload a PDF from the sidebar to start your AI-powered document conversation.</p>
  <span class="chip">Research papers</span><span class="chip">Business reports</span>
  <span class="chip">Textbooks</span><span class="chip">Legal docs</span>
</div>
""",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            render_sources(msg["sources"])

query = st.chat_input(
    "Ask a question about the document..." if ready else "Upload a PDF first to start chatting...",
    disabled=not ready,
)

if query:
    st.session_state.messages.append({"role": "user", "content": query, "sources": []})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and generating an answer..."):
            answer, sources = answer_question(st.session_state.vector_store, query)
            st.markdown(answer)
            if sources and show_sources:
                render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
