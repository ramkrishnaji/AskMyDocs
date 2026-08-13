"""
app.py

Streamlit UI for AskmyDocs.
Upload a PDF -> watch it move through Upload -> Chunking -> Embedding ->
Retrieval & Answer -> chat with citations back to the source page.
"""

import os
import tempfile
import shutil

import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from create_db import ingest_pdf

load_dotenv()

st.set_page_config(page_title="AskmyDocs", page_icon="📄", layout="wide")


def get_api_key():
    # Works both locally (.env) and on Streamlit Cloud (st.secrets)
    return os.getenv("MISTRAL_API_KEY") or st.secrets.get("MISTRAL_API_KEY", None)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "messages": [],
    "vector_store": None,
    "current_file_name": None,
    "chroma_dir": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.chroma_dir is None:
    st.session_state.chroma_dir = tempfile.mkdtemp(prefix="chroma_")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2506", api_key=get_api_key())


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


def answer_question(vector_store, query: str):
    """Retrieve relevant chunks, generate an answer, return (answer, sources)."""
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({"context": context, "question": query})

    try:
        response = get_llm().invoke(final_prompt)
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "429" in msg:
            return (
                "The AI service is temporarily rate-limited. Please wait a few seconds and try again.",
                [],
            )
        return (f"Something went wrong while generating the answer: {e}", [])

    # build source citations (page numbers, deduplicated, in order of relevance)
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


# ---------------------------------------------------------------------------
# Sidebar - PDF upload with 4-step pipeline visualization
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None and st.session_state.current_file_name != uploaded_file.name:
        if not get_api_key():
            st.error("MISTRAL_API_KEY not found. Add it to your .env file or Streamlit secrets.")
        else:
            step_box = st.status("Processing your document...", expanded=True)

            tmp_dir = tempfile.mkdtemp(prefix="upload_")
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)

            try:
                # Step 1: Upload
                step_box.write("**1. Upload** — saving file...")
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if os.path.getsize(tmp_path) == 0:
                    raise ValueError("The uploaded file is empty.")
                step_box.write("✅ File received")

                # Step 2: Chunking
                step_box.write("**2. Chunking** — splitting the document into passages...")
                # Step 3: Embedding happens inside ingest_pdf (chunking + embedding + storing)
                step_box.write("**3. Embedding** — generating vector embeddings...")

                vector_store, n_chunks = ingest_pdf(
                    tmp_path,
                    persist_dir=st.session_state.chroma_dir,
                    api_key=get_api_key(),
                )

                if n_chunks == 0:
                    raise ValueError(
                        "No extractable text was found in this PDF (it may be scanned/image-only)."
                    )

                step_box.write(f"✅ {n_chunks} chunks embedded and indexed")

                # Step 4: Retrieval & Answer (ready state)
                step_box.write("**4. Retrieval & Answer** — ready to take your questions")

                st.session_state.vector_store = vector_store
                st.session_state.current_file_name = uploaded_file.name
                st.session_state.messages = []

                step_box.update(label=f"'{uploaded_file.name}' indexed successfully", state="complete")

            except Exception as e:
                step_box.update(label="Processing failed", state="error")
                st.error(f"Could not process this PDF: {e}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    if st.session_state.current_file_name:
        st.info(f"Active document: **{st.session_state.current_file_name}**")
        if st.button("Clear document / start over"):
            st.session_state.vector_store = None
            st.session_state.current_file_name = None
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Pipeline: Upload → Chunking → Embedding → Retrieval & Answer")

# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------
st.title("📄 AskmyDocs")
st.caption("Upload a PDF on the left, then ask questions about it below. Answers cite the source page.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['label']}** — {s['snippet']}")

query = st.chat_input(
    "Ask a question about the document..."
    if st.session_state.vector_store
    else "Upload a PDF first, then ask your question here..."
)

if query:
    if st.session_state.vector_store is None:
        st.warning("Please upload a PDF before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": query, "sources": []})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages and generating an answer..."):
                answer, sources = answer_question(st.session_state.vector_store, query)
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"**{s['label']}** — {s['snippet']}")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
