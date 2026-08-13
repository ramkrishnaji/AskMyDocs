# AskmyDocs

A Retrieval-Augmented Generation (RAG) app that lets you chat with any PDF.
Upload a document and ask questions answered strictly from its contents,
with a visible 4-step processing pipeline and source-page citations for every answer.

**Live demo:** _add your Streamlit Cloud link here after deploying_

## Features
- Upload any PDF through a Streamlit UI and query it in natural language
- Visible pipeline: **Upload → Chunking → Embedding → Retrieval & Answer**
- Source citations — every answer shows which page(s) it was drawn from
- Vector search with Chroma (MMR retrieval for diverse, relevant context)
- Grounded answers — the model is instructed to answer only from retrieved context
- Error handling for empty/corrupt/scanned PDFs and API rate limits
- CLI mode also available for terminal-based usage

## Folder structure
```
AskmyDocs/
├── app.py              # Streamlit UI: upload, pipeline view, chat + citations
├── create_db.py         # Ingestion: PDF -> chunks -> embeddings -> Chroma DB
│                         # (standalone CLI, also imported by app.py)
├── main.py              # CLI chat script (queries an existing Chroma DB)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup (local)

1. Install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your Mistral API key:
   ```bash
   cp .env.example .env
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

### CLI alternative
```bash
python create_db.py path/to/your_file.pdf
python main.py
```

## Deploying on Streamlit Community Cloud

1. Push this folder to a public GitHub repo (make sure `.env` is **not** committed — it's in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing to your repo, branch, and `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```
   MISTRAL_API_KEY = "your_mistral_api_key_here"
   ```
4. Deploy. `app.py` reads the key from `st.secrets` automatically on Cloud (and from `.env` locally).

## Tech stack
- **LangChain** — document loading, chunking, retrieval orchestration
- **Mistral AI** — `mistral-embed` for embeddings, `mistral-small-2506` for generation
- **Chroma** — vector database
- **Streamlit** — web UI

## Roadmap
- Persistent chat history across sessions
- Multi-PDF support
- Streaming responses
