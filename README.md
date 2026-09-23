# AskmyDocs

A Retrieval-Augmented Generation (RAG) app that lets you chat with any PDF.
Upload a document, process it, and ask questions answered strictly from its contents,
with an animated processing pipeline and source-page citations for every answer.

**Live demo:** https://askmydocs-xyrzpkpqsxuvvypvdptxvu.streamlit.app

## Features
- Upload any PDF through a Streamlit UI and query it in natural language
- Visible pipeline on a **Process Document** click: **Reading → Chunking → Embedding → Indexing**
- Source citations — every answer shows which page(s) it was drawn from, with excerpts (toggleable)
- Vector search with Chroma (MMR retrieval for diverse, relevant context)
- Grounded answers — the model is instructed to answer only from retrieved context
- Automatic retry with backoff on API rate limits (429), plus clear error messages
- Handles empty, corrupt and scanned (image-only) PDFs gracefully
- Custom dark UI theme (Syne + DM Sans, purple accent)
- Offline evaluation script (`evaluate.py`) for measuring answer quality and latency
- CLI mode also available for terminal-based usage

## Folder structure
```
AskmyDocs/
├── app.py                    # Streamlit UI: upload, pipeline view, chat + citations
├── create_db.py              # Ingestion: PDF -> chunks -> embeddings -> Chroma DB
│                             # (standalone CLI, also imported by app.py)
├── main.py                   # CLI chat script (queries an existing Chroma DB)
├── evaluate.py               # Offline RAG evaluation (similarity, ROUGE, latency, ...)
├── test_cases.example.json   # Template for evaluation question/answer pairs
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

2. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here   # embeddings
   GROQ_API_KEY=your_groq_api_key_here         # answer generation
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

### Evaluation (optional)
```bash
pip install rouge-score sentence-transformers   # local only, not needed for the deployed app
cp test_cases.example.json test_cases.json      # fill in question / ground_truth pairs for your PDF
python evaluate.py --pdf your_file.pdf --cases test_cases.json
```
Reports semantic similarity, ROUGE-1/L, faithfulness and relevance proxies, refusal rate,
optional page-hit rate, and retrieval/generation latency. Results are saved to `eval_results.json`.

## Deploying on Streamlit Community Cloud

1. Push this folder to a public GitHub repo (make sure `.env` is **not** committed — it's in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing to your repo, branch, and `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```
   MISTRAL_API_KEY = "your_mistral_api_key_here"
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
4. Deploy. `app.py` reads the keys from `st.secrets` automatically on Cloud (and from `.env` locally).

## Tech stack
- **LangChain** — document loading, chunking, retrieval orchestration
- **Mistral AI** — `mistral-embed` for embeddings
- **Groq** — `openai/gpt-oss-120b` for answer generation
- **Chroma** — vector database
- **Streamlit** — web UI

## Roadmap
- Persistent chat history across sessions
- Multi-PDF support
- Streaming responses
