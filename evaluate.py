"""
evaluate.py — offline RAG evaluation for AskmyDocs

Runs the same pipeline as app.py (create_db.ingest_pdf -> MMR retrieval ->
Groq LLM) over a list of question / ground-truth pairs and reports metrics.

Usage:
    python evaluate.py --pdf your_file.pdf --cases test_cases.json

test_cases.json format (see test_cases.example.json):
    [{"question": "...", "ground_truth": "...", "page": 3}, ...]
    "page" is optional (1-indexed); if given, retrieval page-hit is measured.

Local-only dependencies (do NOT add these to the Streamlit Cloud requirements.txt):
    pip install rouge-score sentence-transformers
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from statistics import mean
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

from create_db import ingest_pdf

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"  
RETRIEVER_KWARGS = {"k": 4, "fetch_k": 10, "lambda_mult": 0.5} 

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a friendly, helpful AI assistant that answers questions about an uploaded document.

- For greetings and small talk (hi, thanks, who are you), reply naturally and briefly, and invite the user to ask about the document.
- For questions about the document, use ONLY the provided context.
- If a document question can't be answered from the context, say: "I could not find the answer in the document."
""",
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}\n"),
    ]
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def with_retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if ("429" in msg or "rate limit" in msg) and i < tries - 1:
                time.sleep(2 * (i + 1))
                continue
            raise


def build_pipeline(pdf_path: str):
    persist_dir = tempfile.mkdtemp(prefix="askmydocs_eval_")
    vector_store, n_chunks = with_retry(
        lambda: ingest_pdf(pdf_path, persist_dir=persist_dir, api_key=os.getenv("MISTRAL_API_KEY")),
        tries=3,
    )
    if n_chunks == 0:
        raise SystemExit("No extractable text found in this PDF.")
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs=RETRIEVER_KWARGS)
    llm = ChatGroq(model=GROQ_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=0)
    print(f"[pipeline] {n_chunks} chunks indexed | LLM: {GROQ_MODEL}")
    return retriever, llm, persist_dir


def run_query(question: str, retriever, llm) -> dict:
    t0 = time.time()
    docs = with_retry(lambda: retriever.invoke(question))
    t_retrieval = time.time() - t0

    context = "\n\n".join(d.page_content for d in docs)
    t1 = time.time()
    response = with_retry(lambda: llm.invoke(PROMPT.invoke({"context": context, "question": question})))
    t_generation = time.time() - t1

    return {
        "answer": response.content,
        "context": context,
        "chunks": [d.page_content for d in docs],
        "pages": sorted({d.metadata.get("page", -1) + 1 for d in docs}),
        "retrieval_s": round(t_retrieval, 3),
        "generation_s": round(t_generation, 3),
    }


# ---------------------------------------------------------------------------
# Metrics (local, no API cost). These are proxies, not human judgement.
# ---------------------------------------------------------------------------
_sem = SentenceTransformer("all-MiniLM-L6-v2")
_rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def sim(a: str, b: str) -> float:
    e = _sem.encode([a, b], convert_to_tensor=True)
    return round(float(util.cos_sim(e[0], e[1])), 4)


def context_relevance(question: str, chunks: List[str]) -> float:
    return round(mean(sim(question, c) for c in chunks), 4) if chunks else 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    question: str
    ground_truth: str
    predicted_answer: str
    refused: bool                 # model said "could not find the answer"
    semantic_sim: float           # answer vs ground truth
    rouge1: float
    rougeL: float
    faithfulness: float           # answer vs retrieved context (proxy)
    context_relevance: float      # question vs retrieved chunks
    answer_relevance: float       # question vs answer
    retrieval_s: float
    generation_s: float
    retrieved_pages: List[int]
    page_hit: Optional[bool]      # only if the test case gave a "page"


def evaluate(pdf_path: str, cases_path: str, output_json: str, delay: float):
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)
    if not cases:
        raise SystemExit("No test cases found.")

    print("\n=== AskmyDocs RAG evaluation ===\n")
    retriever, llm, persist_dir = build_pipeline(pdf_path)
    results: List[EvalResult] = []

    try:
        for i, tc in enumerate(cases, 1):
            q, gt = tc["question"], tc["ground_truth"]
            print(f"[{i}/{len(cases)}] {q[:70]}")

            out = run_query(q, retriever, llm)
            ans = out["answer"]
            rg = _rouge.score(gt, ans)
            page_hit = (tc["page"] in out["pages"]) if "page" in tc else None

            r = EvalResult(
                question=q,
                ground_truth=gt,
                predicted_answer=ans,
                refused="could not find the answer" in ans.lower(),
                semantic_sim=sim(ans, gt),
                rouge1=round(rg["rouge1"].fmeasure, 4),
                rougeL=round(rg["rougeL"].fmeasure, 4),
                faithfulness=sim(ans, out["context"]),
                context_relevance=context_relevance(q, out["chunks"]),
                answer_relevance=sim(q, ans),
                retrieval_s=out["retrieval_s"],
                generation_s=out["generation_s"],
                retrieved_pages=out["pages"],
                page_hit=page_hit,
            )
            results.append(r)
            print(f"     sim={r.semantic_sim} faith={r.faithfulness} refused={r.refused} "
                  f"gen={r.generation_s}s\n")
            time.sleep(delay)  # stay under free-tier rate limits
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)

    def avg(key):
        return round(mean(getattr(r, key) for r in results), 4)

    hits = [r.page_hit for r in results if r.page_hit is not None]
    summary = {
        "total_questions": len(results),
        "avg_semantic_similarity": avg("semantic_sim"),
        "avg_rouge1": avg("rouge1"),
        "avg_rougeL": avg("rougeL"),
        "avg_faithfulness": avg("faithfulness"),
        "avg_context_relevance": avg("context_relevance"),
        "avg_answer_relevance": avg("answer_relevance"),
        "avg_retrieval_s": avg("retrieval_s"),
        "avg_generation_s": avg("generation_s"),
        "refusal_rate": round(sum(r.refused for r in results) / len(results), 4),
        "page_hit_rate": round(sum(hits) / len(hits), 4) if hits else None,
    }

    print("--- AGGREGATE ---")
    for k, v in summary.items():
        print(f"  {k:<26} {v}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": [asdict(r) for r in results]}, f, indent=2)
    print(f"\nSaved -> {output_json}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate the AskmyDocs RAG pipeline")
    ap.add_argument("--pdf", required=True, help="Path to the test PDF")
    ap.add_argument("--cases", default="test_cases.json", help="JSON file of question/ground_truth pairs")
    ap.add_argument("--output", default="eval_results.json")
    ap.add_argument("--delay", type=float, default=2.0, help="Seconds between questions (rate limits)")
    args = ap.parse_args()
    evaluate(args.pdf, args.cases, args.output, args.delay)
