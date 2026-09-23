"""
main.py

Standalone CLI chat script.
Queries an existing Chroma vector store built by create_db.py.

Usage:
    python create_db.py path/to/file.pdf   # run once first
    python main.py                          # then chat
"""

import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

PERSIST_DIR = "chroma_db"

if not os.path.exists(PERSIST_DIR):
    raise SystemExit(
        f"No vector store found at '{PERSIST_DIR}'. "
        f"Run `python create_db.py <your_pdf>` first."
    )

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
)


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0) 

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

print("RAG system created")
print("press 0 to exit")

while True:
    query = input("You : ")
    if query == "0":
        break

    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")
