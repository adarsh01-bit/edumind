# rag_pipeline.py
# Updated: supports both Ollama (local) and Groq (cloud)

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from config import USE_CLOUD, GROQ_API_KEY

import os

# import correct LLM based on environment
if USE_CLOUD:
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
else:
    from langchain_ollama import OllamaEmbeddings, ChatOllama


# ── CONFIGURATION ────────────────────────────────────────

VECTORSTORE_DIR = "./vectorstore"
TOP_K_CHUNKS = 4
TEMPERATURE = 0.1

# model names
OLLAMA_LLM = "llama3.2"
OLLAMA_EMBED = "nomic-embed-text"
GROQ_LLM = "llama3-8b-8192"  # free Groq model


# ── PROMPT ───────────────────────────────────────────────

QA_PROMPT_TEMPLATE = """You are EduMind, a helpful AI assistant
for students. You help students understand their academic documents.

Use ONLY the context provided below to answer the question.
Do not use any outside knowledge.

If the answer is not explicitly written in the document,
do NOT guess or invent information.

Only summarize or quote from the provided context.

If the answer is clearly present → answer directly and clearly.
If the answer is partially present → share what you found.
If the answer is not in the context → say exactly:
"I couldn't find this information in the uploaded document."

Always be helpful, clear, and student-friendly.

─────────────────────────────────
CONTEXT FROM DOCUMENT:
{context}
─────────────────────────────────

STUDENT'S QUESTION: {question}

YOUR ANSWER:"""


# ── FUNCTION 1: Get Embeddings ────────────────────────────


def get_embeddings():
    """Returns correct embedding model based on environment."""

    if USE_CLOUD:
        # use HuggingFace embeddings on cloud (free, no API)
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        return OllamaEmbeddings(model=OLLAMA_EMBED, base_url="http://localhost:11434")


# ── FUNCTION 2: Get LLM ───────────────────────────────────


def get_llm():
    """Returns correct LLM based on environment."""

    if USE_CLOUD:
        return ChatGroq(
            model=GROQ_LLM, temperature=TEMPERATURE, api_key=os.getenv("GROQ_API_KEY")
        )
    else:
        return ChatOllama(
            model=OLLAMA_LLM,
            temperature=TEMPERATURE,
            base_url="http://localhost:11434",
            num_ctx=4096,
        )


print("USE_GROQ =", USE_CLOUD)

# ── FUNCTION 3: Load ChromaDB ─────────────────────────────


def load_vectorstore():
    """Loads ChromaDB with correct embeddings."""

    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(
            "No vectorstore found. " "Please upload and process a PDF first."
        )

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR, embedding_function=get_embeddings()
    )
    return vectorstore


# ── FUNCTION 4: Format Docs ───────────────────────────────


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Section {i+1}]:\n{doc.page_content}" for i, doc in enumerate(docs)
    )


# ── FUNCTION 5: Build RAG Chain ───────────────────────────


def build_rag_chain():
    """Builds complete RAG pipeline."""

    print("🔧 Building RAG chain...")

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K_CHUNKS}
    )

    llm = get_llm()
    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    parser = StrOutputParser()

    rag_chain = (
        RunnableMap(
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
        )
        | prompt
        | llm
        | parser
    )

    print("✅ RAG chain ready!")
    return rag_chain, retriever


# ── FUNCTION 6: Ask Question ──────────────────────────────


def ask_question(question, rag_chain, retriever):
    if not question or len(question.strip()) == 0:
        return "Please type a question.", []

    answer = rag_chain.invoke(question)
    source_docs = retriever.invoke(question)
    return answer, source_docs


# ── FUNCTION 7: Format Sources ────────────────────────────


def format_sources(source_docs):
    formatted = []
    for i, doc in enumerate(source_docs):
        formatted.append(
            {
                "chunk_number": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index", "?"),
            }
        )
    return formatted


# ── FUNCTION 8: Is Ready ──────────────────────────────────


def is_ready():
    return os.path.exists(VECTORSTORE_DIR) and len(os.listdir(VECTORSTORE_DIR)) > 0
