# ingest.py
# PURPOSE: Read a PDF, split into chunks, convert to
#          numbers, save in ChromaDB for AI to search later
#
# BEGINNER NOTE: Read every comment carefully.
# Each section is explained in plain English.

# ── IMPORTS ──────────────────────────────────────────────
# These are tools we're borrowing from installed libraries

import pdfplumber  # reads PDF files
import os  # interacts with files/folders
import shutil  # helps delete folders
from config import USE_CLOUD
from config import USE_CLOUD
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ↑ splits long text into smaller overlapping chunks

from langchain_ollama import OllamaEmbeddings

# ↑ connects to Ollama to convert text → numbers

from langchain_community.vectorstores import Chroma

# ↑ ChromaDB — our local database to store the numbers

from langchain_core.documents import Document

# ↑ LangChain's standard format for storing text + info


# ── CONFIGURATION ────────────────────────────────────────
# Settings you can change if needed

VECTORSTORE_DIR = "vectorstore"  # folder to save ChromaDB data
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks
EMBEDDING_MODEL = "nomic-embed-text"  # Ollama model for embeddings


# ── FUNCTION 1: Read PDF ─────────────────────────────────


def load_pdf(file_path):
    """
    Opens a PDF and extracts all text from every page.
    Returns the full text as one big string.
    """

    print(f"📄 Reading PDF: {file_path}")

    full_text = ""  # start with empty text
    page_count = 0  # count how many pages we read

    # open the PDF using pdfplumber
    with pdfplumber.open(file_path) as pdf:

        # loop through every page one by one
        for page_number, page in enumerate(pdf.pages):

            # extract text from this page
            page_text = page.extract_text()

            # some pages might be images/blank — skip those
            if page_text and len(page_text.strip()) > 0:
                full_text += page_text + "\n\n"
                page_count += 1

    print(f"✅ Read {page_count} pages, {len(full_text)} characters extracted")

    # if no text was found, the PDF might be scanned image
    if len(full_text.strip()) == 0:
        raise ValueError(
            "No text found in PDF. "
            "This might be a scanned image PDF. "
            "Please use a text-based PDF."
        )

    return full_text


# ── FUNCTION 2: Split Into Chunks ────────────────────────


def split_into_chunks(text, file_name):
    """
    Takes the full text and splits it into smaller pieces.
    Each piece becomes one searchable unit in ChromaDB.

    file_name: we save this so we know which PDF each chunk came from
    """

    print(f"✂️  Splitting text into chunks...")

    # create the splitter with our settings
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # each chunk = max 1000 chars
        chunk_overlap=CHUNK_OVERLAP,  # 200 chars repeated between chunks
        length_function=len,  # measure by character count
        separators=["\n\n", "\n", ". ", " ", ""],
        # ↑ tries to split at paragraph breaks first,
        #   then line breaks, then sentences, then words
    )

    # split the text into raw string chunks
    raw_chunks = splitter.split_text(text)

    # wrap each chunk in a Document object
    # Document stores: the text + metadata (extra info)
    documents = []
    for i, chunk in enumerate(raw_chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": file_name,  # which PDF this came from
                "chunk_index": i,  # which chunk number
                "total_chunks": len(raw_chunks),
            },
        )
        documents.append(doc)

    print(f"✅ Created {len(documents)} chunks from the document")
    return documents


# ── FUNCTION 3: Save to ChromaDB ─────────────────────────

VECTORSTORE_DIR = "vectorstore"


def save_to_chromadb(documents, reset=False):
    """
    Converts each chunk to a number vector using Ollama,
    then saves everything in ChromaDB locally.

    reset=True  → deletes old data before saving (fresh start)
    reset=False → adds to existing data (keep old PDFs)
    """

    print("🧠 Converting chunks to embeddings...")
    print("   (This may take 2-5 minutes for first run)")
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    # if reset is True, delete old vectorstore data
    if reset and os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
        print("🗑️  Cleared old vectorstore")

    # create the embedding model connection
    # this connects to Ollama running on your Mac

    if USE_CLOUD:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url="http://localhost:11434",
        )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    print(f"✅ Saved {len(documents)} chunks to ChromaDB")
    print(f"📁 Data stored in: {VECTORSTORE_DIR}/")

    return vectorstore


# ── FUNCTION 4: Load Existing ChromaDB ───────────────────


def load_chromadb():
    """
    Loads previously saved ChromaDB data.
    Used when the app starts — no need to re-process PDFs.
    """

    # create embedding model based on environment

    if USE_CLOUD:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434",
    )

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR, embedding_function=embeddings
    )

    return vectorstore


# ── FUNCTION 5: Check If Data Exists ─────────────────────


def vectorstore_exists():
    """
    Returns True if ChromaDB already has data saved.
    Returns False if no documents have been ingested yet.
    """
    return os.path.exists(VECTORSTORE_DIR) and len(os.listdir(VECTORSTORE_DIR)) > 0


# ── MAIN FUNCTION: Run Everything ────────────────────────


def ingest_document(file_path, reset=False):
    """
    MASTER FUNCTION — runs all steps in order.
    Call this with a PDF path to fully process it.

    file_path: full path to the PDF file
    reset: True = clear old data first, False = add to existing
    """

    print("\n" + "=" * 50)
    print("🚀 EduMind Document Ingestion Started")
    print("=" * 50)

    # get just the filename (not full path)
    file_name = os.path.basename(file_path)

    # Step 1: Read the PDF
    text = load_pdf(file_path)

    # Step 2: Split into chunks
    documents = split_into_chunks(text, file_name)

    # Step 3: Save to ChromaDB
    vectorstore = save_to_chromadb(documents, reset=reset)

    print("\n" + "=" * 50)
    print("✅ INGESTION COMPLETE!")
    print(f"   File: {file_name}")
    print(f"   Chunks saved: {len(documents)}")
    print("=" * 50 + "\n")

    return vectorstore


# ── TEST BLOCK ────────────────────────────────────────────
# This only runs when you directly run this file
# It does NOT run when app.py imports it

if __name__ == "__main__":

    # change this to your actual PDF path
    TEST_PDF = "sample_docs/test.pdf"

    # check the file exists
    if not os.path.exists(TEST_PDF):
        print(f"❌ File not found: {TEST_PDF}")
        print("Please put a PDF named 'test.pdf' in sample_docs/ folder")
    else:
        # run the full ingestion
        ingest_document(TEST_PDF, reset=True)

        # verify data was saved
        if vectorstore_exists():
            print("🎉 ChromaDB has data!.")
        else:
            print("❌ Something went wrong. ChromaDB is empty.")
