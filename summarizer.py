# summarizer.py
# PURPOSE: Generate structured summaries of academic documents
#          using Llama3.2 locally via Ollama
#
# BEGINNER NOTE: Unlike rag_pipeline.py which searches ChromaDB,
#                this file sends text DIRECTLY to the LLM.
#                Best for summarization tasks.


# ── IMPORTS ──────────────────────────────────────────────
# summarizer.py — updated for cloud + local

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pdfplumber
import os
from config import USE_GROQ, GROQ_API_KEY

if USE_GROQ:
    from langchain_groq import ChatGroq
else:
    from langchain_ollama import ChatOllama

LLM_MODEL = "llama3.2"
GROQ_LLM = "llama3-8b-8192"
TEMPERATURE_FACTS = 0.1
TEMPERATURE_CLASS = 0.0
MAX_CHARS = 6000


def get_llm(temperature=0.1):
    if USE_GROQ:
        return ChatGroq(model=GROQ_LLM, temperature=temperature, api_key=GROQ_API_KEY)
    else:
        return ChatOllama(
            model=LLM_MODEL,
            temperature=temperature,
            base_url="http://localhost:11434",
            num_ctx=4096,
        )


# ── PROMPT TEMPLATES ──────────────────────────────────────

# For detecting document type
CLASSIFY_PROMPT = """You are a document classifier.
Read the text below and respond with ONLY one of these labels:
SYLLABUS, NOTICE, RESEARCH PAPER, TEXTBOOK, REPORT, OTHER

Do not explain. Just output the single label word.

TEXT:
{text}

LABEL:"""


# For full structured summary
SUMMARY_PROMPT = """You are EduMind, an AI academic assistant.
Read the following academic document text carefully.

Generate a structured summary with these exact sections:

📚 DOCUMENT TYPE:
[What kind of document this is]

📌 MAIN TOPIC:
[What this document is about in 1-2 sentences]

🔑 KEY POINTS:
- [Point 1]
- [Point 2]
- [Point 3]
- [Point 4]
- [Point 5]
(list up to 7 most important points)

📅 IMPORTANT DATES:
[Any dates or deadlines mentioned, or "None found"]

✅ ACTION ITEMS FOR STUDENT:
- [What the student needs to do based on this document]

⚠️ CRITICAL INFORMATION:
[Anything the student absolutely must not miss]

Use ONLY information from the text below. Be specific.

DOCUMENT TEXT:
{text}

STRUCTURED SUMMARY:"""


# For extracting clean bullet points only
KEYPOINTS_PROMPT = """Read this academic document and extract
the 5 most important key points a student must know.

Format your response as a numbered list ONLY:
1. [key point]
2. [key point]
3. [key point]
4. [key point]
5. [key point]

Be specific. Use only information from the document.
No introduction, no conclusion — just the numbered list.

DOCUMENT TEXT:
{text}

KEY POINTS:"""


# ── FUNCTION 1: Read PDF Text ─────────────────────────────


def extract_text_from_pdf(file_path):
    """
    Reads a PDF and returns its full text.
    Same as ingest.py but returns raw string directly.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"

    if not text.strip():
        raise ValueError("No text found in PDF.")

    return text


# ── FUNCTION 2: Truncate Text Safely ─────────────────────


def truncate_text(text, max_chars=MAX_CHARS):
    """
    Truncates text to max_chars to avoid overwhelming the LLM.
    Tries to cut at a sentence boundary for cleaner input.
    """

    if len(text) <= max_chars:
        return text

    # cut at max_chars
    truncated = text[:max_chars]

    # find last period to cut cleanly at sentence end
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:  # only if near the end
        truncated = truncated[: last_period + 1]

    return truncated + "\n\n[Document continues...]"


# ── FUNCTION 3: Classify Document Type ───────────────────


def classify_document(text):
    """
    Detects what TYPE of document this is.
    Returns one of: SYLLABUS, NOTICE, RESEARCH PAPER,
                    TEXTBOOK, REPORT, OTHER
    """

    print("🔍 Detecting document type...")

    llm = get_llm(temperature=TEMPERATURE_CLASS)
    parser = StrOutputParser()

    prompt = PromptTemplate(template=CLASSIFY_PROMPT, input_variables=["text"])

    chain = prompt | llm | parser

    # use first 2000 chars for classification (faster)
    short_text = text[:2000]

    doc_type = chain.invoke({"text": short_text})
    doc_type = doc_type.strip().upper()

    # validate — if LLM returned something unexpected
    valid_types = [
        "SYLLABUS",
        "NOTICE",
        "RESEARCH PAPER",
        "TEXTBOOK",
        "REPORT",
        "OTHER",
    ]
    if doc_type not in valid_types:
        doc_type = "OTHER"

    print(f"✅ Document type: {doc_type}")
    return doc_type


# ── FUNCTION 4: Generate Full Summary ────────────────────


def generate_summary(text):
    """
    Generates a complete structured academic summary.
    Returns formatted string with all sections.
    """

    print("📋 Generating structured summary...")
    print("   (Takes 30-90 seconds — Llama3.2 is reading...)")

    llm = get_llm(temperature=TEMPERATURE_FACTS)
    parser = StrOutputParser()

    prompt = PromptTemplate(template=SUMMARY_PROMPT, input_variables=["text"])

    chain = prompt | llm | parser

    # truncate safely for M1
    safe_text = truncate_text(text)

    summary = chain.invoke({"text": safe_text})

    print("✅ Summary generated!")
    return summary.strip()


# ── FUNCTION 5: Extract Key Points Only ──────────────────


def extract_key_points(text):
    """
    Extracts ONLY the 5 key points as a clean numbered list.
    Used for the quick-view panel in the UI.
    """

    print("🔑 Extracting key points...")

    llm = get_llm(temperature=TEMPERATURE_FACTS)
    parser = StrOutputParser()

    prompt = PromptTemplate(template=KEYPOINTS_PROMPT, input_variables=["text"])

    chain = prompt | llm | parser

    safe_text = truncate_text(text, max_chars=4000)

    key_points_text = chain.invoke({"text": safe_text})

    # parse into a clean Python list
    lines = key_points_text.strip().split("\n")
    key_points = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # remove numbering like "1." "2." etc
        if line[0].isdigit() and len(line) > 2:
            cleaned = line[2:].strip()  # remove "1."
            if cleaned:
                key_points.append(cleaned)
        elif line.startswith("•"):
            cleaned = line[1:].strip()
            if cleaned:
                key_points.append(cleaned)

    print(f"✅ Extracted {len(key_points)} key points")
    return key_points


# ── FUNCTION 6: MASTER — Full Analysis ───────────────────


def analyze_document(text):
    """
    MASTER FUNCTION — runs all summarization steps.
    Returns a complete analysis dict.

    Returns:
    {
        "doc_type"   : "SYLLABUS",
        "summary"    : "full structured summary text",
        "key_points" : ["point1", "point2", ...],
        "char_count" : 24853,
        "word_count" : 4120
    }
    """

    print("\n" + "=" * 50)
    print("🚀 EduMind Document Analysis Started")
    print("=" * 50)

    # word + char counts
    char_count = len(text)
    word_count = len(text.split())

    print(f"📊 Document size: {word_count} words, {char_count} chars")

    # Step 1: classify
    doc_type = classify_document(text)

    # Step 2: full summary
    summary = generate_summary(text)

    # Step 3: key points
    key_points = extract_key_points(text)

    result = {
        "doc_type": doc_type,
        "summary": summary,
        "key_points": key_points,
        "char_count": char_count,
        "word_count": word_count,
    }

    print("\n" + "=" * 50)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 50 + "\n")

    return result


# ── FUNCTION 7: Analyze From PDF Path ────────────────────


def analyze_pdf(file_path):
    """
    Convenience function — takes PDF path, does everything.
    Reads PDF → analyzes → returns result dict.
    """

    text = extract_text_from_pdf(file_path)
    return analyze_document(text)


# ── TEST BLOCK ────────────────────────────────────────────

if __name__ == "__main__":

    # Test with sample academic text first
    sample_text = """
    COURSE SYLLABUS — Artificial Intelligence & Machine Learning
    Galgotias University | Semester 4 | 2025-2026

    COURSE OVERVIEW:
    This course introduces students to fundamental concepts of
    Artificial Intelligence and Machine Learning. Students will
    learn supervised learning, unsupervised learning, neural
    networks, and generative AI systems.

    TOPICS COVERED:
    Unit 1: Introduction to AI, history, applications
    Unit 2: Machine Learning — regression, classification
    Unit 3: Deep Learning — neural networks, CNNs, RNNs
    Unit 4: Natural Language Processing
    Unit 5: Generative AI and Large Language Models

    ASSESSMENT PATTERN:
    Mid Term Exam     : 30 marks
    End Term Exam     : 50 marks
    Assignments (x3)  : 15 marks
    Project           : 05 marks

    IMPORTANT DATES:
    Assignment 1 due  : April 10, 2026
    Mid Term Exam     : April 20, 2026
    Assignment 2 due  : May 5, 2026
    Project Submission: May 25, 2026
    End Term Exam     : June 15, 2026

    TEXTBOOKS:
    1. Artificial Intelligence — Stuart Russell & Peter Norvig
    2. Deep Learning — Ian Goodfellow

    Students must attend minimum 75% classes to appear in exams.
    """

    print("\n" + "=" * 50)
    print("🧪 Testing Summarizer")
    print("=" * 50)

    # run full analysis
    result = analyze_document(sample_text)

    # display results
    print("\n📄 DOCUMENT TYPE:", result["doc_type"])
    print(f"📊 Size: {result['word_count']} words\n")

    print("─" * 50)
    print("📋 FULL SUMMARY:")
    print("─" * 50)
    print(result["summary"])

    print("\n" + "─" * 50)
    print("🔑 KEY POINTS:")
    print("─" * 50)
    for i, point in enumerate(result["key_points"], 1):
        print(f"  {i}. {point}")

    print("\n✅ Summarizer Test Complete!")

    # if you have a real PDF, test it too
    test_pdf = "sample_docs/test.pdf"
    if os.path.exists(test_pdf):
        print(f"\n🔄 Also testing with real PDF: {test_pdf}")
        pdf_result = analyze_pdf(test_pdf)
        print(f"\n📄 Your PDF type: {pdf_result['doc_type']}")
        print(f"📊 Size: {pdf_result['word_count']} words")
        print("\n🔑 Key Points from your PDF:")
        for i, point in enumerate(pdf_result["key_points"], 1):
            print(f"  {i}. {point}")

    print("\n🎉 Ready\n")
