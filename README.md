# 🧠 EduMind — AI Academic Assistant
 > Built a full-stack RAG system with real-time document Q&A, semantic search, and NLP-powered deadline extraction.  
 > An AI-powered academic assistant that lets students chat with PDFs, extract key insights, detect deadlines, and generate summaries using RAG.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Live Demo
👉 **[Try EduMind Here](https://edumind-wephtnkbqfj9mb9mtjqbfn.streamlit.app/)**
---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 Chat with PDF | Ask questions in natural language |
| 📋 Smart Summary | Structured academic summary + key points |
| 🗓️ Deadline Detection | Auto-finds all dates using NLP |
| 🔒 100% Offline | Local mode runs without internet |
| ☁️ Cloud Mode | Deployed via Streamlit + Groq |

---

## 🧪 Example Use Cases

- Analyze university syllabus PDFs
- Extract assignment deadlines automatically
- Generate quick revision summaries
- Ask questions from research papers 

--- 

## ⚙️ How It Works

1. 📄 **PDF Upload**
   - User uploads an academic document

2. ✂️ **Text Processing**
   - PDF text is extracted and split into smaller chunks

3. 🧠 **Embedding Generation**
   - Each chunk is converted into vector embeddings

4. 🗄️ **Vector Storage**
   - Embeddings are stored in ChromaDB for fast retrieval

5. ❓ **User Query**
   - User asks a question in natural language

6. 🔍 **Semantic Search**
   - System finds the most relevant chunks using similarity search

7. 🤖 **AI Response Generation**
   - LLM (Groq / Ollama) generates a context-aware answer

8. 📊 **Extra Features**
   - spaCy detects deadlines
   - Summarizer generates structured insights

## 🛠️ Tech Stack

- **LLM**: Llama3.2 (local via Ollama) / Llama3-8b (cloud via Groq)
- **RAG**: LangChain + ChromaDB
- **Embeddings**: nomic-embed-text / all-MiniLM-L6-v2
- **NLP**: spaCy (deadline detection)
- **UI**: Streamlit
- **Language**: Python 3.11

---

## 📦 Local Installation
```bash
# Clone the repo
git clone https://github.com/adarsh01-bit/edumind.git
cd edumind

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

# Install Ollama + pull models
# Download Ollama from https://ollama.com
ollama pull llama3.2
ollama pull nomic-embed-text

# Run the app
streamlit run app.py
```

---

## 🔧 Environment Variables

Create a `.env` file for cloud deployment:
```
GROQ_API_KEY=groq_api_key_here
```

---

## 📁 Project Structure
```
edumind/
├── app.py                  ← Streamlit UI
├── ingest.py               ← PDF → ChromaDB pipeline
├── rag_pipeline.py         ← RAG Q&A engine
├── deadline_detector.py    ← NLP date extraction
├── summarizer.py           ← Document summarization
├── config.py               ← Local/Cloud switcher
└── requirements.txt
```

---

## 🎓 Built By

**Adarsh Kumar** — 2nd Year CSE (AI-ML), Galgotias University

Certifications:
- STTP: Introduction to ML, DL & GenAI — Galgotias University
- AI-ML Virtual Internship (Grade O) — AICTE + Google for Developers

---

## 📄 License
MIT License 