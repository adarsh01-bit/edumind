# 🧠 EduMind — AI Academic Assistant

> Upload any academic PDF and chat with it using AI.
> Get summaries, find deadlines, and ask questions — 
> fully offline or deployed on the cloud.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Live Demo
👉 **[Try EduMind Here](YOUR_STREAMLIT_URL_HERE)**

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
git clone https://github.com/YOUR_USERNAME/edumind.git
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
GROQ_API_KEY=your_groq_api_key_here
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