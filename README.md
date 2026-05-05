# 🧠 RAG Insight Engine

A production-style Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions using natural language or voice input.

🔗 **Live App:**  
https://rag-insight-engine-qxxt5cypf8jneqt9diu6ve.streamlit.app/

---

## 🚀 Project Overview

RAG Insight Engine is an AI-powered document intelligence system built to extract information from PDF documents and generate context-aware answers.

The application follows a Retrieval-Augmented Generation workflow:

## ✨ Features

- PDF upload and text extraction
- Text cleaning and preprocessing
- Document chunking
- HuggingFace embedding generation
- FAISS vector search
- Gemini-powered answer generation
- Voice question input
- AI voice response
- Chat history
- Retrieval similarity scores
- Confidence score

## 🧱 Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Gemini API
- HuggingFace Embeddings
- PyMuPDF
- gTTS
- Speech Recognition

## 🏗️ Architecture

```text
PDF Upload
      ↓
Text Extraction
      ↓
Text Cleaning
      ↓
Document Chunking
      ↓
Embedding Generation
      ↓
FAISS Vector Store
      ↓
Similarity Search
      ↓
Gemini Answer Generation
      ↓
Streamlit UI Response ```

## ▶️ How to Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
