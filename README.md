
# 🧠 RAG Insight Engine

A production-style Retrieval-Augmented Generation application that allows users to upload PDF documents and ask questions using text or voice.

## 🚀 Project Overview

RAG Insight Engine is an AI-powered document intelligence system. It extracts text from uploaded PDF files, chunks the content, converts the chunks into vector embeddings, retrieves the most relevant context, and generates answers using Gemini.

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
Streamlit UI Response```

## ▶️ How to Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py