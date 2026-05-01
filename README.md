# RAG Insight Engine

A production-style Retrieval-Augmented Generation application that allows users to upload PDF documents and ask questions using text or voice.

## Features
- PDF upload and text extraction
- NLP-based text cleaning
- Document chunking
- BGE embeddings
- FAISS vector search
- Gemini-powered answer generation
- Voice question input
- AI voice answer
- Chat history
- Retrieval similarity scores
- Confidence score

## Tech Stack
Python, Streamlit, LangChain, FAISS, Gemini API, HuggingFace Embeddings, PyMuPDF, gTTS

## Architecture
PDF Upload → Text Extraction → NLP Cleaning → Chunking → Embeddings → FAISS → Retrieval → Gemini → Answer

## How to Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py