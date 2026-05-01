import streamlit as st
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import tempfile

from src.pdf_loader import extract_text_from_pdf
from src.nlp_processor import clean_text, get_document_stats
from src.chunker import create_chunks
from src.vector_db import create_faiss_vector_store, save_vector_store
from src.rag_chain import answer_question

st.set_page_config(
    page_title="RAG Insight Engine",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero-card {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 32px;
    border-radius: 24px;
    color: white;
    margin-bottom: 24px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 17px;
    color: #dbeafe;
}

.feature-card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.06);
    border: 1px solid #e5e7eb;
}

.answer-card {
    background: white;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.07);
    border-left: 6px solid #2563eb;
}

.chat-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 12px;
    border: 1px solid #e5e7eb;
}

.user-msg {
    color: #1e40af;
    font-weight: 700;
}

.ai-msg {
    color: #111827;
}

.small-muted {
    color: #6b7280;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def build_vector_store(documents):
    return create_faiss_vector_store(documents)

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("---")
    st.markdown("### Features")
    st.markdown("""
    ✅ PDF Upload  
    ✅ NLP Cleaning  
    ✅ FAISS Vector Search  
    ✅ LLM Answering  
    ✅ Voice Input  
    ✅ Voice Output  
    ✅ Chat History  
    ✅ Retrieval Scores  
    """)
    st.markdown("---")

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("""
<div class="hero-card">
    <div class="hero-title">🤖 RAG Insight Engine</div>
    <div class="hero-subtitle">
        Upload a PDF, ask questions using text or voice, and get grounded AI answers with source context.
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📄 Upload your PDF document",
    type=["pdf"],
    help="Upload resume, report, policy document, project report, or any PDF."
)

if uploaded_file is not None:
    with st.spinner("Reading and processing your PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    if not raw_text.strip():
        st.error("No readable text found in this PDF.")
        st.stop()

    cleaned_text = clean_text(raw_text)
    stats = get_document_stats(cleaned_text)

    st.success("Document processed successfully.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="feature-card">
                <h4>📝 Words</h4>
                <h2>{stats["total_words"]}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="feature-card">
                <h4>📌 Sentences</h4>
                <h2>{stats["total_sentences"]}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="feature-card">
                <h4>🔤 Characters</h4>
                <h2>{stats["total_characters"]}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="feature-card">
                <h4>📦 File</h4>
                <h2>PDF</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    with st.expander("👀 Preview Extracted Document Text"):
        st.write(cleaned_text[:4000])

    with st.spinner("Creating document chunks..."):
        documents = create_chunks(cleaned_text)

    st.info(f"Created {len(documents)} chunks for retrieval.")

    with st.spinner("Building FAISS vector database..."):
        vector_store = build_vector_store(documents)
        save_vector_store(vector_store)

    st.success("Vector database ready.")

    st.markdown("## 💬 Ask Your Question")

    input_col, voice_col = st.columns([2, 1])

    with input_col:
        typed_question = st.text_input(
            "Type your question",
            placeholder="Example: What are the key skills mentioned in this document?"
        )

    with voice_col:
        voice_question = speech_to_text(
            language="en",
            start_prompt="🎙️ Speak",
            stop_prompt="⏹️ Stop",
            just_once=True,
            use_container_width=True,
            key="voice_input"
        )

    question = typed_question

    if voice_question:
        st.success(f"Voice detected: {voice_question}")
        question = voice_question

    if question:
        with st.spinner("Retrieving relevant context and generating answer..."):
            result = answer_question(vector_store, question)

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": result["answer"],
                "sources": result["source_documents"],
                "confidence": result.get("confidence", "Unknown")
            }
        )

        confidence = result.get("confidence", "Unknown")

        st.markdown("## ✅ Latest Answer")

        if confidence == "High":
            st.success(f"Confidence: {confidence}")
        elif confidence == "Medium":
            st.warning(f"Confidence: {confidence}")
        else:
            st.error(f"Confidence: {confidence}")

        st.markdown(
            f"""
            <div class="answer-card">
                <p>{result["answer"]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.spinner("Generating voice response..."):
            tts = gTTS(text=result["answer"], lang="en")
            audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(audio_file.name)

        st.markdown("### 🔊 Voice Answer")
        st.audio(audio_file.name, format="audio/mp3")

        st.markdown("### 📈 Retrieval Scores + Source Context")

        for i, item in enumerate(result["source_documents"], start=1):
            doc, score = item
            with st.expander(f"Source Chunk {i} | Distance Score: {score:.4f}"):
                st.write(doc.page_content)
                st.caption(f"Metadata: {doc.metadata}")

    st.markdown("---")
    st.markdown("## 🧠 Chat History")

    if st.session_state.chat_history:
        for index, chat in enumerate(reversed(st.session_state.chat_history), start=1):
            st.markdown(
                f"""
                <div class="chat-card">
                    <p class="user-msg">You: {chat["question"]}</p>
                    <p class="ai-msg">AI: {chat["answer"]}</p>
                    <p class="small-muted">Confidence: {chat.get("confidence", "Unknown")}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No chat history yet.")

else:
    st.warning("Upload a PDF document to start.")