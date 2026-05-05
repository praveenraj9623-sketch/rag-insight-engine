import streamlit as st
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import tempfile

from src.pdf_loader import extract_text_from_pdf
from src.nlp_processor import clean_text, get_document_stats
from src.chunker import create_chunks
from src.vector_db import create_faiss_vector_store, save_vector_store
from src.rag_chain import answer_question

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Insight Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg: #f4f6fa;
    --surface: #ffffff;
    --surface-2: #f0f2f7;
    --border: #e2e5ed;
    --border-strong: #cdd1db;
    --text: #111827;
    --text-2: #4b5563;
    --text-3: #9ca3af;
    --blue: #1d4ed8;
    --blue-bg: #eff3ff;
    --blue-border: #bfccf9;
    --blue-text: #1d4ed8;
    --green: #15803d;
    --green-bg: #f0fdf4;
    --green-border: #86efac;
    --green-text: #15803d;
    --amber-bg: #fffbeb;
    --amber-border: #fcd34d;
    --amber-text: #92400e;
    --red-bg: #fff1f2;
    --red-border: #fca5a5;
    --red-text: #b91c1c;
    --radius: 10px;
    --radius-lg: 14px;
    --radius-xl: 20px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide default Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ─────────────────────────────────────────
   SIDEBAR FIX — keep toggle always visible
───────────────────────────────────────── */
[data-testid="collapsedControl"] {
    display:       flex       !important;
    visibility:    visible    !important;
    opacity:       1          !important;
    z-index:       9999       !important;
    background:    #ffffff    !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow:    2px 2px 8px rgba(0,0,0,0.10) !important;
    padding:       12px 8px   !important;
    position:      fixed      !important;
    top:           50vh       !important;
    left:          0          !important;
    transform:     translateY(-50%) !important;
    cursor:        pointer    !important;
}
[data-testid="collapsedControl"]:hover {
    background: var(--blue-bg) !important;
}
[data-testid="collapsedControl"] svg {
    fill:   #111827 !important;
    width:  18px    !important;
    height: 18px    !important;
}

/* Sidebar panel */
section[data-testid="stSidebar"] {
    background:   #ffffff !important;
    border-right: 1px solid #e2e5ed !important;
    min-width:    230px   !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem !important;
}

/* Main content area */
.block-container {
    padding:   1.5rem 2rem 2rem !important;
    max-width: 960px             !important;
}

/* ── Hero ── */
.hero {
    background:    linear-gradient(130deg, #0f172a 0%, #1e3a8a 100%);
    border-radius: var(--radius-xl);
    padding:       26px 28px;
    display:       flex;
    align-items:   center;
    justify-content: space-between;
    gap:           16px;
    margin-bottom: 20px;
}
.hero-title { font-size: 22px; font-weight: 600; color: #fff; margin-bottom: 6px; }
.hero-sub   { font-size: 13px; color: #93c5fd; line-height: 1.6; }
.hero-badge {
    background:    rgba(255,255,255,0.08);
    border:        1px solid rgba(255,255,255,0.18);
    border-radius: 999px;
    padding:       5px 16px;
    font-size:     11px;
    color:         #bfdbfe;
    font-weight:   500;
    white-space:   nowrap;
    flex-shrink:   0;
}

/* ── Stat grid ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0,1fr));
    gap: 12px;
    margin: 16px 0;
}
.stat-card {
    background:    var(--surface);
    border:        1px solid var(--border);
    border-radius: var(--radius-lg);
    padding:       16px 14px;
    text-align:    center;
}
.stat-label { font-size: 11px; color: var(--text-3); margin-bottom: 6px; }
.stat-value {
    font-size:   24px;
    font-weight: 600;
    color:       var(--text);
    font-family: 'DM Mono', monospace;
}

/* ── Banners ── */
.banner {
    border-radius: var(--radius);
    padding:       10px 14px;
    font-size:     13px;
    display:       flex;
    align-items:   center;
    gap:           8px;
    margin-bottom: 14px;
}
.banner-success { background: var(--green-bg); border: 1px solid var(--green-border); color: var(--green-text); }
.banner-info    { background: var(--blue-bg);  border: 1px solid var(--blue-border);  color: var(--blue-text); }
.banner-warn    { background: var(--amber-bg); border: 1px solid var(--amber-border); color: var(--amber-text); }
.banner-error   { background: var(--red-bg);   border: 1px solid var(--red-border);   color: var(--red-text); }

/* ── Answer box ── */
.answer-box {
    background:    var(--surface);
    border:        1px solid var(--border);
    border-left:   4px solid var(--blue);
    border-radius: var(--radius);
    padding:       16px 18px;
    font-size:     13.5px;
    line-height:   1.8;
    color:         var(--text);
    margin:        12px 0;
}

/* ── Confidence badge ── */
.conf-badge {
    display:        inline-block;
    font-size:      11px;
    font-weight:    500;
    padding:        3px 12px;
    border-radius:  999px;
    border:         1px solid;
    margin-left:    10px;
    vertical-align: middle;
}
.conf-high { background: var(--green-bg); border-color: var(--green-border); color: var(--green-text); }
.conf-med  { background: var(--amber-bg); border-color: var(--amber-border); color: var(--amber-text); }
.conf-low  { background: var(--red-bg);   border-color: var(--red-border);   color: var(--red-text); }

/* ── Source chunk cards ── */
.chunk-card {
    border:        1px solid var(--border);
    border-radius: var(--radius);
    overflow:      hidden;
    margin-bottom: 8px;
    background:    var(--surface);
}
.chunk-header {
    background:    var(--surface-2);
    padding:       8px 14px;
    display:       flex;
    justify-content: space-between;
    align-items:   center;
    border-bottom: 1px solid var(--border);
}
.chunk-title { font-size: 12px; font-weight: 500; color: var(--text); }
.score-badge {
    font-size:     11px;
    padding:       2px 9px;
    border-radius: var(--radius);
    border:        1px solid;
    font-family:   'DM Mono', monospace;
}
.score-high { background: var(--green-bg); border-color: var(--green-border); color: var(--green-text); }
.score-med  { background: var(--amber-bg); border-color: var(--amber-border); color: var(--amber-text); }
.score-low  { background: var(--red-bg);   border-color: var(--red-border);   color: var(--red-text); }
.chunk-body { padding: 10px 14px; font-size: 12.5px; color: var(--text-2); line-height: 1.65; }
.chunk-meta { padding: 4px 14px 10px; font-size: 11px; color: var(--text-3); font-family: 'DM Mono', monospace; }

/* ── Chat history ── */
.chat-bubble {
    background:    var(--surface);
    border:        1px solid var(--border);
    border-radius: var(--radius-lg);
    padding:       14px 16px;
    margin-bottom: 10px;
}
.chat-you  { font-size: 13px; font-weight: 600; color: var(--blue-text); margin-bottom: 5px; }
.chat-ai   { font-size: 13px; color: var(--text); line-height: 1.65; margin-bottom: 6px; }
.chat-meta { font-size: 11px; color: var(--text-3); }

/* ── Section header ── */
.section-header {
    font-size:   14px;
    font-weight: 600;
    color:       var(--text);
    margin:      18px 0 10px;
    display:     flex;
    align-items: center;
    gap:         8px;
}

/* ── Sidebar labels ── */
.sidebar-label {
    font-size:      10px;
    font-weight:    600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color:          var(--text-3);
    margin-bottom:  8px;
}
.feature-item {
    display:       flex;
    align-items:   center;
    gap:           8px;
    font-size:     12.5px;
    color:         var(--text-2);
    margin-bottom: 6px;
}
.feature-dot {
    width:         6px;
    height:        6px;
    border-radius: 50%;
    background:    var(--green);
    flex-shrink:   0;
    display:       inline-block;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid var(--border); margin: 18px 0; }

/* ── Streamlit overrides ── */
.stTextInput > div > div > input {
    border-radius: var(--radius)         !important;
    border:        1px solid var(--border-strong) !important;
    font-family:   'DM Sans', sans-serif !important;
    font-size:     13px                  !important;
    padding:       9px 13px              !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--blue)     !important;
    box-shadow:   0 0 0 2px var(--blue-bg) !important;
}
.stButton > button {
    border-radius: var(--radius)         !important;
    font-family:   'DM Sans', sans-serif !important;
    font-weight:   500                   !important;
    font-size:     13px                  !important;
    border:        1px solid var(--border-strong) !important;
    background:    var(--surface)        !important;
    color:         var(--text)           !important;
    transition:    background 0.15s      !important;
}
.stButton > button:hover { background: var(--surface-2) !important; }
.stFileUploader {
    border:        1.5px dashed var(--border-strong) !important;
    border-radius: var(--radius-lg) !important;
    background:    var(--surface)   !important;
    padding:       12px             !important;
}
.stExpander {
    border:        1px solid var(--border) !important;
    border-radius: var(--radius)           !important;
    background:    var(--surface)          !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Cached vector store ───────────────────────────────────────────────────────
@st.cache_resource
def build_vector_store(documents):
    return create_faiss_vector_store(documents)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("---")

    st.markdown('<div class="sidebar-label">Features</div>', unsafe_allow_html=True)
    features = [
        "PDF Upload", "NLP Cleaning", "FAISS Vector Search",
        "LLM Answering", "Voice Input", "Voice Output",
        "Chat History", "Retrieval Scores"
    ]
    items_html = "".join(
        f'<div class="feature-item"><span class="feature-dot"></span>{f}</div>'
        for f in features
    )
    st.markdown(items_html, unsafe_allow_html=True)
    st.markdown("---")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div>
        <div class="hero-title">🤖 RAG Insight Engine</div>
        <div class="hero-sub">
            Upload a PDF, ask questions using text or voice,<br>
            and get grounded AI answers with source context.
        </div>
    </div>
    <div class="hero-badge">FAISS · LLM · RAG</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PDF UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "📄 Upload your PDF document",
    type=["pdf"],
    help="Upload a resume, report, policy document, or any PDF."
)

if uploaded_file is not None:

    with st.spinner("Reading and processing your PDF…"):
        raw_text = extract_text_from_pdf(uploaded_file)

    if not raw_text.strip():
        st.markdown(
            '<div class="banner banner-error">⚠️ No readable text found in this PDF.</div>',
            unsafe_allow_html=True
        )
        st.stop()

    cleaned_text = clean_text(raw_text)
    stats        = get_document_stats(cleaned_text)

    st.markdown(
        '<div class="banner banner-success">✅ Document processed successfully.</div>',
        unsafe_allow_html=True
    )

    # ── Stat cards ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">📝 Words</div>
            <div class="stat-value">{stats["total_words"]:,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">📌 Sentences</div>
            <div class="stat-value">{stats["total_sentences"]:,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">🔤 Characters</div>
            <div class="stat-value">{stats["total_characters"]:,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">📦 File type</div>
            <div class="stat-value">PDF</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Preview ───────────────────────────────────────────────────────────────
    with st.expander("👀 Preview extracted document text"):
        st.write(cleaned_text[:4000])

    # ── Chunking ──────────────────────────────────────────────────────────────
    with st.spinner("Creating document chunks…"):
        documents = create_chunks(cleaned_text)

    st.markdown(
        f'<div class="banner banner-info">ℹ️ Created <strong>{len(documents)}</strong> chunks for retrieval.</div>',
        unsafe_allow_html=True
    )

    # ── Vector store ──────────────────────────────────────────────────────────
    with st.spinner("Building FAISS vector database…"):
        vector_store = build_vector_store(documents)
        save_vector_store(vector_store)

    st.markdown(
        '<div class="banner banner-success">✅ Vector database ready.</div>',
        unsafe_allow_html=True
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Q&A
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">💬 Ask Your Question</div>', unsafe_allow_html=True)

    input_col, voice_col = st.columns([3, 1])

    with input_col:
        typed_question = st.text_input(
            "Type your question",
            placeholder="e.g. What are the key skills mentioned in this document?",
            label_visibility="collapsed"
        )

    with voice_col:
        voice_question = speech_to_text(
            language="en",
            start_prompt="🎙 Speak",
            stop_prompt="⏹ Stop",
            just_once=True,
            use_container_width=True,
            key="voice_input"
        )

    question = typed_question

    if voice_question:
        st.markdown(
            f'<div class="banner banner-success">🎤 Voice detected: <em>{voice_question}</em></div>',
            unsafe_allow_html=True
        )
        question = voice_question

    # ── RAG pipeline ──────────────────────────────────────────────────────────
    if question:
        with st.spinner("Retrieving context and generating answer…"):
            result = answer_question(vector_store, question)

        st.session_state.chat_history.append({
            "question":   question,
            "answer":     result["answer"],
            "sources":    result["source_documents"],
            "confidence": result.get("confidence", "Unknown")
        })

        confidence = result.get("confidence", "Unknown")
        conf_class = {"High": "conf-high", "Medium": "conf-med"}.get(confidence, "conf-low")

        # Answer
        st.markdown(
            f'<div class="section-header">✅ Latest Answer'
            f'<span class="conf-badge {conf_class}">{confidence} confidence</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True
        )

        # Voice output
        with st.spinner("Generating voice response…"):
            tts        = gTTS(text=result["answer"], lang="en")
            audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(audio_file.name)

        st.markdown('<div class="section-header">🔊 Voice Answer</div>', unsafe_allow_html=True)
        st.audio(audio_file.name, format="audio/mp3")

        # Retrieval chunks
        st.markdown(
            '<div class="section-header">📈 Retrieval Scores + Source Context</div>',
            unsafe_allow_html=True
        )

        for i, item in enumerate(result["source_documents"], start=1):
            doc, score = item
            if score < 0.22:
                score_class = "score-high"
            elif score < 0.28:
                score_class = "score-med"
            else:
                score_class = "score-low"

            st.markdown(f"""
            <div class="chunk-card">
                <div class="chunk-header">
                    <span class="chunk-title">Source Chunk {i}</span>
                    <span class="score-badge {score_class}">dist: {score:.4f}</span>
                </div>
                <div class="chunk-body">{doc.page_content}</div>
                <div class="chunk-meta">Metadata: {doc.metadata}</div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # CHAT HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🧠 Chat History</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history):
            conf       = chat.get("confidence", "Unknown")
            conf_class = {"High": "conf-high", "Medium": "conf-med"}.get(conf, "conf-low")
            st.markdown(f"""
            <div class="chat-bubble">
                <div class="chat-you">You: {chat["question"]}</div>
                <div class="chat-ai">AI: {chat["answer"]}</div>
                <div class="chat-meta">
                    Confidence: <span class="conf-badge {conf_class}"
                    style="font-size:10px;padding:1px 8px;">{conf}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="banner banner-info">No chat history yet.</div>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(
        '<div class="banner banner-warn">⚠️ Upload a PDF document above to start asking questions.</div>',
        unsafe_allow_html=True
    )