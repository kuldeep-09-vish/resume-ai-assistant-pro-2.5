from dotenv import load_dotenv
import os
import glob
import streamlit as st
import tempfile
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------------------------
# LOAD ENV (LOCAL ONLY)
# ----------------------------------------------------
load_dotenv()

PDF_FOLDER = "pdf"

# Ensure PDF folder exists
os.makedirs(PDF_FOLDER, exist_ok=True)

# Modern theme colors - Dark Professional Theme
ACCENTS = [
    {"from": "#6366F1", "to": "#818CF8", "soft": "rgba(99,102,241,0.16)"},   # indigo
    {"from": "#06B6D4", "to": "#22D3EE", "soft": "rgba(6,182,212,0.16)"},    # cyan
    {"from": "#F43F5E", "to": "#FB7185", "soft": "rgba(244,63,94,0.16)"},    # rose
    {"from": "#10B981", "to": "#34D399", "soft": "rgba(16,185,129,0.16)"},   # emerald
    {"from": "#F59E0B", "to": "#FBBF24", "soft": "rgba(245,158,11,0.16)"},   # amber
    {"from": "#3B82F6", "to": "#60A5FA", "soft": "rgba(59,130,246,0.16)"},   # blue
]


def accent_for(index: int):
    return ACCENTS[index % len(ACCENTS)]


# ----------------------------------------------------
# SAFE API KEY LOADER (LOCAL + STREAMLIT CLOUD)
# ----------------------------------------------------
def get_api_key():
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.getenv("GOOGLE_API_KEY")


# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Resume AI Assistant",
    page_icon="✨",
    layout="wide",
)

# ----------------------------------------------------
# MODERN DARK THEME WITH GRADIENT ACCENTS
# ----------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #0F1117;
    --surface: rgba(255,255,255,0.03);
    --surface-2: rgba(255,255,255,0.06);
    --surface-solid: #1A1D27;
    --border: rgba(255,255,255,0.08);
    --text: #F1F5F9;
    --text-soft: #94A3B8;
    --text-faint: #64748B;
    --radius: 14px;
    --shadow: 0 8px 32px rgba(0,0,0,0.4);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.stApp {
    background: var(--bg);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 960px;
    margin: 0 auto;
}

/* ---------- Header ---------- */
.header-wrap {
    text-align: center;
    padding: 4px 0 20px 0;
}

.main-title {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 38px;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #818CF8 0%, #22D3EE 50%, #34D399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}

.subtitle {
    color: var(--text-soft);
    font-size: 14px;
    letter-spacing: 0.2px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #0A0D14;
    border-right: 1px solid var(--border);
}

.sidebar-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: var(--shadow);
}

.sidebar-title {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 18px;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}

.sidebar-label {
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 1.5px;
    color: #818CF8;
    font-weight: 700;
    margin-bottom: 8px;
}

.sidebar-text {
    font-size: 13px;
    color: var(--text-soft);
    line-height: 1.6;
}

section[data-testid="stSidebar"] button {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
}

section[data-testid="stSidebar"] button:hover {
    background: linear-gradient(135deg, #6366F1, #06B6D4) !important;
    color: white !important;
    border-color: transparent !important;
    transform: translateX(2px);
}

/* ---------- Content panel ---------- */
div[class*="st-key-content_panel"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
    border-top: 2px solid var(--panel-accent, #6366F1);
}

div[class*="st-key-live_panel"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
    border-top: 2px solid var(--panel-accent, #6366F1);
}

.file-meta {
    font-size: 12px;
    color: var(--text-faint);
    letter-spacing: 0.2px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
}

.file-meta .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--panel-accent, #6366F1);
    box-shadow: 0 0 10px var(--panel-accent, #6366F1);
}

/* ---------- Chat bubbles ---------- */
.chat-row {
    display: flex;
    gap: 10px;
    margin: 14px 0;
    align-items: flex-end;
}

.chat-row.user {
    justify-content: flex-end;
}

.chat-row.assistant {
    justify-content: flex-start;
}

.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 12px;
    flex-shrink: 0;
}

.avatar.assistant {
    background: linear-gradient(135deg, #6366F1, #06B6D4);
    color: white;
}

.avatar.user {
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
}

.bubble {
    max-width: 70%;
    padding: 12px 16px;
    line-height: 1.6;
    font-size: 14px;
}

.bubble.user {
    background: linear-gradient(135deg, #6366F1 0%, #818CF8 100%);
    color: white;
    border-radius: 16px 16px 4px 16px;
    font-weight: 500;
}

.bubble.assistant {
    background: var(--surface-solid);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
}

.stamp-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #22D3EE;
    border: 1px solid rgba(34, 211, 238, 0.4);
    background: rgba(34, 211, 238, 0.08);
    border-radius: 999px;
    padding: 3px 10px;
    margin-bottom: 8px;
}

/* ---------- Empty state ---------- */
.empty-state {
    text-align: center;
    padding: 18px 10px 6px 10px;
}

.empty-state .icon {
    font-size: 32px;
    margin-bottom: 10px;
}

.empty-state .title {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 18px;
    color: var(--text);
    margin-bottom: 4px;
}

.empty-state .sub {
    font-size: 13px;
    color: var(--text-soft);
    margin-bottom: 18px;
}

.stApp button[kind="tertiary"] {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--panel-accent, #6366F1) !important;
    border-radius: 10px !important;
    text-align: left !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 12px 16px !important;
    white-space: normal !important;
    height: auto !important;
    transition: all 0.2s ease !important;
}

.stApp button[kind="tertiary"]:hover {
    background: var(--surface-2) !important;
    border-left-width: 5px !important;
    transform: translateX(2px);
}

/* ---------- Upload area ---------- */
div[data-testid="stFileUploader"] {
    border-radius: 12px !important;
}

div[data-testid="stFileUploader"] section {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface) !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: #6366F1 !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

/* ---------- Popover ---------- */
div[data-testid="stPopover"] button {
    background: linear-gradient(90deg, var(--card-from, #6366F1), var(--card-to, #06B6D4)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    margin-top: 4px;
    margin-bottom: 12px;
}

div[data-testid="stPopover"] button:hover {
    filter: brightness(1.1);
}

div[data-testid="stPopoverBody"] {
    background: #1A1D27 !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: var(--shadow) !important;
    padding: 14px !important;
}

.popover-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-faint);
    margin: 2px 0 10px 4px;
}

/* ---------- File list buttons ---------- */
div[class*="st-key-file_list"] {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 4px;
}

div[class*="st-key-file_list"] button[kind="secondary"] {
    background: var(--surface) !important;
    color: var(--text-soft) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 12px 16px !important;
    justify-content: flex-start !important;
    text-align: left !important;
    white-space: normal !important;
    word-break: break-word !important;
    line-height: 1.35 !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}

div[class*="st-key-file_list"] button[kind="secondary"]:hover {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
    transform: translateX(3px);
}

div[class*="st-key-file_list"] button[kind="primary"] {
    background: linear-gradient(90deg, var(--card-from, #6366F1) 0%, var(--card-to, #06B6D4) 100%) !important;
    color: white !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 12px 16px !important;
    justify-content: flex-start !important;
    text-align: left !important;
    white-space: normal !important;
    word-break: break-word !important;
    line-height: 1.35 !important;
}

/* ---------- Current file pill ---------- */
.current-file-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--surface-2);
    border: 1px solid var(--panel-accent, #6366F1);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    word-break: break-word;
}

.current-file-pill .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--panel-accent, #6366F1);
    box-shadow: 0 0 8px var(--panel-accent, #6366F1);
    flex-shrink: 0;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: var(--text-faint);
    font-size: 12px;
    padding: 16px 0 6px 0;
    letter-spacing: 0.2px;
}

/* ---------- Animations ---------- */
@media (prefers-reduced-motion: no-preference) {
    .chat-row {
        animation: fadeSlideIn 0.25s ease;
    }
}

@keyframes fadeSlideIn {
    from {
        opacity: 0;
        transform: translateY(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 640px) {
    .main-title {
        font-size: 28px;
    }
    .bubble {
        max-width: 85%;
    }
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# DISCOVER PDFs IN FOLDER
# ----------------------------------------------------
def list_pdfs():
    if not os.path.isdir(PDF_FOLDER):
        return []
    files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    return sorted(files)


all_pdfs = list_pdfs()
pdf_names = [os.path.basename(p) for p in all_pdfs]

# ----------------------------------------------------
# SESSION STATE INIT
# ----------------------------------------------------
if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = all_pdfs[0] if all_pdfs else None

if "chat_by_pdf" not in st.session_state:
    st.session_state.chat_by_pdf = {}

if "suggestions_by_pdf" not in st.session_state:
    st.session_state.suggestions_by_pdf = {}

if "uploaded_pdf_path" not in st.session_state:
    st.session_state.uploaded_pdf_path = None

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.markdown("""
<div class="header-wrap">
    <div class="main-title">Resume AI Assistant</div>
    <div class="subtitle">LangChain · Gemini · FAISS</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">✨ Resume AI</div>
        <div class="sidebar-text">Ask questions about any resume. Upload new PDFs or select from existing ones.</div>
    </div>
    """, unsafe_allow_html=True)

    # Upload new PDF
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-label">📤 Upload New Resume</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        key="pdf_uploader",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        temp_path = os.path.join(PDF_FOLDER, uploaded_file.name)
        if not os.path.exists(temp_path):
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            st.cache_resource.clear()
            all_pdfs = list_pdfs()
            pdf_names = [os.path.basename(p) for p in all_pdfs]
            st.session_state.selected_pdf = temp_path
            st.session_state.chat_by_pdf[temp_path] = []
            st.rerun()
        else:
            st.warning(f"⚠️ {uploaded_file.name} already exists")
            existing_path = os.path.join(PDF_FOLDER, uploaded_file.name)
            if st.session_state.selected_pdf != existing_path:
                st.session_state.selected_pdf = existing_path
                st.rerun()

    st.divider()

    if all_pdfs and st.session_state.selected_pdf:
        selected_name = os.path.basename(st.session_state.selected_pdf)
        selected_index = all_pdfs.index(st.session_state.selected_pdf)
        selected_accent = accent_for(selected_index)

        st.markdown(f"""
        <div class="sidebar-card" style="padding-bottom:10px;">
            <div class="sidebar-label">Current file</div>
            <div class="current-file-pill">
                <span class="dot"></span>{selected_name}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Better switch resume - using radio buttons in a scrollable container
        st.markdown('<div class="sidebar-label" style="margin-top:12px;">🔄 Switch Resume</div>', unsafe_allow_html=True)
        
        # Scrollable container for many PDFs
        with st.container(height=250):  # Fixed height with scroll
            pdf_display_names = []
            for name in pdf_names:
                display_name = name[:-4] if name.lower().endswith(".pdf") else name
                pdf_display_names.append(display_name)
            
            for i, (path, display_name) in enumerate(zip(all_pdfs, pdf_display_names)):
                is_active = (path == st.session_state.selected_pdf)
                
                # Custom styled button for each resume
                button_style = "primary" if is_active else "secondary"
                if st.button(
                    display_name,
                    key=f"switch_{i}_{path}",
                    type=button_style,
                    icon="📄" if not is_active else "✅",
                    use_container_width=True,
                ):
                    if not is_active:
                        st.session_state.selected_pdf = path
                        st.rerun()

    elif not all_pdfs:
        st.warning("No PDFs found. Please upload a resume.")
# ----------------------------------------------------
# MAIN CONTENT
# ----------------------------------------------------
if all_pdfs and st.session_state.selected_pdf:
    selected_path = st.session_state.selected_pdf
    selected_name = os.path.basename(selected_path)
    selected_index = all_pdfs.index(selected_path)
    selected_accent = accent_for(selected_index)

    # Push accent colors to CSS
    st.markdown(f"""
    <style>
    :root {{
        --panel-accent: {selected_accent['from']};
        --card-from: {selected_accent['from']};
        --card-to: {selected_accent['to']};
        --card-glow: {selected_accent['soft']};
    }}
    </style>
    """, unsafe_allow_html=True)

    # Content panel
    content_panel = st.container(key="content_panel")
    content_panel.markdown(
        f'<div class="file-meta"><span class="dot"></span>Now viewing · {selected_name}</div>',
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------
    # LOAD VECTOR STORE (cached per PDF path)
    # ----------------------------------------------------
    @st.cache_resource(show_spinner="Indexing PDF...")
    def load_vectorstore(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=get_api_key(),
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore, chunks


    vectorstore, all_chunks = load_vectorstore(selected_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ----------------------------------------------------
    # LLM
    # ----------------------------------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=get_api_key(),
    )

    # ----------------------------------------------------
    # PROMPTS
    # ----------------------------------------------------
    prompt = ChatPromptTemplate.from_template(
        """
    You are a professional Resume Assistant.

    Answer ONLY from the given context.

    If answer is unavailable reply:
    "I don't know based on the resume content."

    Context:
    {context}

    Question:
    {question}
    """
    )

    suggestion_prompt = ChatPromptTemplate.from_template(
        """
    Read the resume content below and generate exactly 4 short, specific
    questions a recruiter might ask about it (skills, projects, education,
    experience, tools). Return ONLY the 4 questions, one per line, no numbering,
    no extra text.

    Resume content:
    {context}
    """
    )


    def get_suggestions(pdf_path, chunks):
        if pdf_path in st.session_state.suggestions_by_pdf:
            return st.session_state.suggestions_by_pdf[pdf_path]

        sample_context = "\n\n".join(c.page_content for c in chunks[:4])
        try:
            formatted = suggestion_prompt.invoke({"context": sample_context})
            result = llm.invoke(formatted)
            questions = [q.strip("-• ").strip() for q in result.content.split("\n") if q.strip()]
            questions = questions[:4]
        except Exception:
            questions = [
                "What skills are listed?",
                "Tell me about the projects",
                "What is the education background?",
                "Summarize the work experience"
            ]

        st.session_state.suggestions_by_pdf[pdf_path] = questions
        return questions


    suggestions = get_suggestions(selected_path, all_chunks)

    # Add suggestions to sidebar
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-card">
            <div class="sidebar-label">💡 Suggested Questions</div>
        </div>
        """, unsafe_allow_html=True)

        for q in suggestions:
            if st.button(q, key=f"sugg_{selected_path}_{q}", use_container_width=True):
                st.session_state["pending_question"] = q

        if st.button("🗑 Clear chat", use_container_width=True, key="clear_chat_btn"):
            st.session_state.chat_by_pdf[selected_path] = []
            st.rerun()

    # ----------------------------------------------------
    # CHAT MEMORY (per selected PDF)
    # ----------------------------------------------------
    if selected_path not in st.session_state.chat_by_pdf:
        st.session_state.chat_by_pdf[selected_path] = []

    messages = st.session_state.chat_by_pdf[selected_path]


    def render_bubble(role, content, chunks=None):
        if role == "user":
            st.markdown(f"""
            <div class="chat-row user">
                <div class="bubble user">{content}</div>
                <div class="avatar user">You</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-row assistant">
                <div class="avatar assistant">AI</div>
                <div class="bubble assistant">
                    <div class="stamp-badge">● Grounded in document</div><br/>
                    {content}
                </div>
            </div>
            """, unsafe_allow_html=True)


    # Display chat or empty state
    with content_panel:
        if not messages:
            st.markdown(f"""
            <div class="empty-state">
                <div class="icon">💬</div>
                <div class="title">Ask anything about {selected_name}</div>
                <div class="sub">Type a question below, or try one of the suggestions in the sidebar</div>
            </div>
            """, unsafe_allow_html=True)

            # Display suggestion chips in grid
            grid_rows = [suggestions[i:i + 2] for i in range(0, len(suggestions), 2)]
            for row in grid_rows:
                row_cols = st.columns(len(row))
                for col, q in zip(row_cols, row):
                    with col:
                        if st.button(q, key=f"chip_{selected_path}_{q}", type="tertiary",
                                     icon="💡", use_container_width=True):
                            st.session_state["pending_question"] = q
        else:
            for msg in messages:
                render_bubble(msg["role"], msg["content"], msg.get("chunks"))

    # ----------------------------------------------------
    # CHAT INPUT
    # ----------------------------------------------------
    question = st.chat_input(f"Ask anything about {selected_name}...")

    if "pending_question" in st.session_state and st.session_state["pending_question"]:
        question = st.session_state.pop("pending_question")

    if question:
        messages.append({"role": "user", "content": question})

        with st.container(key="live_panel"):
            render_bubble("user", question)

            typing_placeholder = st.empty()
            typing_placeholder.markdown("""
            <div style="display:flex; gap:10px; align-items:flex-end; margin:14px 0;">
                <div class="avatar assistant">AI</div>
                <div style="background: var(--surface-solid); border:1px solid var(--border); 
                     border-radius: 16px 16px 16px 4px; padding: 14px 18px; display:flex; gap:5px;">
                    <div style="width:6px; height:6px; border-radius:50%; background:#22D3EE; opacity:0.6;"></div>
                    <div style="width:6px; height:6px; border-radius:50%; background:#22D3EE; opacity:0.6;"></div>
                    <div style="width:6px; height:6px; border-radius:50%; background:#22D3EE; opacity:0.6;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)

            formatted_prompt = prompt.invoke({
                "context": context,
                "question": question,
            })

            response = llm.invoke(formatted_prompt)

            typing_placeholder.empty()
            render_bubble("assistant", response.content, docs)

        messages.append({
            "role": "assistant",
            "content": response.content,
            "chunks": docs,
        })
        st.rerun()

else:
    # No PDFs available - show upload prompt
    st.markdown("""
    <div style="text-align:center; padding:40px 20px;">
        <div style="font-size:48px; margin-bottom:16px;">📄</div>
        <div style="font-family:'Sora', sans-serif; font-weight:700; font-size:20px; 
             color: var(--text); margin-bottom:8px;">No Resumes Found</div>
        <div style="font-size:14px; color: var(--text-soft);">
            Upload a PDF resume using the sidebar to get started
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Embedding", "Gemini")
col2.metric("Vector DB", "FAISS")
col3.metric("LLM", "Gemini 2.0 Flash")

st.markdown("""
<div class="footer">Resume AI Assistant · Built with Streamlit, LangChain, Gemini & FAISS</div>
""", unsafe_allow_html=True)