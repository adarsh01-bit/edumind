# app.py
# PURPOSE: Main Streamlit UI for EduMind
#          Connects all backend modules into one web app
#
# BEGINNER NOTE: Streamlit works top to bottom.
#                Every time user interacts, the whole
#                file reruns from top. st.session_state
#                remembers things between reruns.


# ── IMPORTS ──────────────────────────────────────────────

import streamlit as st
import tempfile
import os
from datetime import date

# import our backend modules
from ingest import ingest_document, vectorstore_exists
from rag_pipeline import build_rag_chain, ask_question, format_sources, is_ready
from summarizer import analyze_document, extract_text_from_pdf
from deadline_detector import detect_deadlines, print_deadlines
import pdfplumber


# ── PAGE CONFIGURATION ────────────────────────────────────
# Must be the FIRST streamlit command

st.set_page_config(
    page_title="EduMind — AI Academic Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CUSTOM CSS ────────────────────────────────────────────
# Makes the app look professional

st.markdown(
    """
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Card style for info boxes */
    .info-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #4f8ef7;
    }

    /* Deadline cards */
    /* Common style for all deadline cards */
    .deadline-upcoming,
    .deadline-soon,
    .deadline-urgent {
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.12);    }
    .deadline-urgent {
        background-color: #ffebee;
        border-left: 6px solid #F44336;
        color: #000;
    }
    .deadline-soon {
        background-color: #fff8e1;
        border-left: 6px solid #FFC107;
        color: #000;
    }

    .deadline-upcoming {
        background-color: #f1f3f6;
        border-left: 6px solid #9e9e9e;
        color: #000;    
    }

    .deadline-upcoming:hover,
    .deadline-soon:hover,
    .deadline-urgent:hover {
        transform: translateY(-2px);
        transition: 0.2s ease;
    }
    /* Chat bubbles */
    .chat-user {
        background: #1e3a5f;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        text-align: right;
    }
    .chat-ai {
        background: #1e2130;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 3px solid #4f8ef7;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ── SESSION STATE INITIALIZATION ──────────────────────────
# st.session_state persists data between Streamlit reruns
# Think of it as EduMind's short-term memory


def init_session_state():
    """Initialize all session state variables if not set."""

    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    if "rag_retriever" not in st.session_state:
        st.session_state.rag_retriever = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "doc_text" not in st.session_state:
        st.session_state.doc_text = ""

    if "doc_analysis" not in st.session_state:
        st.session_state.doc_analysis = None

    if "deadlines" not in st.session_state:
        st.session_state.deadlines = None

    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Home"

    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = ""


init_session_state()


# ── SIDEBAR ───────────────────────────────────────────────


def render_sidebar():
    """Renders the left sidebar with navigation + upload."""

    with st.sidebar:

        # Logo + Title
        st.markdown("## 🧠 EduMind")
        st.markdown("*AI Academic Assistant*")
        st.divider()

        # Navigation
        st.markdown("### 📍 Navigation")
        pages = ["🏠 Home", "💬 Chat with PDF", "📋 Summary", "🗓️ Deadlines"]

        for page in pages:
            # disable non-home pages until PDF uploaded
            disabled = not st.session_state.pdf_processed and page != "🏠 Home"

            if st.button(
                page, use_container_width=True, disabled=disabled, key=f"nav_{page}"
            ):
                st.session_state.current_page = page
                st.rerun()

        st.divider()

        # Upload section
        st.markdown("### 📄 Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a PDF", type=["pdf"], help="Upload any academic PDF"
        )

        if uploaded_file:
            st.info(f"📎 {uploaded_file.name}")

            if st.button(
                "🚀 Process Document", type="primary", use_container_width=True
            ):
                process_pdf(uploaded_file)

        # show status
        st.divider()
        if st.session_state.pdf_processed:
            st.success(f"✅ Ready\n\n" f"📄 {st.session_state.uploaded_filename}")

            # reset button
            if st.button("🔄 Upload New PDF", use_container_width=True):
                reset_app()
        else:
            st.warning("⚠️ No document loaded")

        # footer
        st.divider()
        st.markdown(
            "<small>🔒 100% Offline | 💰 Zero Cost | " "📚 RAG Powered</small>",
            unsafe_allow_html=True,
        )


# ── PDF PROCESSING ────────────────────────────────────────


def process_pdf(uploaded_file):
    """
    Handles the full PDF processing pipeline.
    Called when user clicks 'Process Document'.
    """

    with st.spinner("📖 Reading your document..."):

        # save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

    try:
        # Step 1: extract raw text
        with st.spinner("✂️ Extracting text..."):
            text = ""
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"

            if not text.strip():
                st.error("❌ No text found. " "Please use a text-based PDF.")
                return

            st.session_state.doc_text = text

        # Step 2: ingest into ChromaDB
        with st.spinner("🧠 Creating AI index... (2-5 mins first time)"):
            ingest_document(tmp_path, reset=True)

        # Step 3: build RAG chain
        with st.spinner("🔧 Building AI pipeline..."):
            chain, retriever = build_rag_chain()
            st.session_state.rag_chain = chain
            st.session_state.rag_retriever = retriever

        # Step 4: mark as processed
        st.session_state.pdf_processed = True
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.chat_history = []
        st.session_state.doc_analysis = None
        st.session_state.deadlines = None

        st.success("✅ Document ready! Navigate using the menu.")
        st.balloons()
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")

    finally:
        # always clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── RESET APP ─────────────────────────────────────────────


def reset_app():
    """Clears all session state for a fresh start."""

    keys_to_clear = [
        "pdf_processed",
        "rag_chain",
        "rag_retriever",
        "chat_history",
        "doc_text",
        "doc_analysis",
        "deadlines",
        "uploaded_filename",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()


# ── PAGE 1: HOME ──────────────────────────────────────────


def render_home():
    """Renders the home/welcome page."""

    # Hero section
    st.markdown(
        """
    <div style='text-align: center; padding: 40px 0 20px 0;'>
        <h1>🧠 EduMind</h1>
        <h3 style='color: #4f8ef7;'>
            Your AI-Powered Academic Assistant
        </h3>
        <p style='color: #888; font-size: 18px;'>
            Upload any academic PDF and get instant answers,
            summaries, and deadline tracking — fully offline.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Feature cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class='info-card'>
            <h3>💬 Chat</h3>
            <p>Ask anything about your document in plain English</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class='info-card'>
            <h3>📋 Summary</h3>
            <p>Get instant structured summaries and key points</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class='info-card'>
            <h3>🗓️ Deadlines</h3>
            <p>Auto-detect all dates and deadlines in your docs</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class='info-card'>
            <h3>🔒 Offline</h3>
            <p>100% private — nothing leaves your Mac</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    if not st.session_state.pdf_processed:
        # Getting started guide
        st.markdown("### 🚀 Getting Started")
        st.markdown(
            """
        1. **Upload a PDF** using the sidebar on the left
        2. Click **Process Document** and wait ~2 minutes
        3. Navigate to **Chat**, **Summary**, or **Deadlines**
        4. Ask questions and get instant AI answers!
        """
        )

        st.info(
            "💡 Works best with: Syllabi, Course Notes, "
            "Academic Notices, Research Papers, Assignments"
        )

    else:
        # Document loaded — show quick stats
        st.markdown("### 📊 Document Loaded")

        word_count = len(st.session_state.doc_text.split())
        char_count = len(st.session_state.doc_text)

        col1, col2, col3 = st.columns(3)
        col1.metric("📄 File", st.session_state.uploaded_filename)
        col2.metric("📝 Words", f"{word_count:,}")
        col3.metric("🔤 Characters", f"{char_count:,}")

        st.success(
            "✅ Document ready! Use the navigation menu to "
            "Chat, Summarize, or find Deadlines."
        )


# ── PAGE 2: CHAT ──────────────────────────────────────────


def render_chat():
    """Renders the chat interface."""

    st.markdown("## 💬 Chat with Your Document")
    st.markdown(f"*Talking to: **{st.session_state.uploaded_filename}***")
    st.divider()

    # suggested questions
    if not st.session_state.chat_history:
        st.markdown("#### 💡 Try asking:")

        suggestions = [
            "What is this document about?",
            "What are the main topics covered?",
            "Are there any important deadlines?",
            "Summarize the key information",
        ]

        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(
                    suggestion, use_container_width=True, key=f"suggestion_{i}"
                ):
                    # add to chat as if user typed it
                    handle_chat_message(suggestion)

        st.divider()

    # display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.write(message["content"])

                # show sources if available
                if message.get("sources"):
                    with st.expander("📖 View source sections from document"):
                        formatted = format_sources(message["sources"])
                        for src in formatted:
                            st.markdown(
                                f"**Section {src['chunk_number']} "
                                f"from {src['source']}:**"
                            )
                            st.text(src["content"][:300] + "...")
                            st.divider()

    # chat input
    user_input = st.chat_input("Ask anything about your document...")

    if user_input:
        handle_chat_message(user_input)

    # clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


def handle_chat_message(question):
    """Processes a chat message through RAG pipeline."""

    # add user message to history
    st.session_state.chat_history.append({"role": "user", "content": question})

    # get answer from RAG
    with st.spinner("🧠 Thinking..."):
        answer, sources = ask_question(
            question, st.session_state.rag_chain, st.session_state.rag_retriever
        )

    # add AI response to history
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

    st.rerun()


# ── PAGE 3: SUMMARY ───────────────────────────────────────


def render_summary():
    """Renders the document summary page."""

    st.markdown("## 📋 Document Summary")
    st.markdown(f"*Analyzing: **{st.session_state.uploaded_filename}***")
    st.divider()

    # generate summary if not already done
    if st.session_state.doc_analysis is None:

        st.info(
            "🧠 Click below to generate a complete AI summary. " "Takes 2-4 minutes."
        )

        if st.button(
            "📋 Generate Full Summary", type="primary", use_container_width=True
        ):
            with st.spinner("🧠 Analyzing document... (2-4 minutes)"):
                analysis = analyze_document(st.session_state.doc_text)
                st.session_state.doc_analysis = analysis
            st.rerun()

    else:
        analysis = st.session_state.doc_analysis

        # document type badge
        doc_type = analysis["doc_type"]
        st.markdown(
            f"**Document Type:** `{doc_type}`  |  "
            f"**Words:** `{analysis['word_count']:,}`  |  "
            f"**Characters:** `{analysis['char_count']:,}`"
        )
        st.divider()

        # two columns layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📋 Full Summary")
            st.markdown(analysis["summary"])

        with col2:
            st.markdown("### 🔑 Key Points")
            if analysis["key_points"]:
                for i, point in enumerate(analysis["key_points"], 1):
                    st.markdown(
                        f"""
                        <div class='info-card'>
                            <strong>{i}.</strong> {point}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No key points extracted.")

        st.divider()

        # regenerate button
        if st.button("🔄 Regenerate Summary"):
            st.session_state.doc_analysis = None
            st.rerun()


# ── PAGE 4: DEADLINES ─────────────────────────────────────


def render_deadlines():
    """Renders the deadline dashboard."""

    st.markdown("## 🗓️ Deadline Dashboard")
    st.markdown(f"*Scanning: **{st.session_state.uploaded_filename}***")
    st.markdown(f"📅 Today: **{date.today().strftime('%B %d, %Y')}**")
    st.divider()

    # detect deadlines if not already done
    if st.session_state.deadlines is None:

        st.info("🔍 Click below to scan document for all " "dates and deadlines.")

        if st.button("🗓️ Scan for Deadlines", type="primary", use_container_width=True):
            with st.spinner("🔍 Scanning document for dates..."):
                deadlines = detect_deadlines(st.session_state.doc_text)
                st.session_state.deadlines = deadlines
            st.rerun()

    else:
        deadlines = st.session_state.deadlines

        if not deadlines:
            st.warning(
                "⚠️ No specific dates found in this document. "
                "Try uploading a syllabus or academic calendar."
            )
            if st.button("🔄 Scan Again"):
                st.session_state.deadlines = None
                st.rerun()
            return

        # summary stats
        today = date.today()
        upcoming = [d for d in deadlines if d["days_left"] >= 0]
        urgent = [d for d in deadlines if 0 <= d["days_left"] <= 7]
        passed = [d for d in deadlines if d["days_left"] < 0]

        # metric row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📌 Total Found", len(deadlines))
        col2.metric("⏳ Upcoming", len(upcoming))
        col3.metric("🔴 Urgent (≤7 days)", len(urgent))
        col4.metric("✅ Passed", len(passed))

        st.divider()

        # filter tabs
        tab1, tab2, tab3 = st.tabs(["⏳ All Upcoming", "🔴 Urgent Only", "✅ Passed"])

        with tab1:
            render_deadline_list(upcoming, "upcoming")

        with tab2:
            render_deadline_list(urgent, "urgent")

        with tab3:
            render_deadline_list(passed, "passed")

        st.divider()

        # rescan button
        if st.button("🔄 Rescan Document"):
            st.session_state.deadlines = None
            st.rerun()


def render_deadline_list(deadlines, mode):
    """Renders a list of deadline cards."""

    if not deadlines:
        st.info("No deadlines in this category.")
        return

    for d in deadlines:
        days = d["days_left"]

        # pick card style based on urgency
        if days < 0:
            card_class = "deadline-upcoming"
        elif days <= 3:
            card_class = "deadline-urgent"
        elif days <= 7:
            card_class = "deadline-soon"
        else:
            card_class = "deadline-upcoming"

        # format date nicely
        nice_date = d["parsed_date"].strftime("%B %d, %Y")

        st.markdown(
            f"""
            <div class='{card_class}'>
                <strong>{d['label']}</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                📅 {nice_date}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                {d['status']}
                <br><br>
                <small style='color: #666;'>
                    Context: ...{d['context'][:100]}...
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── MAIN ROUTER ───────────────────────────────────────────


def main():
    """
    Main function — renders sidebar + correct page
    based on current_page in session state.
    """

    # always render sidebar
    render_sidebar()

    # route to correct page
    page = st.session_state.current_page

    if page == "🏠 Home":
        render_home()

    elif page == "💬 Chat with PDF":
        if st.session_state.pdf_processed:
            render_chat()
        else:
            st.warning("Please upload a PDF first.")

    elif page == "📋 Summary":
        if st.session_state.pdf_processed:
            render_summary()
        else:
            st.warning("Please upload a PDF first.")

    elif page == "🗓️ Deadlines":
        if st.session_state.pdf_processed:
            render_deadlines()
        else:
            st.warning("Please upload a PDF first.")


# ── RUN ───────────────────────────────────────────────────

if __name__ == "__main__":
    main()
