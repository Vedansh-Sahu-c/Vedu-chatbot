import streamlit as st
from rag_engine import RAGEngine
import os

# Configure the Streamlit page layout and title
st.set_page_config(page_title="SISTec Info Bot", page_icon="🎓", layout="wide")

# Inject Dark Theme CSS styling
st.markdown("""
<style>
    :root {
        --bg-color: #0f1117;
        --sidebar-bg: #1a1f2e;
        --card-bg-1: #162032;
        --card-bg-2: #1e293b;
        --accent-blue: #3b82f6;
        --success-green: #34d399;
        --error-red: #ef4444;
        --text-color: #e2e8f0;
        --muted-color: #64748b;
    }
    
    /* Ensure overall backgrounds follow the theme */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
    }

    /* Message containers */
    .user-message {
        background-color: var(--card-bg-2);
        border-right: 4px solid var(--accent-blue);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    
    .bot-message {
        background-color: var(--card-bg-1);
        border-left: 4px solid var(--success-green);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    
    .out-of-scope-message {
        background-color: var(--card-bg-1);
        border: 2px solid var(--error-red);
        border-left: 4px solid var(--error-red);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    
    /* Document chunk displays */
    .chunk-box {
        background-color: var(--bg-color);
        border: 1px solid var(--muted-color);
        padding: 10px;
        border-radius: 5px;
        font-size: 0.85em;
        margin-top: 5px;
        margin-bottom: 10px;
        color: var(--text-color);
    }
    
    .score-pill {
        background-color: var(--accent-blue);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-right: 8px;
    }
    
    /* Header styling */
    .header-banner {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #1a1f2e, #162032);
        border-radius: 10px;
        margin-bottom: 20px;
        border-bottom: 2px solid var(--accent-blue);
    }
    .header-title {
        color: var(--accent-blue);
        margin: 0;
        font-size: 2.5em;
        font-weight: 800;
    }
    .header-badges {
        margin-top: 10px;
        color: var(--muted-color);
        font-size: 0.9em;
        font-weight: bold;
        letter-spacing: 1px;
    }
    
    /* Button Styles for Quick Test panel */
    div.stButton > button {
        width: 100%;
        text-align: left;
        border: 1px solid var(--muted-color);
        border-radius: 6px;
        margin-bottom: 5px;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        border-color: var(--accent-blue);
        color: var(--accent-blue);
        background-color: rgba(59, 130, 246, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to render a chat message
def render_message(role, content, is_out_of_scope=False):
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <div style="color: var(--accent-blue); font-weight: bold; margin-bottom: 5px; text-align: right;">You</div>
            <div style="text-align: right;">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if is_out_of_scope:
            st.markdown(f"""
            <div class="out-of-scope-message">
                <div style="color: var(--error-red); font-weight: bold; margin-bottom: 5px;">Out of Scope</div>
                <div>{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                <div style="color: var(--success-green); font-weight: bold; margin-bottom: 5px;">SISTec Bot</div>
                <div>{content}</div>
            </div>
            """, unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    
    try:
        import google.generativeai as genai
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        st.success("✅ Gemini API Connected")
    except Exception as e:
        st.error("❌ Gemini API Key not found")
        st.stop()
    
    kb_source = st.radio("Knowledge Base Source", ["SISTec Built-in KB", "Upload Documents"])
    
    uploaded_files = None
    if kb_source == "Upload Documents":
        uploaded_files = st.file_uploader("Upload PDF/TXT", accept_multiple_files=True, type=['pdf', 'txt'])
        
    st.markdown("### Advanced Settings")
    chunk_size = st.slider("Chunk Size (words)", 150, 700, 350)
    overlap = st.slider("Overlap (words)", 20, 120, 50)
    top_k = st.slider("Top-K Chunks", 2, 8, 4)
    
    show_chunks = st.checkbox("Show Source Chunks", value=True)
    st.session_state.show_chunks = show_chunks
    
    if st.button("Build Knowledge Base", type="primary"):
        with st.spinner("Building KB..."):
            try:
                engine = RAGEngine(api_key=api_key)
                doc_blocks = []
                if kb_source == "SISTec Built-in KB":
                        if os.path.exists("sistec_knowledge.txt"):
                            doc_blocks = engine.load_text_file("sistec_knowledge.txt", "SISTec KB")
                        else:
                            st.error("sistec_knowledge.txt not found!")
                    else:
                        if uploaded_files:
                            for f in uploaded_files:
                                doc_blocks.extend(engine.load_uploaded_file(f.read(), f.name))
                        else:
                            st.warning("Please upload files first.")
                            
                    if doc_blocks:
                        stats = engine.build(doc_blocks, chunk_size, overlap)
                        st.session_state.engine = engine
                        st.session_state.kb_stats = stats
                except Exception as e:
                    st.error(f"Error building KB: {e}")
            
    st.markdown("---")
    if "kb_stats" in st.session_state:
        st.success("🟢 Ready")
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Total Chunks", st.session_state.kb_stats["total_chunks"])
        col_s2.metric("Vectors Indexed", st.session_state.kb_stats["index_size"])
    else:
        st.warning("🟡 Not built yet")
        
    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN AREA ---
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    <div class="header-banner">
        <h1 class="header-title">SISTec Info Bot</h1>
        <div class="header-badges">RAG | Gemini 1.5 | FAISS | Source-Aware</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render chat history inside a container
    chat_container = st.container(height=540)
    
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Welcome! Build the Knowledge Base first, then ask me anything about SISTec.")
            
        for msg in st.session_state.messages:
            render_message(msg["role"], msg["content"], msg.get("is_out_of_scope", False))
            
            # Show chunks if enabled and if the message has chunks
            if msg.get("chunks") and st.session_state.show_chunks:
                for chunk in msg["chunks"]:
                    st.markdown(f"""
                    <div class="chunk-box">
                        <span class="score-pill">{chunk['score']:.2f}</span>
                        <span style="color: var(--accent-blue); font-weight: bold;">{chunk['source']}</span>
                        <div style="margin-top: 8px;">{chunk['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Handle quick test query from session state
    if "trigger_query" in st.session_state and st.session_state.trigger_query:
        user_input = st.session_state.trigger_query
        st.session_state.trigger_query = None
    else:
        user_input = st.chat_input("Ask about SISTec...")
        
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        if "engine" in st.session_state:
            with chat_container:
                render_message("user", user_input)
                with st.spinner("Thinking..."):
                    engine = st.session_state.engine
                    result = engine.generate_answer(user_input, top_k=top_k)
                    
                    answer_content = result.answer.replace("OUT_OF_SCOPE:", "").strip()
                    if result.is_out_of_scope:
                        answer_content = "I don't have information about that in the SISTec knowledge base."
                        
                    msg_data = {
                        "role": "assistant",
                        "content": answer_content,
                        "is_out_of_scope": result.is_out_of_scope,
                        "chunks": [{"source": c.source + (" - " + c.section_name if c.section_name else ""), "text": c.text, "score": s} for c, s in zip(result.chunks, result.scores)]
                    }
                    
                    st.session_state.messages.append(msg_data)
                    render_message("assistant", answer_content, result.is_out_of_scope)
                    
            st.rerun()
        else:
            with chat_container:
                render_message("user", user_input)
                st.error("Please build the Knowledge Base first!")

# --- RIGHT COLUMN: TEST PANEL ---
with col2:
    st.markdown("### Quick Test Queries")
    
    st.markdown("**In-Scope**")
    test_queries_in_scope = [
        "What is the intake for CSE?",
        "Who is the HOD of AI & Data Science?",
        "Does SISTec have a swimming pool?",
        "What are the placement training modules?",
        "What MoUs does SISTec have with companies?",
        "What is the address of SISTec?",
        "List all B.Tech programs offered",
        "What is the library reading room capacity?",
        "How many buses does SISTec have?",
        "What are cyber security job profiles?",
        "What is SISTec's India Today ranking?",
        "Tell me about hostel facilities"
    ]
    for q in test_queries_in_scope:
        if st.button(q):
            st.session_state.trigger_query = q
            st.rerun()

    st.markdown("---")
    st.markdown("**Out-Of-Scope**")
    test_queries_out_scope = [
        "What is the weather today?",
        "Who is the Prime Minister of India?",
        "Tell me a joke",
        "What is the stock price of TCS?"
    ]
    for q in test_queries_out_scope:
        if st.button(q):
            st.session_state.trigger_query = q
            st.rerun()

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--muted-color); font-size: 0.9em; padding: 10px;">
    SISTec Info Bot | Gemini 1.5 Flash + FAISS + Streamlit | RAG Pipeline
</div>
""", unsafe_allow_html=True)
