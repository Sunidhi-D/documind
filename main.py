import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

load_dotenv()

# Works locally with .env and on cloud with Streamlit secrets
import streamlit as st
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #141824 100%);
        border-right: 1px solid #2d3748;
    }
    
    /* Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    
    .subtitle {
        color: #718096;
        font-size: 1rem;
        margin-top: 0;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 4px;
        margin: 8px 0;
    }

    /* Input box */
    [data-testid="stChatInput"] {
        background: #1a1f2e;
        border: 1px solid #4a5568;
        border-radius: 12px;
    }

    /* Metric cards */
    .stat-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Source badge */
    .source-badge {
        display: inline-block;
        background: #2d3748;
        color: #a0aec0;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px;
        border: 1px solid #4a5568;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #1a1f2e;
        border: 2px dashed #4a5568;
        border-radius: 12px;
        padding: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* Success/info boxes */
    .stSuccess, .stInfo {
        border-radius: 10px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px;'>
            <span style='font-size:3rem'>🧠</span>
            <h2 style='color:#667eea; margin:8px 0 4px;'>DocuMind</h2>
            <p style='color:#718096; font-size:0.85rem;'>AI Document Assistant</p>
        </div>
        <hr style='border-color:#2d3748; margin-bottom:20px'>
    """, unsafe_allow_html=True)

    st.markdown("#### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type="pdf",
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown(f"**{uploaded_file.name}**")
        st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")

        if st.button("⚡ Process & Analyze"):
            with st.spinner("Analyzing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                pages = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                chunks = splitter.split_documents(pages)

                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"}
                )
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 7}
                )

                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    api_key=os.getenv("GROQ_API_KEY")
                )

                prompt = ChatPromptTemplate.from_template("""
                    You are DocuMind, a helpful document assistant. 
                    Use the context below to answer the question as accurately as possible.
                    If the context contains partial information, use it to give the best possible answer.
                    Only say you don't know if the context has absolutely nothing relevant.
                    Use bullet points for detailed answers. Be clear and concise.

                    Context:
                    {context}

                    Question: {question}

                    Answer:
                    """)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                st.session_state.rag_chain = (
                    {"context": st.session_state.retriever | format_docs,
                     "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                st.session_state.doc_stats = {
                    "pages": len(pages),
                    "chunks": len(chunks),
                    "name": uploaded_file.name
                }
                st.session_state.messages = []
                os.unlink(tmp_path)

            st.success("✅ Document ready!")

    # Stats cards
    if st.session_state.doc_stats:
        st.markdown("<hr style='border-color:#2d3748'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Document Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-number'>{st.session_state.doc_stats['pages']}</div>
                    <div class='stat-label'>Pages</div>
                </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-number'>{st.session_state.doc_stats['chunks']}</div>
                    <div class='stat-label'>Chunks</div>
                </div>""", unsafe_allow_html=True)

    # How it works
    st.markdown("<hr style='border-color:#2d3748'>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ How it works")
    steps = [
        ("📥", "PDF loaded & parsed"),
        ("✂️", "Split into chunks"),
        ("🔢", "Converted to embeddings"),
        ("🔍", "Semantic search on query"),
        ("🤖", "LLM generates answer"),
    ]
    for icon, step in steps:
        st.markdown(f"<p style='color:#a0aec0; font-size:0.85rem; margin:4px 0'>{icon} {step}</p>",
                    unsafe_allow_html=True)

# Main area
if st.session_state.rag_chain is None:
    # Landing screen
    st.markdown("""
        <div style='text-align:center; padding: 80px 20px;'>
            <div style='font-size:5rem; margin-bottom:20px'>🧠</div>
            <h1 class='main-title'>DocuMind AI</h1>
            <p class='subtitle'>Chat with any PDF using the power of RAG + LLM</p>
            <br>
            <p style='color:#4a5568; font-size:0.9rem'>Upload a PDF from the sidebar to get started</p>
        </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3)
    features = [
        ("🔍", "Semantic Search", "Finds relevant content even when exact words don't match"),
        ("📄", "Source Citations", "Every answer shows which pages it came from"),
        ("💬", "Multi-turn Chat", "Ask follow-up questions — context is remembered"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(f"""
                <div class='stat-card' style='padding:24px; text-align:center'>
                    <div style='font-size:2rem'>{icon}</div>
                    <h4 style='color:#e2e8f0; margin:10px 0 6px'>{title}</h4>
                    <p style='color:#718096; font-size:0.85rem'>{desc}</p>
                </div>""", unsafe_allow_html=True)
else:
    # Chat header
    st.markdown(f"""
        <div style='padding: 12px 0 20px;'>
            <h2 style='color:#e2e8f0; margin:0'>💬 Chat with <span style='color:#667eea'>{st.session_state.doc_stats['name']}</span></h2>
        </div>
    """, unsafe_allow_html=True)

    # Clear chat button
    if st.button("🗑️ Clear Chat", key="clear"):
        st.session_state.messages = []
        st.rerun()

    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                pages_html = "".join(
                    f"<span class='source-badge'>Page {p}</span>"
                    for p in message["sources"]
                )
                st.markdown(f"<div style='margin-top:8px'>📄 {pages_html}</div>",
                            unsafe_allow_html=True)

    # Chat input
    if question := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner(""):
                answer = st.session_state.rag_chain.invoke(question)
                source_docs = st.session_state.retriever.invoke(question)
                source_pages = sorted(set(
                    [doc.metadata['page'] + 1 for doc in source_docs]
                ))

            st.write(answer)
            pages_html = "".join(
                f"<span class='source-badge'>Page {p}</span>"
                for p in source_pages
            )
            st.markdown(f"<div style='margin-top:8px'>📄 {pages_html}</div>",
                        unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": source_pages
        })