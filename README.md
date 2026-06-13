# 🧠 DocuMind — RAG Based Document Q&A System

Upload any PDF and chat with it using AI. Built with LangChain, FAISS, Groq LLM, and Streamlit.

## 🔗 Live Demo
https://2ycdaatmt2duv3ggrhowfh.streamlit.app/

## 🛠️ Tech Stack
- LangChain — RAG pipeline
- FAISS — Vector similarity search
- Groq (Llama 3.1) — LLM for answer generation
- HuggingFace Sentence Transformers — Embeddings
- Streamlit — Web interface

## ⚙️ How it works
1. PDF is loaded and split into chunks
2. Chunks are converted to embeddings using sentence-transformers
3. Embeddings stored in FAISS vector database
4. User query is semantically matched to relevant chunks
5. Groq LLM generates grounded answer from retrieved chunks
