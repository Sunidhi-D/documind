# 📄 DocuMind – AI-Powered Document Intelligence Assistant

DocuMind is an intelligent RAG (Retrieval-Augmented Generation) application that allows users to upload PDF documents and ask questions in natural language. The system retrieves the most relevant content from the document, generates context-aware answers using LLMs, and provides page-level citations for transparency and trust.

---

## 🚀 Features

- 📄 Upload and analyze PDF documents
- 🔍 Semantic search using vector embeddings
- 🤖 AI-powered question answering
- 📚 Page-level citations for every response
- 💬 Multi-turn conversational chat
- ⚡ Fast document retrieval with FAISS
- 🧠 Context-aware responses grounded in document content
- 🎨 Modern and responsive Streamlit interface

---

## 📸 Screenshots

### Home Page

<img width="935" height="394" alt="Screenshot 2026-06-17 220110" src="https://github.com/user-attachments/assets/5d77c06f-e40f-4163-8ff8-6b228276ce6c" />


### Question Answering

<img width="953" height="421" alt="Screenshot 2026-06-17 220912" src="https://github.com/user-attachments/assets/f563fa74-254a-4076-80bb-586edcad4638" />


---

## 🏗️ System Architecture

```text
User
 │
 ▼
Streamlit Frontend
 │
 ▼
PDF Upload
 │
 ▼
PyPDF Loader
 │
 ▼
Text Extraction
 │
 ▼
Text Chunking
 │
 ▼
Sentence Transformers
 │
 ▼
Vector Embeddings
 │
 ▼
FAISS Vector Store
 │
 ▼
Semantic Retrieval
 │
 ▼
Groq LLM
 │
 ▼
Grounded Response
 │
 ▼
Answer + Page Citation
```

---

## ⚙️ Technologies Used

### Frontend

- Streamlit

### Backend & AI

- Python
- LangChain
- Groq API
- FAISS
- Sentence Transformers

### Document Processing

- PyPDF
- LangChain Text Splitters

### Environment Management

- python-dotenv

---

## 📂 Project Structure

```text
DocuMind/
│
├── app.py
├── requirements.txt
├── .env
├── README.md
│
├── data/
│   └── uploaded_pdfs
│
├── vectorstore/
│
└── screenshots/
    ├── home.png
    ├── uploaded.png
    ├── chat.png
    └── insights.png
```

---

## 🔄 Workflow

### Step 1: Upload PDF

Upload any PDF document through the DocuMind interface.

### Step 2: Document Processing

The system:

- Extracts text from PDF pages
- Splits content into overlapping chunks
- Prepares chunks for embedding generation

### Step 3: Embedding Creation

Sentence Transformer models convert document chunks into vector embeddings.

### Step 4: Vector Storage

Embeddings are stored in a FAISS vector database for efficient semantic retrieval.

### Step 5: Ask Questions

Users can ask questions in natural language about the uploaded document.

### Step 6: Semantic Search

The system retrieves the most relevant document chunks using similarity search.

### Step 7: AI Response Generation

Retrieved content is passed to the Groq-powered LLM to generate accurate answers.

### Step 8: Page Citation

Every answer includes the source page information for verification and transparency.

---

## 🧠 AI Capabilities

### Semantic Retrieval

DocuMind understands the meaning of questions rather than relying solely on keyword matching.

### Context-Aware Answers

Responses are generated using only relevant document content to reduce hallucinations.

### Multi-Turn Conversations

Users can continue asking follow-up questions while maintaining conversation context.

### Explainable Responses

Every answer references the exact page from which the information was retrieved.

---

## 📊 Example Query

### User Question

```text
Give details about the project.
```

### AI Response

```text
The project is an AI-based pet health assistant called PawCheck.

Its primary objectives include:

• Early disease detection in pets
• Preventive healthcare support
• AI-powered chatbot assistance
• Improved accessibility to veterinary services

The system integrates CNN-based disease prediction,
NLP-powered conversational support, and location-based
veterinary recommendations.

Source: Page 8
```

---

## 🛠️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/DocuMind.git
cd DocuMind
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

### Run Application

```bash
streamlit run app.py
```

---

## 📦 Dependencies

```txt
streamlit
langchain
langchain-community
langchain-groq
langchain-core
langchain-text-splitters
faiss-cpu
pypdf
sentence-transformers
python-dotenv
groq
```

Source: requirements.txt :contentReference[oaicite:0]{index=0}

---

## 🎯 Key Features Explained

### PDF Parsing

Extracts text directly from uploaded PDF files for downstream processing.

### Chunking Strategy

Documents are split into overlapping chunks to preserve context and improve retrieval accuracy.

### Vector Search

FAISS enables fast similarity search across thousands of document chunks.

### Retrieval-Augmented Generation (RAG)

Combines semantic search with LLM reasoning to produce grounded answers.

### Source Attribution

Every response includes page references from the original document.

---

## 🔮 Future Enhancements

- Support for DOCX and TXT files
- Multiple document querying
- PDF summarization
- Chat history persistence
- Document comparison
- Voice-based querying
- Export answers to PDF
- OCR support for scanned documents

---

## 👩‍💻 Author

**Sunidhi Divekar**

AI & Data Science Engineering Student

DocuMind was developed to make document understanding easier through AI-powered retrieval, semantic search, and conversational question answering.

---

## 🌟 Highlights

- RAG-based architecture
- Semantic document search
- Groq-powered LLM responses
- Page-level citations
- Multi-turn chat support
- Fast FAISS retrieval
- Modern Streamlit UI
- Explainable AI responses
