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

load_dotenv()

# Step 1 - Load PDF
loader = PyPDFLoader("sample.pdf")
pages = loader.load()
print(f"✅ Loaded {len(pages)} pages")

# Step 2 - Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"✅ Created {len(chunks)} chunks")

# Step 3 - Embeddings
print("⏳ Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Embeddings ready")

# Step 4 - Vector store
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print("✅ Vector store ready")

# Step 5 - LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Step 6 - Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:
""")

# Step 7 - RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 8 - Ask questions
print("\n🤖 RAG system ready!\n")

questions = [
    "What diseases does PawCheck detect?",
    "What CNN models were used in the project?",
    "What is the accuracy of the breed prediction model?"
]

for question in questions:
    print(f"❓ {question}")
    answer = rag_chain.invoke(question)
    print(f"💬 {answer}")
    
    # Get source pages
    source_docs = retriever.invoke(question)
    pages_used = [doc.metadata['page']+1 for doc in source_docs]
    print(f"📄 Sources: Pages {pages_used}")
    print("-" * 60)