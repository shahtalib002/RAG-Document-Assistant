## 🧠 Overview

The **RAG Document Assistant** is an AI-powered web application that allows users to **chat with their documents** — such as PDFs, DOCX, and TXT files.  
It uses the **Retrieval-Augmented Generation (RAG)** technique to find relevant text from documents and answer user questions accurately using **Groq-hosted Large Language Models (LLMs)**.

This project combines the power of:
- 🧩 **LangChain** for building the RAG pipeline  
- ⚡ **Groq LLMs** for fast and intelligent text generation  
- 📚 **FAISS** for local vector search  
- 🖥️ **Streamlit** for an easy-to-use web interface  

---

## 🎯 Features

✅ Upload multiple files (PDF, DOCX, TXT)  
✅ Ask questions and get precise, context-aware answers  
✅ Stores document embeddings locally using **FAISS**  
✅ Conversational memory — remembers previous queries  
✅ Provides cited document sources for transparency  
✅ Fully customizable (chunk size, overlap, top-k, temperature)  
✅ Safe API key management using `.env` or `secrets.toml`  
✅ Works offline for retrieval once embeddings are created  

---
## ⚙️ Tech Stack

| Component | Technology Used |
|------------|----------------|
| Frontend | Streamlit |
| LLM Framework | LangChain |
| LLM Provider | Groq (via `langchain_groq`) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | FAISS |
| Document Parsing | PyPDF2, docx2txt |
| Environment Management | python-dotenv |
| Programming Language | Python 3.10 |

---

## 🧩 How It Works

1. **File Upload** → You upload one or more documents.  
2. **Text Extraction** → Text is extracted from each document (PDF/DOCX/TXT).  
3. **Chunking** → The text is split into smaller overlapping chunks (default: 500 chars).  
4. **Embedding** → Each chunk is converted into a numeric vector using `sentence-transformers`.  
5. **Storage** → All vectors are stored locally in a FAISS index.  
6. **Retrieval** → When you ask a question, the app retrieves the most relevant chunks.  
7. **Generation** → The question + retrieved text are passed to a Groq LLM (e.g., `llama-3.3-70b-versatile`) for generating an answer.  
8. **Display** → The final answer and source documents are shown in the Streamlit interface.

---

## 🧠 Example Use Cases

| Scenario | Example Query | Example Response |
|-----------|----------------|------------------|
| Resume Analysis | “What type of job can this candidate apply for?” | Suggests AI/ML Engineer, Data Scientist, etc. |
| Research Paper Review | “What are the main contributions of this paper?” | Generates concise summary |
| Legal Document | “What are the key clauses in this agreement?” | Lists major clauses and conditions |
| Business Reports | “Summarize financial performance.” | Generates short analysis summary |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/shahtalib002/RAG-Document-Assistant.git
cd RAG-Document-Assistant
