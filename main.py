# main.py
"""
RAG Document Assistant - Chat with your uploaded documents (PDF, DOCX, TXT) using Groq LLMs.

Notes:
- set_page_config is first Streamlit call.
- Safe secrets loader: uses st.secrets only if a local secrets.toml exists; otherwise falls back to .env/env var.
- Uses HuggingFaceMiniLM embeddings, persistent FAISS, metadata, conversational retrieval chain with memory.
- Correct FAISS.from_texts signature.
- Model fallback: tries MODEL_CANDIDATES in order and surfaces clear errors if none accessible.
"""

import os
import tempfile
import shutil
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

# Document parsers
from PyPDF2 import PdfReader
import docx2txt

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Groq LLM wrapper
from langchain_groq import ChatGroq

# -----------------------
# IMPORTANT: set_page_config MUST be the first Streamlit call.
# -----------------------
st.set_page_config(page_title="RAG Document Assistant", layout="wide")

# -----------------------
# Configuration & setup
# -----------------------
load_dotenv()  # loads .env into os.environ if present

VECTOR_DIR = "vectorstore_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Candidate models to try in order (edit to match models you have access to)
MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    # add more if you have access
]

# -----------------------
# Helper: check for a local secrets.toml before using st.secrets to avoid Streamlit noisy errors
# -----------------------
def local_secrets_exist() -> bool:
    possible_paths = [
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    try:
        cwd_parent = os.path.dirname(os.getcwd())
        possible_paths.append(os.path.join(cwd_parent, ".streamlit", "secrets.toml"))
    except Exception:
        pass
    return any(os.path.exists(p) for p in possible_paths)

# -----------------------
# Safe secrets loader (avoid touching st.secrets if no local secrets file)
# -----------------------
def get_groq_api_key():
    # 1) Only use st.secrets if secrets.toml exists locally (prevents FileNotFound noise)
    try:
        if local_secrets_exist():
            if "GROQ_API_KEY" in st.secrets:
                val = st.secrets["GROQ_API_KEY"]
                if isinstance(val, dict):
                    maybe = val.get("api_key") or val.get("key") or val.get("value")
                else:
                    maybe = val
                if maybe:
                    return maybe
    except Exception:
        pass

    # 2) fallback to environment variable (.env loaded above)
    env_val = os.getenv("GROQ_API_KEY")
    if env_val:
        return env_val

    return None

# -----------------------
# Cached resources
# -----------------------
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

embedding_model = get_embedding_model()

@st.cache_resource
def get_memory():
    # Explicitly set input_key and output_key so memory knows which fields to store.
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )

# -----------------------
# Document extraction utils
# -----------------------
def extract_text_from_pdf(file) -> List[Dict]:
    pages = []
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                pages.append({"page": i, "text": text})
    except Exception as e:
        st.error(f"Error parsing PDF {getattr(file, 'name', '')}: {e}")
    return pages

def extract_text_from_docx(file) -> List[Dict]:
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.read())
            tmp.flush()
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path) or ""
        os.remove(tmp_path)
        return [{"page": 1, "text": text}] if text.strip() else []
    except Exception as e:
        st.error(f"Error parsing DOCX {getattr(file, 'name', '')}: {e}")
        return []

def extract_text_from_txt(file) -> List[Dict]:
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        raw = file.read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        return [{"page": 1, "text": raw}] if raw and raw.strip() else []
    except Exception as e:
        st.error(f"Error parsing TXT {getattr(file, 'name', '')}: {e}")
        return []

def extract_text_from_file(uploaded_file) -> List[Dict]:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    if name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    if name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    st.warning(f"Unsupported file type: {uploaded_file.name}. Supported: .pdf, .docx, .txt")
    return []

# -----------------------
# Chunking & metadata
# -----------------------
def chunk_texts_with_metadata(text_pages: List[Dict], filename: str, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    texts = []
    metadatas = []
    for p in text_pages:
        page_no = p.get("page", 1)
        raw = p.get("text", "")
        if not raw.strip():
            continue
        chunks = splitter.split_text(raw)
        for i, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({"source": filename, "page": page_no, "chunk": i})
    return texts, metadatas

# -----------------------
# Vectorstore helpers (correct FAISS.from_texts usage)
# -----------------------
def create_or_load_vectorstore(texts, metadatas, persist_dir=VECTOR_DIR, embedding_model=embedding_model):
    # If vectorstore exists, load and optionally add new texts
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            vs = FAISS.load_local(persist_dir, embeddings=embedding_model, allow_dangerous_deserialization=True)
            if texts:
                vs.add_texts(texts, metadatas=metadatas)
                vs.save_local(persist_dir)
            return vs
        except Exception as e:
            st.warning(f"Failed to load existing vector store (will recreate): {e}")
            try:
                shutil.rmtree(persist_dir)
            except Exception:
                pass

    # Create new - FAISS.from_texts requires the embedding object (positional)
    if not texts:
        texts = ["placeholder"]
        metadatas = [{"source": "placeholder", "page": 0, "chunk": 0}]
    vs = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
    vs.save_local(persist_dir)
    return vs

# -----------------------
# Chain builder for a specific model
# -----------------------
def make_chain_for_model(retriever, groq_api_key, model_name: str, temperature: float = 0.2):
    model = ChatGroq(
        temperature=temperature,
        model_name=model_name,
        groq_api_key=groq_api_key,
    )
    memory = get_memory()
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",  # explicit output key so memory knows which field to store
    )
    return chain

# -----------------------
# Safe run: try candidate models in order
# -----------------------
def safe_run_chain(retriever, groq_api_key, question: str, temperature: float = 0.2):
    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            chain = make_chain_for_model(retriever, groq_api_key, model_name, temperature=temperature)
            result = chain({"question": question})
            answer = result.get("answer") or result.get("output_text") or ""
            sources = result.get("source_documents", [])
            return answer, sources, model_name
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            # If it's clearly a model-not-found or access problem, continue to next candidate
            if "model_not_found" in msg or "does not exist" in msg or ("model" in msg and "not found" in msg):
                continue
            # For other transient errors we also continue to try other models,
            # but we capture the last_error to report if all fail.
            continue

    # If none of the models worked:
    raise RuntimeError(
        "None of the configured Groq models were available or accessible. "
        "Tried models: " + ", ".join(MODEL_CANDIDATES) + ". "
        "Last error: " + (str(last_error) if last_error else "unknown")
    )

# -----------------------
# Streamlit UI (app)
# -----------------------
def app():
    GROQ_API_KEY = get_groq_api_key()
    if not GROQ_API_KEY:
        st.warning("GROQ_API_KEY not found. Provide it via Streamlit secrets or environment (.env) to enable chat responses.")

    st.title("📚 RAG Document Assistant — Chat with your files")

    # Sidebar controls
    st.sidebar.header("Settings")
    chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=2000, value=500, step=50)
    chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=500, value=100, step=25)
    top_k = st.sidebar.slider("Retriever top k", min_value=1, max_value=10, value=3)
    temp = st.sidebar.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Vector store dir: `{VECTOR_DIR}`")
    if st.sidebar.button("Rebuild (clear) index"):
        if os.path.exists(VECTOR_DIR):
            try:
                shutil.rmtree(VECTOR_DIR)
                st.success("Vector store cleared.")
            except Exception as e:
                st.error(f"Failed to clear vector store: {e}")
        else:
            st.info("Vector store not found.")

    st.header("1) Upload documents (PDF / DOCX / TXT)")
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf", "docx", "txt"])
    if uploaded_files:
        if st.button("Process & Index uploaded files"):
            all_texts = []
            all_metadatas = []
            prog = st.progress(0)
            for idx, f in enumerate(uploaded_files):
                st.write(f"Processing: **{f.name}**")
                pages = extract_text_from_file(f)
                texts, metadatas = chunk_texts_with_metadata(pages, filename=f.name, chunk_size=chunk_size, overlap=chunk_overlap)
                all_texts.extend(texts)
                all_metadatas.extend(metadatas)
                prog.progress(int(((idx + 1) / len(uploaded_files)) * 100))
            if all_texts:
                vs = create_or_load_vectorstore(all_texts, all_metadatas, persist_dir=VECTOR_DIR, embedding_model=embedding_model)
                st.success(f"Indexed {len(all_texts)} chunks from {len(uploaded_files)} file(s).")
            else:
                st.warning("No text extracted from uploaded files.")

    st.markdown("---")
    st.header("2) Ask questions (conversational)")
    user_query = st.text_input("Type your question here:")
    if st.button("Get answer") and user_query:
        if not os.path.exists(VECTOR_DIR) or not os.listdir(VECTOR_DIR):
            st.warning("No index found. Upload and index documents first.")
        else:
            try:
                vs = FAISS.load_local(VECTOR_DIR, embeddings=embedding_model, allow_dangerous_deserialization=True)
            except Exception as e:
                st.error(f"Failed to load vector store: {e}")
                return

            retriever = vs.as_retriever(search_kwargs={"k": top_k})
            if not GROQ_API_KEY:
                st.error("GROQ_API_KEY is not configured. Provide it via .env or Streamlit secrets to generate answers.")
                return

            with st.spinner("Generating answer (trying models)..."):
                try:
                    answer, srcs, used_model = safe_run_chain(retriever, GROQ_API_KEY, user_query, temperature=temp)
                    st.markdown(f"**Model used:** `{used_model}`")
                    st.markdown("### Answer")
                    st.write(answer)

                    if srcs:
                        st.markdown("### Sources")
                        for d in srcs:
                            md = d.metadata or {}
                            src = md.get("source", "unknown")
                            page = md.get("page", "?")
                            chunk = md.get("chunk", "?")
                            preview = (d.page_content or "")[:400].replace("\n", " ")
                            st.markdown(f"- **{src}** — page {page}, chunk {chunk}")
                            st.caption(preview + ("..." if len(preview) > 380 else ""))
                except RuntimeError as rte:
                    st.error(str(rte))
                    st.info("If you believe you should have access to one of these models, check your Groq dashboard or request access for: " + ", ".join(MODEL_CANDIDATES))
                except Exception as e:
                    st.error(f"Unexpected error during generation: {e}")

    st.markdown("---")
    st.caption("Tip: For deployment, prefer Streamlit secrets (secrets.toml) or environment variables. For scale, move vector store to a managed DB (Pinecone/Qdrant).")

if __name__ == "__main__":
    app()
