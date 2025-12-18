import os
import requests
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PNGRB Regulatory Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 PNGRB Regulatory Chatbot")
st.caption("Streamlit Cloud–only RAG using Hugging Face Router API")

# --------------------------------------------------
# HUGGING FACE ROUTER CONFIG
# --------------------------------------------------
HF_TOKEN = st.secrets["HF_API_TOKEN"]

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

HF_MODEL = "google/gemma-7b-it"  # Proven, router-supported

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_DIR = "data"
VECTOR_DB_DIR = "vector_db"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
st.sidebar.header("📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PNGRB PDF / TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        with open(os.path.join(DATA_DIR, f.name), "wb") as out:
            out.write(f.read())
    st.sidebar.success("Files uploaded successfully")

# --------------------------------------------------
# VECTOR STORE (SAFE + CLOUD FRIENDLY)
# --------------------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = FakeEmbeddings(size=384)
    index_path = os.path.join(VECTOR_DB_DIR, "index.faiss")

    if os.path.exists(index_path):
        return FAISS.load_local(
            VECTOR_DB_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    docs = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(path).load())

    if not docs:
        st.warning("Upload documents to initialize the chatbot.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_DIR)

    return vectorstore

vectorstore = load_vectorstore()

# --------------------------------------------------
# CHAT UI
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a PNGRB-related question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {
                "model": HF_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 250
            }

            response = requests.post(
                HF_ROUTER_URL,
                headers=HEADERS,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                st.error(response.text)
                st.stop()

            answer = response.json()["choices"][0]["message"]["content"]
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
