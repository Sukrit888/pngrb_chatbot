import os
import requests
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PNGRB Regulatory Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 PNGRB Regulatory Chatbot")

# --------------------------------------------------
# OPENAI CONFIG
# --------------------------------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
st.sidebar.header("📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PNGRB PDF / TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# --------------------------------------------------
# LOAD & SPLIT DOCUMENTS (CACHED SAFELY)
# --------------------------------------------------
@st.cache_data
def load_and_split_docs(file_bytes):
    docs = []
    for name, data in file_bytes.items():
        path = f"/tmp/{name}"
        with open(path, "wb") as f:
            f.write(data)

        if name.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif name.endswith(".txt"):
            docs.extend(TextLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)

if uploaded_files:
    file_bytes = {f.name: f.read() for f in uploaded_files}
    chunks = load_and_split_docs(file_bytes)
else:
    st.info("Upload documents to begin.")
    st.stop()

# --------------------------------------------------
# VECTORSTORE (IN-MEMORY ONLY – NO FAISS SAVE/LOAD)
# --------------------------------------------------
@st.cache_resource
def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
    return FAISS.from_documents(chunks, embeddings)

vectorstore = build_vectorstore(chunks)

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

    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join(d.page_content for d in docs)

    system_prompt = (
        "You are a PNGRB regulatory assistant. "
        "Answer strictly using the provided context. "
        "If the answer is not present in the context, say 'I don't know.'"
    )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 300
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                OPENAI_CHAT_URL,
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
