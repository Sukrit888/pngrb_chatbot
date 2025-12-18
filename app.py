import os
import streamlit as st
from huggingface_hub import InferenceClient

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(page_title="PNGRB Chatbot", page_icon="🤖")
st.title("🤖 PNGRB Regulatory Chatbot")
st.caption("Streamlit Cloud–only RAG (Hugging Face)")

HF_TOKEN = st.secrets["HF_API_TOKEN"]

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

DATA_DIR = "data"
VECTOR_DB_DIR = "vector_db"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload PNGRB documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        with open(os.path.join(DATA_DIR, f.name), "wb") as out:
            out.write(f.read())
    st.sidebar.success("Files uploaded")

# --------------------------------------------------
# VECTORSTORE (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_vectorstore():
    if os.path.exists(os.path.join(VECTOR_DB_DIR, "index.faiss")):
        return FAISS.load_local(
            VECTOR_DB_DIR,
            HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
            allow_dangerous_deserialization=True
        )

    docs = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_DIR)
    return vectorstore

vectorstore = load_vectorstore()

# --------------------------------------------------
# CHAT
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
            response = client.text_generation(
                prompt,
                max_new_tokens=300,
                temperature=0.1
            )
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
