import os
import tempfile
import hashlib
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Based PNGRB Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Based PNGRB Chatbot")
st.caption("Optimized LangChain-based regulatory assistant")

INDEX_DIR = "faiss_index"

# --------------------------------------------------
# API KEY
# --------------------------------------------------
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not api_key:
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload PNGRB PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def get_files_hash(files):
    hasher = hashlib.md5()
    for f in files:
        hasher.update(f.getvalue())
    return hasher.hexdigest()

@st.cache_resource(show_spinner=False)
def build_or_load_vectorstore(file_hash, files):
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(
            INDEX_DIR,
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )

    documents = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            documents.extend(PyPDFLoader(tmp.name).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(
        docs,
        OpenAIEmbeddings()
    )

    vectorstore.save_local(INDEX_DIR)
    return vectorstore

# --------------------------------------------------
# CHAIN
# --------------------------------------------------
def build_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory
    )

# --------------------------------------------------
# INIT
# --------------------------------------------------
if uploaded_files:
    file_hash = get_files_hash(uploaded_files)

    with st.spinner("Loading knowledge base (first time may take a while)..."):
        vectorstore = build_or_load_vectorstore(file_hash, uploaded_files)
        qa_chain = build_chain(vectorstore)
else:
    qa_chain = None

# --------------------------------------------------
# CHAT UI
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a PNGRB-related question...")

if query and qa_chain:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain({"question": query})
            answer = result["answer"]
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
