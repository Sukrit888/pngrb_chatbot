import streamlit as st
import os
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Based PNGRB Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Based PNGRB Regulatory Chatbot")
st.caption("Ask questions from PNGRB documents using LangChain")

# --------------------------------------------------
# API KEY
# --------------------------------------------------
OPENAI_API_KEY = st.sidebar.text_input(
    "Enter OpenAI API Key",
    type="password"
)

if not OPENAI_API_KEY:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --------------------------------------------------
# SIDEBAR – DOCUMENT UPLOAD
# --------------------------------------------------
st.sidebar.header("📄 Upload PNGRB Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------
def process_documents(files):
    docs = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(docs)

# --------------------------------------------------
# VECTOR STORE + CHAIN
# --------------------------------------------------
def create_chain(docs):
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# --------------------------------------------------
# BUILD KNOWLEDGE BASE
# --------------------------------------------------
if uploaded_files and st.session_state.chain is None:
    with st.spinner("Processing documents..."):
        documents = process_documents(uploaded_files)
        st.session_state.chain = create_chain(documents)
        st.success("Documents indexed successfully!")

# --------------------------------------------------
# CHAT UI
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask your PNGRB-related question...")

if user_input and st.session_state.chain:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chain({"question": user_input})
            answer = result["answer"]
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

elif user_input:
    st.warning("Please upload documents first.")
