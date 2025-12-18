import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Based PNGRB Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Based PNGRB Chatbot")
st.caption("LangChain-powered regulatory assistant")

# --------------------------------------------------
# OPENAI API KEY
# --------------------------------------------------
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload PNGRB PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------
def load_documents(files):
    documents = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)

# --------------------------------------------------
# BUILD VECTORSTORE + CHAIN
# --------------------------------------------------
def build_chain(docs):
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(docs, embeddings)

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
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# --------------------------------------------------
# INITIALIZE KNOWLEDGE BASE
# --------------------------------------------------
if uploaded_files and st.session_state.qa_chain is None:
    with st.spinner("Processing documents..."):
        docs = load_documents(uploaded_files)
        st.session_state.qa_chain = build_chain(docs)
        st.success("Documents indexed successfully")

# --------------------------------------------------
# CHAT UI
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a PNGRB-related question...")

if query and st.session_state.qa_chain:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain({"question": query})
            answer = result["answer"]
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

elif query:
    st.warning("Please upload documents first.")
