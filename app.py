import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Chatbot – PNGRB Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI-Based PNGRB Chatbot Assistant")
st.caption("Ask questions based on the provided data & logic")

# --------------------------------------------------
# SESSION STATE (Chat Memory)
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# CORE CHATBOT LOGIC (ADAPT FROM NOTEBOOK)
# --------------------------------------------------
def chatbot_response(user_query):
    """
    Replace this logic with the core processing
    from pngrb_chatbot.ipynb
    """

    # --- Example placeholder logic ---
    user_query = user_query.lower()

    if "pngrb" in user_query:
        return "PNGRB stands for Petroleum and Natural Gas Regulatory Board."
    elif "license" in user_query:
        return "PNGRB licenses regulate city gas distribution and pipelines."
    elif "help" in user_query:
        return "You can ask about PNGRB rules, entities, compliance, or documents."
    else:
        return "Sorry, I couldn't find a relevant answer. Please rephrase your question."

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
user_input = st.chat_input("Ask your question here...")

if user_input:
    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    response = chatbot_response(user_input)

    # Store assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)
