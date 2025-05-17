import streamlit as st
from main import react_graph, CURRENT_PAPER
from PyPDF2 import PdfReader
from langchain_core.messages import HumanMessage, AIMessage

# Title
st.set_page_config(page_title="PDF Assistant", layout="wide")
st.title("📄💬 PDF Chat Assistant")

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph_runner" not in st.session_state:
    st.session_state.graph_runner = react_graph.invoke
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# Sidebar for PDF upload
with st.sidebar:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader("Upload a pdf paper", type="pdf")

    
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
        CURRENT_PAPER["content"] = text
        CURRENT_PAPER["title"] = uploaded_file.name
        CURRENT_PAPER["authors"] = "Unknown"  # You can improve this if metadata is needed

        st.success("PDF uploaded and content loaded.✅")

# Chat interface
st.markdown("### 💬Chat with Assistant🤖")

user_input = st.chat_input("Type your message here...")

# Display past chat
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Handle user input
if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Run through LangGraph assistant
    result = st.session_state.graph_runner({"messages": st.session_state.chat_history})
    response = result["messages"][-1]

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response.content)

    # Append assistant response to chat history
    st.session_state.chat_history.append(response)
