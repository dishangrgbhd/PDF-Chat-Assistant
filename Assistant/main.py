from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode,tools_condition
from dotenv import load_dotenv
import os,re
from langchain_core.documents import Document
from typing import Dict,Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START,END,MessagesState
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage
from IPython.display import Image,display

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
langchain_key = os.getenv("LANGCHAIN_API_KEY")

CURRENT_PAPER = {"title": "", "content": "", "authors": ""}

@tool
def pdf_summarizer() -> Dict[str,str]:
    """Summarizes the currently uploaded research paper using LLM."""
    if not CURRENT_PAPER["content"]:
        return {"summary": "No paper uploaded. Please upload a PDF file first."}

    llm = ChatGroq(model="llama3-8b-8192")
    prompt = ChatPromptTemplate.from_template("""You are an expert research assistant.
       Provide a comprehensive summary of {paper_content}.\
       The summary should cover all the key points and main ideas presented in the original text, \
       while also condensing the information into a concise and easy-to-understand format. \
       Please ensure that the summary includes relevant details and examples that support the main ideas,\
       while avoiding any unnecessary information or repetition. The length of the summary should be appropriate for the length \
       and complexity of the original text, providing a clear and accurate overview without omitting any important information.  
    """)
    chain = prompt | llm
    result = chain.invoke({"paper_content": CURRENT_PAPER["content"][:4000]}) 
    return {"summary": result.content}

@tool
def query_pdf(question: str)->Dict[str,str]:
    """Answers a question related to the uploaded research paper using LCEL."""
    if not CURRENT_PAPER["content"]:
        return {"answer": "No paper uploaded. Please upload a PDF file first."}

    # Chunk and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = text_splitter.split_documents([Document(page_content=CURRENT_PAPER["content"])])
    vectorstore = FAISS.from_documents(docs, HuggingFaceEmbeddings())
    retriever = vectorstore.as_retriever()
    #retriever = BM25Retriever.from_documents(docs,k=5)

    # Prompt and chain
    llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
    prompt = ChatPromptTemplate.from_template(""" You are a helpful assistant. 
    When using tools, never refer to tool call IDs like 'call_xyz123' in your response. 
    Only return the clean and helpful result as if you answered directly. Use the following context to answer the question.

        Context: {context}
        Question: {question}

        If the answer is not found in the context, just say you don't know. Do not hallucinate.
        """)
    chain = ( {"context": retriever, "question": RunnablePassthrough()} | prompt | llm )
    result = chain.invoke(question)
    return {"answer": result.content}

llm= ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
tools=[pdf_summarizer,query_pdf]
llm_with_tools=llm.bind_tools(tools)

tool_node=ToolNode(tools=tools)

class State(TypedDict):
    messages: Annotated[list,add_messages]

system_message = SystemMessage(content="""
You are an intelligent assistant. When using tools, never refer to tool call IDs like 'call_xyz123' in your response. 
Only return the clean and helpful result as if you answered directly
Make normal conversation with the user and also help with pdf queries and summarization.
Follow these steps:
1. Chat normally until the user asks questions related to the pdf or asks for summary or summarization of the pdf.
2. As soon as the user uploads the pdf and asks for queries use query_pdf tool but if the user asks for summary or summarization then use pdf_summarizer tool.
""")
def assistant(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke([system_message] + messages)

    # Clean tool-use from the model's content
    cleaned_content = re.sub(r"<tool-use>.*?</tool-use>", "", response.content, flags=re.DOTALL)
    response.content = cleaned_content.strip()

    return {"messages": [response]}


workflow= StateGraph(MessagesState)

workflow.add_node("chatbot",assistant)
workflow.add_node("tools",tool_node)
workflow.add_edge(START,"chatbot")
workflow.add_conditional_edges("chatbot",tools_condition)
workflow.add_edge("tools","chatbot")

react_graph= workflow.compile()

