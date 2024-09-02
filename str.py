import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv,find_dotenv
import os
from langsmith_core.runnables import RunnablePassthrough, RunnableMap
from langsmith_core.output_parsers import StrOutputParser
from langsmith_core.prompts import PromptTemplate
from langsmith_groq import ChatGroq
from langchain_core.chains import RetrievalQA
from groq import Groq
from langsmith import Client
from langsmith.evaluation import evaluate
import uuid
from langchain import hub
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import time

# Load environment variables
load_dotenv(load_dotenv(find_dotenv()))

# Create a page configuration
st.set_page_config(page_title="Text RAG Application", layout="wide")

# Create a main panel
if 'pdf_refs' not in st.session_state:
    st.session_state.pdf_refs = []
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'user_query' not in st.session_state:
    st.session_state.user_query = []
if 'chat_answers' not in st.session_state:
    st.session_state.chat_answers = []

def create_vector_db(pdf_refs):
    pdf_directory = "pdfs"
    os.makedirs(pdf_directory, exist_ok=True)

    vectordb = {}
    for pdf_ref in pdf_refs:
        pdf_path = os.path.join(pdf_directory, pdf_ref.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_ref.getvalue())
        loader = PyPDFLoader(file_path=pdf_path)
        raw_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        all_splits = text_splitter.split_documents(raw_documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        vectordb[pdf_ref.name] = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

    return vectordb

def get_response(vectordb, query):
    llm = ChatGroq(model="llama3-8b-8192")
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
    res = qa(query)
    unique_sources = set()
    source_documents = []
    for doc in res['source_documents']:
        source = doc.metadata['source']
        if source not in unique_sources:
            unique_sources.add(source)
            source_documents.append({
                "source": source,
                "content": doc.page_content
            })
    result = {
        "result": res['result'],
        "source_documents": source_documents
    }
    return result

st.title("Text RAG Application")

col1, col2 = st.columns([2, 3])

with col1:
    uploaded_pdfs = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True, key='pdfs')

    if uploaded_pdfs:
        st.session_state.pdf_refs = uploaded_pdfs

if st.session_state.pdf_refs and not st.session_state.vectordb:
    st.session_state.vectordb = create_vector_db(st.session_state.pdf_refs)

with col2:
    query = st.text_input("Query", placeholder="Enter your query here...")

    if query:
        with st.spinner("Generating response..."):
            response = get_response(st.session_state.vectordb, query)
            st.session_state.user_query.append(query)
            st.session_state.chat_answers.append(response)

    for i in range(len(st.session_state.user_query)):
        col1.write(f"**User:** {st.session_state.user_query[i]}")
        col1.write(f"**Response:** {st.session_state.chat_answers[i]['result']}")
        col1.write("------")

# Clear chat history
if st.button("Clear Chat History"):
    st.session_state.user_query = []
    st.session_state.chat_answers = []