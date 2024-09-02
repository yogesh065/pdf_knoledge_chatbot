import os
import streamlit as st
from streamlit_chat import message
from streamlit_pdf_viewer import pdf_viewer
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from groq import Groq
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables
api_key_ = os.getenv("LANGCHAIN_API_KEY")
api_key = os.getenv("GROQ_API_KEY")
api_key = st.secrets["k"]["api_key"]

import streamlit as st
from streamlit_chat import message
from streamlit_pdf_viewer import pdf_viewer

import os
from rag import run
from pdf_reader_fun import pdf_reader

st.set_page_config(page_title="Text RAG Application")
st.title("Text RAG Application")

st.sidebar.title("Chat History")

# Create a unique session state for each user
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())

# Initialize session state
if 'pdf_refs' not in st.session_state:
    st.session_state['pdf_refs'] = {}
if 'vectordb' not in st.session_state:
    st.session_state['vectordb'] = {}
if 'pdf_view' not in st.session_state:
    st.session_state['pdf_view'] = None

if "user_query_history" not in st.session_state['vectordb']:
    st.session_state['vectordb']["user_query_history"] = []
if "chat_answers_history" not in st.session_state['vectordb']:
    st.session_state['vectordb']["chat_answers_history"] = []

with st.sidebar:
    for generated_response, user_query in zip(
        st.session_state['vectordb']["chat_answers_history"],
        st.session_state['vectordb']["user_query_history"],
    ):
        st.write(f"**User:** {user_query}")
        st.write(f"**Response:** {generated_response}")
        st.write("---")

col1, col2 = st.columns([2, 3])

with col1:
    uploaded_pdfs = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True, key='pdfs')

    if uploaded_pdfs:
        st.session_state['pdf_refs'][st.session_state['user_id']] = uploaded_pdfs

    for pdf_ref in st.session_state['pdf_refs'][st.session_state['user_id']]:
        if st.button(f"View {pdf_ref.name}"):
            st.session_state['pdf_view'] = pdf_ref

        if st.session_state['pdf_view'] == pdf_ref:
            binary_data = pdf_ref.getvalue()
            pdf_viewer(input=binary_data, width=700, height=600)

with col2:
    query = st.text_input("Query", key='query', placeholder="Enter your query here...")

    def process_pdfs_to_vectordb(pdf_refs):
        with st.spinner("Processing PDFs..."):
            pdf_directory = "pdfs"
            os.makedirs(pdf_directory, exist_ok=True)

            vectordb = {}
            for pdf_ref in pdf_refs:
                pdf_path = os.path.join(pdf_directory, pdf_ref.name)
                with open(pdf_path, "wb") as f:
                    f.write(pdf_ref.getvalue())
                vectordb[pdf_ref.name] = pdf_reader(pdf_path)

            return vectordb

    if st.session_state['pdf_refs'][st.session_state['user_id']] and not st.session_state['vectordb'].get(st.session_state['user_id']):
        st.session_state['vectordb'][st.session_state['user_id']] = process_pdfs_to_vectordb(st.session_state['pdf_refs'][st.session_state['user_id']])

    if query and st.session_state['vectordb'].get(st.session_state['user_id']):
        with st.spinner("Generating response..."):
            combined_response = []
            for pdf_name, db in st.session_state['vectordb'][st.session_state['user_id']].items():
                generated_response = run(db, query)
                if isinstance(generated_response, dict) and 'result' in generated_response:
                    response_text = f"**{pdf_name}:** {generated_response['result']}"
                    combined_response.append(response_text)


                else:
                    combined_response.append(f"**{pdf_name}:** Error: Invalid response format")

            formatted_response = "\n\n".join(combined_response)

            st.write("Generated Response:", formatted_response)

            st.session_state['vectordb'][st.session_state['user_id']]["user_query_history"].append(query)
            st.session_state['vectordb'][st.session_state['user_id']]["chat_answers_history"].append(formatted_response)

    if st.session_state['vectordb'].get(st.session_state['user_id']) and st.session_state['vectordb'][st.session_state['user_id']].get("chat_answers_history"):
        for generated_response, user_query in zip(
            st.session_state['vectordb'][st.session_state['user_id']]["chat_answers_history"],
            st.session_state['vectordb'][st.session_state['user_id']]["user_query_history"],
        ):
            message(user_query, is_user=True)
            message(generated_response)