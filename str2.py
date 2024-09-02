import os
import streamlit as st
from streamlit_chat import message
from streamlit_pdf_viewer import pdf_viewer
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from groq import Groq
from langsmith import traceable, Client
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables
load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
api_key_ = os.getenv("LANGCHAIN_API_KEY")
client = Client(api_key=api_key_)
api_key = os.getenv("GROQ_API_KEY")
api_key = st.secrets["k"]["api_key"]

# Initialize Groq client
client = Groq(api_key=api_key)

# Model configuration
model_name = "sentence-transformers/all-mpnet-base-v2"
batch_size = 166

# Initialize session state for each user
def init_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'pdfRefs' not in st.session_state:
        st.session_state.pdfRefs = []
    if 'pdfView' not in st.session_state:
        st.session_state.pdfView = None
    if 'pdfFiles' not in st.session_state:
        st.session_state.pdfFiles = {}
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = {}
    if "userQueryHistory" not in st.session_state:
        st.session_state["userQueryHistory"] = []
    if "chatAnswersHistory" not in st.session_state:
        st.session_state["chatAnswersHistory"] = []

init_session_state()

# Function to read and process PDF
def pdf_reader(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    all_splits = text_splitter.split_documents(raw_documents)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = {}
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i + batch_size]
        if vectordb:
            vectordb.add_documents(documents=batch)
        else:
            vectordb = Chroma.from_documents(documents=batch, embedding=embeddings, persist_directory="chroma_db")
    return vectordb

# Function to run the retrieval and generation process
def run(vectordb, query):
    print("Starting retrieval and generation process...")
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
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

# Streamlit app configuration
st.set_page_config(page_title="Text RAG Application")
st.title("Text RAG Application")

col1, col2 = st.columns([2, 3])

with col1:
    uploaded_pdfs = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True, key='pdfs')
    if uploaded_pdfs:
        if not st.session_state.pdfFiles:
            st.session_state.pdfFiles = {}
        for pdf_ref in uploaded_pdfs:
            pdf_bytes = pdf_ref.getvalue()
            if pdf_ref.name not in st.session_state.pdfFiles:
                st.session_state.pdfFiles[pdf_ref.name] = pdf_bytes
                if pdf_ref.name not in st.session_state.vectordb:
                    st.session_state.vectordb[pdf_ref.name] = pdf_reader(os.path.join("pdfs", pdf_ref.name))
                else:
                    st.session_state.vectordb[pdf_ref.name].add_documents(documents=pdf_reader(os.path.join("pdfs", pdf_ref.name)).get_documents()[0])
            st.session_state.pdfRefs = [pdf_key for pdf_key in st.session_state.pdfFiles.keys()]
    for pdf_ref in st.session_state.pdfRefs:
        if st.button(f"View {pdf_ref}"):
            st.session_state.pdfView = pdf_ref
        if st.session_state.pdfView == pdf_ref:
            binary_data = st.session_state.pdfFiles[pdf_ref]
            pdf_viewer(input=binary_data, width=700, height=600)

with col2:
    query = st.text_input("Query", key='query', placeholder="Enter your query here...")
    if query and st.session_state.vectordb:
        combined_response = []
        for pdf_name, db in st.session_state.vectordb.items():
            try:
                generated_response = run(db, query)
                if isinstance(generated_response, dict) and 'result' in generated_response:
                    response_text = f"**{pdf_name}:** {generated_response['result']}"
                    combined_response.append(response_text)
                else:
                    combined_response.append(f"**{pdf_name}:** Error: Invalid response format")
            except Exception as e:
                combined_response.append(f"**{pdf_name}:** Error: {str(e)}")
        formatted_response = "\n\n".join(combined_response)
        st.write("Generated Response:", formatted_response)
        st.session_state["userQueryHistory"].append(query)
        st.session_state["chatAnswersHistory"].append(formatted_response)

# Display chat history
if st.session_state["chatAnswersHistory"]:
    for generated_response, user_query in zip(
        st.session_state["chatAnswersHistory"],
        st.session_state["userQueryHistory"],
    ):
        message(user_query, is_user=True)
        message(generated_response)