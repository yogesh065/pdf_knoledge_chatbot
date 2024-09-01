import os
import time
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
from langsmith.evaluation import evaluate
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
api_key_ = os.getenv("LANGCHAIN_API_KEY")
client = Client(api_key=api_key_)
api_key = os.getenv("GROQ_API_KEY")
api_key=st.secrets["k"]["api_key"]
# Initialize Groq client
client = Groq(api_key=api_key)

# Model configuration
model_name = "sentence-transformers/all-mpnet-base-v2"
batch_size = 166

# Function to read and process PDF
def pdf_reader(pdf_path):
    start_time = time.time()  
    
    loader = PyPDFLoader(file_path=pdf_path)
    raw_documents = loader.load()
    print("1")
    print(f"loaded {len(raw_documents)} documents ")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    all_splits = text_splitter.split_documents(raw_documents)
    print(f"split into {len(all_splits)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("2")
    
    vectordb = None
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i + batch_size]
        if vectordb is None:
            vectordb = Chroma.from_documents(documents=batch, embedding=embeddings, persist_directory="chroma_db")
        else:
            vectordb.add_documents(documents=batch)

    print("3")

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"Time taken to execute the function: {elapsed_time:.2f} seconds")
    
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

# Initialize session state
if 'pdf_refs' not in st.session_state:
    st.session_state.pdf_refs = []
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'pdf_view' not in st.session_state:
    st.session_state.pdf_view = None
if "user_query_history" not in st.session_state:
    st.session_state["user_query_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

# Sidebar for chat history
st.sidebar.title("Chat History")
with st.sidebar:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_query_history"],
    ):
        st.write(f"**User:** {user_query}")
        st.write(f"**Response:** {generated_response}")
        st.write("---")

col1, col2 = st.columns([2, 3])

with col1:
    uploaded_pdfs = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True, key='pdfs')
    if uploaded_pdfs:
        st.session_state.pdf_refs = uploaded_pdfs

    for pdf_ref in st.session_state.pdf_refs:
        if st.button(f"View {pdf_ref.name}"):
            st.session_state.pdf_view = pdf_ref

        if st.session_state.pdf_view == pdf_ref:
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

    if st.session_state.pdf_refs and not st.session_state.vectordb:
        st.session_state.vectordb = process_pdfs_to_vectordb(st.session_state.pdf_refs)

    if query and st.session_state.vectordb:
        with st.spinner("Generating response..."):
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

            st.session_state["user_query_history"].append(query)
            st.session_state["chat_answers_history"].append(formatted_response)

    if st.session_state["chat_answers_history"]:
        for generated_response, user_query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_query_history"],
        ):
            message(user_query, is_user=True)
            message(generated_response)