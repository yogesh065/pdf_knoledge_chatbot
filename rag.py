import os
from dotenv import load_dotenv,find_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from groq import Groq
from langchain import hub
from langsmith import Client
import uuid
import streamlit as st 

load_dotenv(load_dotenv(find_dotenv()))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
api_key_=os.getenv("LANGCHAIN_API_KEY")
api_key = os.getenv("GROQ_API_KEY")
api_key = st.secrets["k"]["api_key"]

client = Groq(api_key=api_key)
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

