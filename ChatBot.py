import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain

# Load Groq API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.header("ChatBot Application")

# Upload PDF documents
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload your PDF documents here", type="pdf")

# Extract text from PDFs
text = ""
if file is not None:
    pdf_pages = PdfReader(file)
    for page in pdf_pages.pages:
        text += page.extract_text() or ""

# Break text into chunks
if text:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Store embeddings in FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user query
    user_query = st.text_input("Enter your query here")

    if user_query:
        match = vector_store.similarity_search(user_query)
        
        # ✅ Pass the API key directly
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-70b-versatile",
            temperature=0,
            max_retries=2,
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_query)

        # Remove <think> tags if present
        import re
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        st.subheader("Answer")
        st.write(response)
