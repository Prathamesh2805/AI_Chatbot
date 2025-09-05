import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
import re
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

st.header("ChatBot Application")

# Upload PDF documents
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload your PDF documents here", type="pdf")

if file is not None:
    # Extract text from PDFs
    text = ""
    pdf_pages = PdfReader(file)
    for page in pdf_pages.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted  

    if text.strip():  # only proceed if we have some text
        # Break text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Generate embeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
        )

        # Store in FAISS
        vector_store = FAISS.from_texts(chunks, embeddings)

        # Get user query
        user_query = st.text_input("Enter your query here")

        if user_query:
            match = vector_store.similarity_search(user_query)
            llm = ChatGroq(
                model="deepseek-r1-distill-llama-70b",
                temperature=0,
                max_retries=2,
            )
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=match, question=user_query)

            # Clean <think> ... </think>
            clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

            st.subheader("Answer")
            st.write(clean_response)
    else:
        st.warning("The uploaded PDF has no extractable text.")
else:
    st.info("Please upload a PDF to get started.")
