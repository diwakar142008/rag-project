import streamlit as st
import os
import hashlib
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.base import BaseCallbackHandler

# Load environment variables
load_dotenv()

# UI
st.set_page_config(page_title="Chat with PDF", layout="centered")
st.title("📄 Chat with your PDF (RAG AI)")

# Streaming callback handler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

@st.cache_resource
def build_or_load_vector_store(pdf_bytes, file_hash):
    embeddings = get_embeddings()
    index_path = f"faiss_streamlit_{file_hash}"

    if os.path.exists(index_path):
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return db

    # Write temp file
    with open("_temp_upload.pdf", "wb") as f:
        f.write(pdf_bytes)

    loader = PyPDFLoader("_temp_upload.pdf")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_path)
    return db

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    with st.spinner("⏳ Processing PDF and building index (first time only)..."):
        db = build_or_load_vector_store(file_bytes, file_hash)

    st.success("✅ PDF ready! Ask your question below")

    query = st.text_input("Ask something from the PDF")

    if query:
        with st.spinner("🔍 Searching relevant chunks..."):
            docs = db.similarity_search(query, k=4)

        st.subheader("🤖 Answer:")

        # Streaming response
        llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True, temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")

        response_container = st.empty()
        stream_handler = StreamHandler(response_container)

        with st.spinner("💬 Generating answer..."):
            response = chain.run(
                input_documents=docs,
                question=query,
                callbacks=[stream_handler]
            )
