import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

PDF_PATH = r"C:\Users\subha\Downloads\Introduction_to_Python_Programming_-_WEB_copy (1).pdf"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "phi3"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3

def load_and_split_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} pages")
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(docs)
    print(f"✂️  Total chunks: {len(chunks)}")
    return chunks

def build_or_load_index(chunks, embeddings):
    if os.path.exists(FAISS_INDEX_PATH):
        print("💾 Loading cached FAISS index from disk...")
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("✅ Index loaded from cache")
    else:
        print("🔨 Building new FAISS index (this runs only once)...")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(FAISS_INDEX_PATH)
        print("💾 Index saved to disk")
    return db

print("⏳ Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print("✅ Embedding model loaded")

chunks = load_and_split_pdf(PDF_PATH)
db = build_or_load_index(chunks, embeddings)

print("🤖 Loading LLM (Ollama)...")
llm = Ollama(model=LLM_MODEL)
print("\n🔥 Chatbot Ready! (type 'exit' to quit)\n")

while True:
    query = input("You: ").strip()
    if not query:
        continue
    if query.lower() == "exit":
        break

    t0 = time.time()
    results = db.similarity_search(query, k=TOP_K)
    context = "\n\n".join([doc.page_content for doc in results])[:1500]

    if not context.strip():
        print("⚠ No relevant context found in PDF!")
        continue

    prompt = f"""You are a helpful AI assistant. Answer clearly and in simple words using ONLY the context below.
- If the answer is not in the context, say: "I don't know"
- Give a proper explanation (not just one line)
- Avoid short or unclear answers

Context:
{context}

Question: {query}

Answer:"""

    print("⏳ Generating answer...")
    response = llm.invoke(prompt)
    elapsed = time.time() - t0

    if not response:
        print("⚠ No response from model!")
    else:
        print(f"\n🤖 AI: {str(response).strip()}\n")
        print(f"⏱️  Response time: {elapsed:.1f}s\n")
