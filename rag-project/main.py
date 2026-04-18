from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

def loader():
    loaders=PyPDFLoader(r"C:\Users\subha\Downloads\Introduction_to_Python_Programming_-_WEB_copy (1).pdf")
    document=loaders.load()
    return document

docs=loader()

print("total no of pages:",len(docs))


text_splitter = CharacterTextSplitter(
    chunk_size=300,      # ✅ correct
    chunk_overlap=50
)

chunks=text_splitter.split_documents(docs)

print("total no of chunks:",len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("embedding is loaded")
db = FAISS.from_documents(chunks, embeddings)
print("vector Dready")
llm = Ollama(model="phi3")

print("\n🔥 Chatbot Ready!\n")
print("model loaded")
while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    results = db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in results])[:1500]

    print("\n--- DEBUG ---")
    print("Context length:", len(context))

    if not context.strip():
        print("⚠ PDF has no readable text!")
        continue

    prompt = f"""
You are a helpful AI assistant.

Answer clearly and in simple words using ONLY the context below.

- If the answer is not in the context, say: "I don't know"
- Give a proper explanation (not just one line)
- Avoid short or unclear answers

Context:
{context}

Question: {query}

Answer:
"""

    print("⏳ Generating answer...")

    response = llm.invoke(prompt)

    if not response:
        print("⚠ No response from model!")
    else:
        print("\n🤖 AI:", str(response).strip(), "\n")  