from langchain_community.document_loaders import PyPDFLoader as TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# Load document
loader = TextLoader(r"C:\Users\subha\Downloads\DSA Unit - 1.pdf")
docs = loader.load()

# Split text
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(docs)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Store in vector DB
db = Chroma.from_documents(texts, embeddings)

# Create retriever
retriever = db.as_retriever()

# LLM
llm = ChatOpenAI()

# 🔥 Manual RAG (instead of RetrievalQA)
query = "What is this document about?"

docs = retriever.get_relevant_documents(query)

context = "\n".join([doc.page_content for doc in docs])

prompt = f"""
Answer the question using the context below:

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print(response.content)