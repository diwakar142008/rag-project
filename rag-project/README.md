# 📚 RAG PDF Chatbot

A high-performance Retrieval-Augmented Generation (RAG) application that lets you chat with your PDF documents using AI. Built with LangChain, FAISS, and support for both local (Ollama) and cloud (OpenAI) LLMs.

## ✨ Features

- **Two Interfaces:**
  - `main.py` — CLI chatbot with local Ollama (phi3) — no API key needed
  - `rag2.py` — Streamlit web app with OpenAI GPT-3.5 — streaming responses
- **Persistent Vector Store** — FAISS index saved to disk, loaded instantly on restart
- **Streaming Responses** — real-time token streaming for faster perceived response
- **Smart Caching** — `@st.cache_resource` prevents re-embedding on every query
- **Fast Retrieval** — top-k similarity search with configurable chunk overlap

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) (for `main.py` only)
- OpenAI API key (for `rag2.py` only)

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/rag-project.git
cd rag-project

# Create virtual environment
python -m venv rag-env
rag-env\Scripts\activate   # Windows
# source rag-env/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Setup

1. Copy `.env.example` to `.env` and add your OpenAI API key:

   ```bash
   cp .env.example .env
   # Edit .env with your key
   ```

2. Pull the Ollama model (for `main.py`):
   ```bash
   ollama pull phi3
   ```

### Usage

#### Option A: CLI Chatbot (main.py)

```bash
python main.py
```

- First run: builds FAISS index (~30s)
- Subsequent runs: loads index from disk instantly
- Type your question, get an AI answer
- Type `exit` to quit

#### Option B: Web App (rag2.py)

```bash
streamlit run rag2.py
```

- Upload any PDF via the web UI
- First upload: builds and caches the vector index
- Subsequent queries: instant retrieval + streaming response

## 📁 Project Structure

```
rag-project/
├── main.py              # CLI chatbot (Ollama + FAISS)
├── rag2.py              # Streamlit web app (OpenAI + FAISS)
├── ragp.py              # Manual RAG example (Chroma + OpenAI)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
├── LICENSE              # MIT License
└── README.md            # This file
```

## ⚙️ Configuration

Edit these constants in `main.py` to customize:

| Variable        | Default                           | Description                  |
| --------------- | --------------------------------- | ---------------------------- |
| `PDF_PATH`      | `...Introduction_to_Python...pdf` | Path to your PDF             |
| `LLM_MODEL`     | `phi3`                            | Ollama model name            |
| `CHUNK_SIZE`    | `300`                             | Text chunk size              |
| `CHUNK_OVERLAP` | `50`                              | Overlap between chunks       |
| `TOP_K`         | `3`                               | Number of chunks to retrieve |

## 🛠️ Tech Stack

- [LangChain](https://python.langchain.com/) — LLM orchestration
- [FAISS](https://faiss.ai/) — vector similarity search
- [Ollama](https://ollama.com/) — local LLM inference
- [OpenAI](https://openai.com/) — cloud LLM API
- [Streamlit](https://streamlit.io/) — web UI
- [Sentence Transformers](https://www.sbert.net/) — embeddings

## 📝 Notes

- FAISS indexes are cached locally (`faiss_index/` for CLI, `faiss_streamlit_<hash>/` for web)
- Large PDFs may take a while to embed on first run
- For best results with `main.py`, use a GPU-enabled Ollama setup

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
