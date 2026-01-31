# 📄 Conversational RAG Chatbot (PDF Q&A) — with Chat History(Practise based only no fancy shit)

A Streamlit-based **Retrieval-Augmented Generation (RAG)** app that lets you **upload one or multiple PDFs** and **chat with their content** — including **session-based chat history** for more contextual Q&A.

---

## ✨ Features

- 📚 Upload **multiple PDFs**
- 🔎 Splits documents into chunks + creates embeddings
- 🧠 Uses **history-aware retriever** for better follow-up questions
- 💬 Keeps **chat history per session**
- ⚡ Fast vector search using **ChromaDB**
- 🤖 LLM powered by **Groq (Llama 3.1 8B Instant)**

---

## 🧰 Tech Stack

- **UI:** Streamlit  
- **LLM:** Groq (`llama-3.1-8b-instant`)  
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)  
- **Vector DB:** ChromaDB  
- **Framework:** LangChain (modular + classic chains)  

---

## 📸 Demo

> Add screenshots/gif here after you run it once.

- **Upload PDFs**
- **Ask questions**
- **Follow-up questions work using chat history**

📷 **Screenshot placeholder:**  
`/assets/demo.png`

---

## 🚀 Run Locally

### 1) Clone the repository
```bash
git clone <YOUR_GITHUB_REPO_LINK>
cd <YOUR_REPO_FOLDER>
2) Install dependencies
pip install -r requirements.txt
3) Start the Streamlit app
streamlit run app.py
4) Use the app
Paste your Groq API key in the input box

Upload one or multiple PDFs

Start asking questions

☁️ Run on Google Colab (Streamlit + Tunnel)
Colab doesn’t directly expose Streamlit ports, so you run Streamlit + tunnel.

1) Install packages
%pip install -qU streamlit pypdf chromadb langchain-classic \
  langchain langchain-core langchain-community langchain-chroma \
  langchain-groq langchain-huggingface langchain-text-splitters
2) Create app.py inside Colab
%%writefile app.py
# paste your full code here
3) Run Streamlit
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
4) Expose via tunnel (Cloudflare)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

!cloudflared tunnel --url http://localhost:8501
Open the .trycloudflare.com link in your browser ✅

🧠 How It Works (High-Level)
Upload PDFs

Load PDF pages using PyPDFLoader

Split into chunks using RecursiveCharacterTextSplitter

Store chunks in Chroma vector database

Use history-aware retriever to convert follow-up questions into standalone questions

Retrieve relevant chunks → pass to LLM → answer with context

📂 Project Structure
.
├── app.py
├── requirements.txt
└── README.md
✅ Requirements
Python 3.10+

Groq API Key

🔐 Notes on API Key
Your Groq API key is entered in the Streamlit UI and used only during runtime.
Do not hardcode keys in source code for public repos.

📌 Future Improvements
✅ Better chat UI (st.chat_input, st.chat_message)

✅ Persist vectorstore per uploaded PDF (avoid re-embedding each rerun)

✅ Add source citations (page number + chunk references)

✅ Add “Clear chat / Reset session” button

✅ Support DOCX / TXT / URLs

🙌 Credits
Built using:

LangChain

Streamlit

ChromaDB

Groq API

HuggingFace embeddings

⭐ If you like this project
Star the repo ⭐ and feel free to open issues / suggestions!


If you tell me your **repo name** + **your GitHub username**, I can personalize the README (badges, correct clone URL, demo section, and a cleaner “About” line).
::contentReference[oaicite:0]{index=0}
