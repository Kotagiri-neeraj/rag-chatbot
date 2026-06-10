# RAG Chatbot

## What is this project?
This repository implements a Retrieval-Augmented Generation (RAG) chatbot using a FAISS vector database and a local Hugging Face language model pipeline. It ingests documents, creates a FAISS vector store, and answers user questions by retrieving relevant context and generating responses.
                            or
This project implements a Retrieval-Augmented Generation (RAG) chatbot using a FAISS vector database and a local Hugging Face language model. Documents are ingested, converted into embeddings, stored in a FAISS index, and retrieved at query time to provide context-aware answers through a Streamlit interface.

## Why it is useful
- Enables local question-answering over documents without requiring cloud API usage.
- Uses a local vector store for fast retrieval of relevant information.
- Supports document ingestion and retrieval-based generation with a Streamlit user interface.
- Helps build a searchable, conversational AI assistant for internal or offline workflows.
- Uses Retrieval-Augmented Generation (RAG) to improve answer accuracy by grounding responses in retrieved document content.

## Project stack and languages
- Python 3.14.3
- Streamlit
- LangChain and LangChain Community
- FAISS (via `faiss-cpu`)
- Transformers
- Hugging Face model pipeline
- PDF processing with `pypdf`
- Environment management via `venv`

### Versions
The project is designed for a Python 3.14 environment. The exact package versions are managed inside `requirements.txt` and can be installed with pip.

## Key files
- `app.py` - Streamlit app for the chatbot UI.
- `src/chatbot.py` - Builds the retriever and local LLM pipeline.
- `src/vectorstore.py` - Loads and saves the FAISS vector store.
- `ingest.py` - Reads documents and builds the vector store.
- `requirements.txt` - Python dependencies for the project.
- `vectorstore/` - Persistent index data used by the app.


## Model Used

This project uses the `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model from Hugging Face for local text generation and question answering.

### About TinyLlama

TinyLlama is a lightweight Large Language Model (LLM) designed to provide good performance while requiring significantly fewer computational resources than larger models. The model contains approximately 1.1 billion parameters and is optimized for conversational and instruction-following tasks.

### Key Features

- **1.1 Billion Parameters** – Small enough to run on consumer hardware while still providing useful responses.
- **Chat Optimized** – Fine-tuned for conversational interactions and question answering.
- **Open Source** – Freely available through Hugging Face.
- **Efficient Resource Usage** – Requires less memory and processing power compared to larger models such as Llama 2, Llama 3, or GPT-style models.
- **Suitable for Local Deployment** – Can be used without relying on external APIs or cloud services.

### Role in This Project

In this RAG chatbot, TinyLlama receives the user's question along with the relevant document context retrieved from the FAISS vector store. The model then generates a context-aware answer based on the retrieved information.

### Why TinyLlama Was Chosen

- Supports fully local execution.
- Eliminates dependency on paid API services.
- Works well for small to medium-sized RAG applications.
- Provides a good balance between response quality and hardware requirements.
- Suitable for educational, demonstration, and portfolio projects.

## How to run this project
1. Open a terminal in the project folder:

```powershell
cd C:\Users\HP\Desktop\rag-chatbot
```

2. Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

3. Install dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

4. Ingest documents into the vector store (if you have new or updated PDFs in `data/`):

```powershell
python ingest.py
```

5. Run the Streamlit app:

```powershell
python -m streamlit run app.py
```

6. Open the local Streamlit URL shown in the terminal to use the chatbot.

## How to use the chatbot
1. Enter a question in the text input field.
2. The app searches the vector store for relevant document chunks.
3. The system generates an answer using the retrieved context.
4. The answer appears on the page, along with the source documents used for retrieval.

## Notes
- The app is configured to use a local Hugging Face model pipeline.
- The vector store is stored in `vectorstore/` and should remain available for the app to load properly.

## Architecture

User Question
      ↓
FAISS Vector Store
      ↓
Relevant Document Chunks
      ↓
Local Hugging Face LLM
      ↓
Generated Answer
