# Smart Research Paper Summarizer & Q&A System

A web application that allows users to upload research papers (PDF), extract their text content, generate abstractive and extractive summaries, and interactively ask questions about the paper’s content using retrieval-augmented question answering.

---

## Features

- **PDF to Text Extraction:** Upload research papers in PDF format and extract their full text.
- **Abstractive + Extractive Summarization:** Generate concise summaries capturing the key points.
- **Retrieval-Based Q&A:** Ask questions about the paper; get precise answers with source excerpts.
- **Semantic Search:** Uses vector embeddings and FAISS for efficient similarity search.
- **Conversation Memory:** Maintains chat history for contextual QA.

---

## Tech Stack

- **Backend:** Python, Flask
- **PDF Parsing:** PyMuPDF / PDFMiner
- **NLP Models:** LangChain with Google Generative AI (Gemini), BioBERT (optional for domain-specific tasks)
- **Vector Search:** FAISS
- **Frontend:** Static HTML, CSS, Vanilla JavaScript

---

