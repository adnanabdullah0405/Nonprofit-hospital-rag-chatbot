# Nonprofit Hospital Research Chatbot – RAG-Powered with LLM 
A domain-specific RAG-based chatbot built using  LLM and vector DB. It allows users to explore a research paper on nonprofit hospitals through natural language queries, using semantic search and LLM-powered Q&amp;A — deployed via a Streamlit interface.

##  Overview

This project is a **Retrieval Augmented Generation (RAG)** based AI chatbot focused on the domain of **nonprofit hospitals**. It allows users to interact with a research paper discussing the importance and advantages of nonprofit healthcare institutions. The system uses **Gemini LLM**, **semantic search with sentence-transformers**, and **Astra DB** as the vector database — wrapped in a user-friendly **Streamlit app** with text-based Q&A.

---

##  Objective

To build a **domain-aware chatbot** that:
- Reads and understands a research paper on nonprofit hospitals
- Answers user queries using relevant research-backed information
- Encourages AI for **ethical and socially beneficial use cases**
- Is deployable and usable via a simple GUI

---

##  System Architecture (RAG Flow)

[PDF: Why Hospitals Should Be Nonprofit]
↓
[Sentence Transformer (Embedding)]
↓
[Vector Storage in Astra DB]
↓
[User Asks a Question]
↓
[Embed Question → Semantic Search → Retrieve Chunks]
↓
[Gemini LLM → Generate Contextual Answer]
↓
[Return Answer via Streamlit Interface]

---

##  Tech Stack

| Component | Technology |
|----------|------------|
| Embedding Model | Hugging Face (`all-MiniLM-L6-v2`) |
| Vector Database | **Astra DB (by DataStax)** |
| Language Model | **Gemini (Google Generative AI)** |
| User Interface | **Streamlit** |
| File Parsing | `PyMuPDF` or `pdfplumber` |
| Deployment | Streamlit Cloud / Hugging Face Spaces |

---

