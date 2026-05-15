# SISTec Info Bot 🎓

RAG-based AI assistant for SISTec Gandhi Nagar, Bhopal.
Built with Gemini 1.5 Flash + FAISS + Streamlit.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. Enter your Gemini API key in the sidebar
2. Click "Build Knowledge Base" (embeds all college data into FAISS)
3. Ask any question about SISTec in the chat
4. Bot answers only from college documents and refuses out-of-scope questions

## RAG Pipeline
Document → Section Splitting → Sentence-aware Chunking → 
Gemini Embeddings → FAISS Index → Query Retrieval → Gemini Answer Generation

## Test Questions
- "What is the intake for CSE department?"
- "Who is the HOD of AI and Data Science?"
- "Does SISTec have a swimming pool?"
- "What are Kaizen training modules?"
