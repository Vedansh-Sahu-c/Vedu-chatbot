import os
import io
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import google.generativeai as genai
import faiss
import numpy as np
from pypdf import PdfReader

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    section_name: str

@dataclass
class RetrievalResult:
    answer: str
    chunks: List[Chunk]
    scores: List[float]
    is_out_of_scope: bool
    query: str

class RAGEngine:
    def __init__(self, api_key: str = None):
        # Initialize Google Generative AI with the provided API key if passed
        if api_key:
            genai.configure(api_key=api_key)
        self.embedding_model = "models/text-embedding-001"
        self.generation_model = "gemini-1.5-flash"
        self.index = None
        self.chunks = []
        
    def load_text_file(self, path: str, source_name: str) -> List[Tuple[str, str]]:
        """Loads .txt with section splitting based on --- SECTION: markers"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        blocks = []
        parts = re.split(r'---\s*SECTION:\s*(.*?)---', content)
        
        if len(parts) == 1:
            blocks.append((source_name, content.strip()))
        else:
            # First part might be a preamble before the first section
            if parts[0].strip():
                blocks.append((f"{source_name} - Preamble", parts[0].strip()))
                
            for i in range(1, len(parts), 2):
                section_name = parts[i].strip()
                section_text = parts[i+1].strip() if i+1 < len(parts) else ""
                if section_text:
                    blocks.append((f"{source_name} - {section_name}", section_text))
                    
        return blocks

    def load_uploaded_file(self, file_bytes: bytes, filename: str) -> List[Tuple[str, str]]:
        """Handles PDF and TXT uploads and extracts text."""
        blocks = []
        if filename.lower().endswith('.pdf'):
            reader = PdfReader(io.BytesIO(file_bytes))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    blocks.append((f"{filename} - Page {i+1}", text))
        elif filename.lower().endswith('.txt'):
            text = file_bytes.decode('utf-8')
            blocks.append((filename, text))
            
        return blocks

    def chunk_documents(self, doc_blocks: List[Tuple[str, str]], chunk_size: int = 350, overlap: int = 50) -> List[Chunk]:
        """Sentence-aware chunking based on word count with overlap."""
        chunks = []
        chunk_id = 0
        
        for source_label, text in doc_blocks:
            # Split on sentences based on period and spaces, or double newline
            sentences = re.split(r'(?<=\.)\s+|\n\n+', text)
            
            current_chunk_words = []
            current_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence_words = sentence.split()
                # Check if adding this sentence exceeds chunk_size
                if len(current_chunk_words) + len(sentence_words) > chunk_size and current_chunk_words:
                    # Save current chunk
                    chunk_text = " ".join(current_sentences)
                    source_base = source_label.split(" - ")[0] if " - " in source_label else source_label
                    section = source_label.split(" - ", 1)[1] if " - " in source_label else ""
                    
                    chunks.append(Chunk(
                        text=chunk_text,
                        source=source_base,
                        chunk_id=chunk_id,
                        section_name=section
                    ))
                    chunk_id += 1
                    
                    # Create overlap from the end of the previous chunk words
                    overlap_words = current_chunk_words[-overlap:] if overlap > 0 else []
                    overlap_text = " ".join(overlap_words)
                    
                    current_chunk_words = overlap_words + sentence_words
                    current_sentences = [overlap_text, sentence] if overlap_text else [sentence]
                else:
                    current_chunk_words.extend(sentence_words)
                    current_sentences.append(sentence)
                    
            # Add remaining words as a chunk
            if current_chunk_words:
                chunk_text = " ".join(current_sentences)
                source_base = source_label.split(" - ")[0] if " - " in source_label else source_label
                section = source_label.split(" - ", 1)[1] if " - " in source_label else ""
                
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source_base,
                    chunk_id=chunk_id,
                    section_name=section
                ))
                chunk_id += 1
                
        return chunks

    def _get_embeddings(self, texts: List[str], task_type: str) -> np.ndarray:
        """Fetch embeddings from Gemini and apply L2 normalization."""
        all_embeddings = []
        batch_size = 20 # Batch size as requested
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = genai.embed_content(
                model=self.embedding_model,
                content=batch,
                task_type=task_type
            )
            # Response contains a list of embeddings under the 'embedding' key
            all_embeddings.extend(response['embedding'])
            
        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        # L2 normalize all vectors for cosine similarity (IndexFlatIP will now compute cosine)
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norms[norms == 0] = 1 # Avoid division by zero
        embeddings_np = embeddings_np / norms
        
        return embeddings_np

    def build_faiss_index(self, embeddings: np.ndarray):
        """Builds a FAISS index using Inner Product."""
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

    def build(self, doc_blocks: List[Tuple[str, str]], chunk_size: int, overlap: int) -> Dict[str, Any]:
        """Runs the full RAG pipeline: chunking -> embedding -> FAISS."""
        self.chunks = self.chunk_documents(doc_blocks, chunk_size, overlap)
        if not self.chunks:
            return {"total_chunks": 0, "embedding_dim": 0, "index_size": 0}
            
        texts = [c.text for c in self.chunks]
        embeddings = self._get_embeddings(texts, task_type="retrieval_document")
        self.build_faiss_index(embeddings)
        
        return {
            "total_chunks": len(self.chunks),
            "embedding_dim": embeddings.shape[1],
            "index_size": self.index.ntotal
        }

    def retrieve(self, query: str, top_k: int = 4) -> Tuple[List[Chunk], List[float]]:
        """Retrieve top-k chunks for the given query."""
        if not self.index:
            return [], []
            
        query_emb = self._get_embeddings([query], task_type="retrieval_query")
        scores, indices = self.index.search(query_emb, top_k)
        
        retrieved_chunks = []
        retrieved_scores = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
                retrieved_scores.append(float(scores[0][i]))
                
        return retrieved_chunks, retrieved_scores

    def generate_answer(self, query: str, top_k: int = 4) -> RetrievalResult:
        """Retrieves context and generates an answer using Gemini 1.5 Flash."""
        chunks, scores = self.retrieve(query, top_k)
        
        # Out-of-scope detection
        if not chunks or (scores and scores[0] < 0.30):
            return RetrievalResult(
                answer="OUT_OF_SCOPE: I don't have information about that in the SISTec knowledge base.",
                chunks=chunks,
                scores=scores,
                is_out_of_scope=True,
                query=query
            )
            
        # Prepare context with source labels
        context_parts = []
        for chunk in chunks:
            source_info = chunk.source
            if chunk.section_name:
                source_info += f" - {chunk.section_name}"
            context_parts.append(f"[Source: {source_info}]\n{chunk.text}")
            
        context_str = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f'''You are SISTec Info Bot. Answer ONLY from the provided context.
If context is insufficient, respond starting with OUT_OF_SCOPE:

Context:
{context_str}

Query: {query}'''

        model = genai.GenerativeModel(self.generation_model)
        try:
            response = model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            
        # Parse out-of-scope case returned by model
        is_out_of_scope = answer.strip().startswith("OUT_OF_SCOPE")
        
        return RetrievalResult(
            answer=answer,
            chunks=chunks,
            scores=scores,
            is_out_of_scope=is_out_of_scope,
            query=query
        )
