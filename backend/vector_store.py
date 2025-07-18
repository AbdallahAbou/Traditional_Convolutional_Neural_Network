"""
Vector Store using ChromaDB with OpenAI embeddings
Uses text-embedding-3-large for optimal medical terminology representation
"""

import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Any, Optional
import httpx
import json
import re
import logging

logger = logging.getLogger(__name__)


class EmbeddingAPI:
    """OpenAI Embeddings API client"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.model_name = "text-embedding-3-large"
        self.dimension = 3072
        logger.info(f"Initialized OpenAI embeddings: {self.model_name} ({self.dimension} dims)")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via OpenAI API"""
        if not texts:
            return []
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "input": texts
                }
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
    
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text"""
        return self.embed([text])[0]


class Reranker:
    """Cross-encoder reranking using GPT-4o-mini for relevance scoring"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.model_name = "gpt-4o-mini"
        logger.info(f"Initialized reranker using {self.model_name}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder scoring.
        
        Args:
            query: The search query
            documents: List of document dicts with 'text' field
            top_n: Number of top results to return
        
        Returns:
            Reranked documents with relevance_score added
        """
        if not documents:
            return []
        
        if len(documents) <= top_n:
            for doc in documents:
                doc["relevance_score"] = doc.get("score", 0.5)
            return documents
        
        # Build reranking prompt
        doc_texts = []
        for i, doc in enumerate(documents):
            doc_texts.append(f"[{i}] {doc.get('text', '')[:500]}")
        
        prompt = f"""You are a medical document relevance scorer. Given a query and a list of document excerpts, 
score each document's relevance to the query on a scale of 0-100.

Query: {query}

Documents:
{chr(10).join(doc_texts)}

Return ONLY a JSON array of objects with "index" and "score" fields, ordered by relevance (highest first).
Return only the top {top_n} most relevant documents.
Example: [{{"index": 2, "score": 95}}, {{"index": 0, "score": 72}}]"""

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 500
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                content = data["choices"][0]["message"]["content"]
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    scores = json.loads(json_match.group())
                    
                    reranked = []
                    for item in scores[:top_n]:
                        idx = item.get("index", 0)
                        if 0 <= idx < len(documents):
                            doc = documents[idx].copy()
                            doc["relevance_score"] = item.get("score", 0) / 100.0
                            reranked.append(doc)
                    
                    return reranked
                    
        except Exception as e:
            logger.warning(f"Reranking failed, returning original order: {e}")
        
        # Fallback: return top_n by original score
        sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
        for doc in sorted_docs:
            doc["relevance_score"] = doc.get("score", 0.5)
        return sorted_docs[:top_n]


class VectorStore:
    """Vector store with OpenAI embeddings and reranking"""
    
    def __init__(self):
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", 8000))
        
        logger.info(f"Connecting to ChromaDB at {chroma_host}:{chroma_port}")
        
        self.client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.embedder = EmbeddingAPI()
        self.reranker = Reranker()
        self.model_name = self.embedder.model_name
        
        self.collection = self.client.get_or_create_collection(
            name="ehr_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"VectorStore initialized with {self.collection.count()} chunks")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings"""
        return self.embedder.embed(texts)
    
    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 20):
        """Add chunks to the vector store"""
        if not chunks:
            return
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [{
                "doc_id": c["doc_id"],
                "patient_id": c["patient_id"],
                "doc_type": c["doc_type"],
                "section": c.get("section", ""),
                "chunk_index": c.get("chunk_index", 0)
            } for c in batch]
            
            embeddings = self.embed(texts)
            self.collection.upsert(
                ids=ids, 
                embeddings=embeddings, 
                documents=texts, 
                metadatas=metadatas
            )
            logger.info(f"Indexed batch {i//batch_size + 1}: {len(batch)} chunks")
        
        logger.info(f"Total indexed: {len(chunks)} chunks")
    
    def search(
        self, 
        query: str, 
        patient_id: Optional[str] = None,
        doc_type: Optional[str] = None, 
        k: int = 50,
        rerank: bool = True,
        rerank_top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Multi-stage retrieval.
        
        Stage 1: Dense retrieval (k candidates)
        Stage 2: Cross-encoder reranking (top_n results)
        
        Args:
            query: Search query
            patient_id: Patient scope filter (None for global search)
            doc_type: Optional document type filter
            k: Number of candidates for stage 1
            rerank: Whether to apply reranking
            rerank_top_n: Number of results after reranking
        """
        # Build metadata filter
        where = None
        if patient_id and doc_type:
            where = {
                "$and": [
                    {"patient_id": {"$eq": patient_id}}, 
                    {"doc_type": {"$eq": doc_type}}
                ]
            }
        elif patient_id:
            where = {"patient_id": {"$eq": patient_id}}
        elif doc_type:
            where = {"doc_type": {"$eq": doc_type}}
        
        # Stage 1: Dense retrieval
        query_embedding = self.embedder.embed_single(query)
        results = self.collection.query(
            query_embeddings=[query_embedding], 
            where=where, 
            n_results=k, 
            include=["documents", "metadatas", "distances"]
        )
        
        # Format candidates
        candidates = []
        if results and results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                score = 1 - results['distances'][0][i]
                candidates.append({
                    "chunk_id": chunk_id,
                    "doc_id": results['metadatas'][0][i].get("doc_id", ""),
                    "patient_id": results['metadatas'][0][i].get("patient_id", ""),
                    "doc_type": results['metadatas'][0][i].get("doc_type", ""),
                    "section": results['metadatas'][0][i].get("section", ""),
                    "chunk_index": results['metadatas'][0][i].get("chunk_index", 0),
                    "text": results['documents'][0][i],
                    "score": round(score, 4)
                })
        
        if not candidates:
            return []
        
        # Stage 2: Reranking
        if rerank and len(candidates) > rerank_top_n:
            reranked = self.reranker.rerank(query, candidates, top_n=rerank_top_n)
            return reranked
        
        return candidates[:rerank_top_n]
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks"""
        return self.collection.count()
    
    def delete_patient_chunks(self, patient_id: str):
        """Delete all chunks for a patient"""
        self.collection.delete(where={"patient_id": {"$eq": patient_id}})
    
    def clear_all(self):
        """Clear all chunks and recreate collection"""
        try:
            self.client.delete_collection("ehr_chunks")
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name="ehr_chunks", 
            metadata={"hnsw:space": "cosine"}
        )
