# vector_store.py
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Using a multilingual model that works well for Arabic and English
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.dimension = 768  # Default for many transformer models
        
    async def add_document(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the vector store."""
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        self.chunks = chunks
        
        try:
            # Extract text from chunks for embedding
            texts = [chunk["text"] for chunk in chunks]
            
            if not texts:
                logger.warning("No texts to embed")
                return 0
                
            # Generate embeddings
            embeddings = model.encode(texts, show_progress_bar=True)
            
            # Create FAISS index
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
            
            # Add embeddings to index
            embeddings_np = np.array(embeddings).astype('float32')
            self.index.add(embeddings_np)
            
            logger.info(f"Successfully created index with dimension {self.dimension}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    async def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant chunks based on query."""
        if self.index is None:
            logger.warning("No index available for search")
            return []
        
        if not self.chunks:
            logger.warning("No chunks available in vector store")
            return []
            
        try:
            # Generate query embedding
            query_embedding = model.encode([query])
            query_embedding_np = np.array(query_embedding).astype('float32')
            
            # Use basic search method
            distances, indices = self.index.search(query_embedding_np, k)
            
            # Debug output
            logger.info(f"Search found indices: {indices[0]}, distances: {distances[0]}")
            
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx == -1:  # FAISS sometimes returns -1 for no match
                    continue
                    
                if 0 <= idx < len(self.chunks):  # Ensure index is valid
                    chunk = self.chunks[idx]
                    results.append({
                        "chunk": chunk["text"],
                        "metadata": chunk["metadata"],
                        "score": float(distances[0][i])
                    })
            
            logger.info(f"Search query: '{query}' returned {len(results)} results")
            
            # Log the actual chunks for debugging
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: Score={result['score']}, Content={result['chunk'][:100]}...")
                
            return results
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []