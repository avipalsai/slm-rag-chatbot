from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import uuid
import sys
sys.path.insert(0, 'C:\\Users\\avina\\Desktop\\AI Projects\\slm-rag-chatbot')
from config.settings import settings

class VectorStore:
    """Production-ready Qdrant vector store wrapper"""
    
    def __init__(self, collection_name: str = "client_docs"):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = collection_name
        self.embed_model = SentenceTransformer(settings.EMBED_MODEL)
        self.embedding_dim = 384
    
    def create_collection(self) -> bool:
        """Initialize collection if not exists"""
        try:
            self.client.get_collection(self.collection_name)
            print(f"✓ Collection '{self.collection_name}' already exists")
            return False
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Created collection '{self.collection_name}'")
            return True
    
    def add_documents(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> int:
        """Add documents to vector store"""
        if not texts:
            return 0
        
        print(f"Embedding {len(texts)} documents...")
        embeddings = self.embed_model.encode(texts, show_progress_bar=True)
        
        points = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            payload = {
                "text": text,
                "source": "client_docs",
                "chunk_index": i
            }
            
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb.tolist(),
                    payload=payload
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✓ Added {len(points)} documents to '{self.collection_name}'")
        return len(points)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """Search for relevant documents"""
        query_vector = self.embed_model.encode(query)
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k
        ).points
        
        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in results
            if hit.score >= score_threshold
        ]
    
    def delete_collection(self) -> None:
        """Delete collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✓ Deleted collection '{self.collection_name}'")
        except:
            print(f"Collection '{self.collection_name}' does not exist")
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "status": "ready"
            }
        except:
            return {
                "name": self.collection_name,
                "vectors_count": 0,
                "status": "not_found"
            }
