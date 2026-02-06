
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import sys
sys.path.insert(0, 'C:\\Users\\avina\\Desktop\\AI Projects\\slm-rag-chatbot')
from config.settings import settings

# Test documents
docs = [
    "Our refund policy is 30 days no questions asked.",
    "We support Python, JavaScript, and Go integrations.",
    "Pricing starts at $99/month for the professional plan.",
    "Enterprise customers get dedicated support and SLA.",
    "Our API has 99.9% uptime guarantee."
]

print("Loading embedding model...")
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("Embedding documents...")
embeddings = embed_model.encode(docs)
print(f"Embedding dimension: {embeddings.shape[1]}")

print(f"\nConnecting to Qdrant Cloud...")
print(f"URL: {settings.QDRANT_URL[:50]}...")
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

collection_name = "hello_world_test"

print(f"Creating collection '{collection_name}'...")
try:
    client.delete_collection(collection_name)
except:
    pass

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

print("Inserting documents...")
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=emb.tolist(),
        payload={"text": doc, "doc_id": i}
    )
    for i, (doc, emb) in enumerate(zip(docs, embeddings))
]
client.upsert(collection_name=collection_name, points=points)

print("\n" + "="*60)
print("TESTING RETRIEVAL")
print("="*60)

# Test queries
test_queries = [
    "What's your refund policy?",
    "How much does it cost?",
    "What programming languages do you support?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    query_vector = embed_model.encode(query)
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=2
    ).points
    
    print(f"Top Result: {results[0].payload['text']}")
    print(f"Score: {results[0].score:.3f}")

print("\n" + "="*60)
print("✅ HELLO WORLD RAG TEST PASSED!")
print("="*60)

