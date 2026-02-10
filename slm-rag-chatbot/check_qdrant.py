from qdrant_client import QdrantClient

# Explicitly set your credentials here
QDRANT_URL = "https://7734f728-3020-4e65-be22-06150396403f.us-east-1-1.aws.cloud.qdrant.io"  
QDRANT_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Oz_6g5aLJhglgbYSr-OVzhxePEgMaaAnnJKZ_LoTdF4"      

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)

# List collections
collections = client.get_collections()
print("Collections:", [c.name for c in collections.collections])

# Check documents collection
try:
    info = client.get_collection("documents")
    print(f"\nCollection 'documents' has {info.points_count} points")
except Exception as e:
    print(f"\nError with 'documents' collection: {e}")
