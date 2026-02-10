from src.core.vector_store import VectorStore
from src.core.document_processor import DocumentProcessor

print("Step 1: Creating vector store...")
store = VectorStore()
store.create_collection()

print("\nStep 2: Processing sample document...")
processor = DocumentProcessor()

sample_doc = """
Product Documentation

Our SaaS platform helps teams collaborate effectively.

Features:
- Real-time messaging
- File sharing
- Video conferencing
- Task management

Pricing:
Starter: $29/month
Professional: $99/month
Enterprise: Custom pricing

Refund Policy:
We offer a 30-day money-back guarantee, no questions asked.
"""

chunks = processor.chunk_text(sample_doc)
print(f"Created {len(chunks)} chunks")

print("\nStep 3: Adding to vector store...")
count = store.add_documents(chunks)
print(f"\n✓ Successfully added {count} documents!")

print("\nStep 4: Testing search...")
results = store.search("refund policy", top_k=2)
print(f"Found {len(results)} results")
for i, result in enumerate(results):
    print(f"\nResult {i+1} (score: {result['score']:.3f}):")
    print(result['text'][:100] + "...")
