from src.core.generator_template import TemplateGenerator as ResponseGenerator
from src.core.vector_store import VectorStore

print("Step 1: Loading vector store...")
store = VectorStore('test_phase2')

print("Step 2: Searching for context...")
context = store.search('What is the refund policy?')
print(f"Found {len(context)} results")

print("Step 3: Loading Llama (this takes 2-3 min)...")
gen = ResponseGenerator()

print("Step 4: Generating answer (this may take 1-2 min on CPU)...")
answer = gen.generate('What is your refund policy?', context, max_tokens=50)

print(f"\n{'='*60}")
print(f"Answer: {answer}")
print(f"{'='*60}")
