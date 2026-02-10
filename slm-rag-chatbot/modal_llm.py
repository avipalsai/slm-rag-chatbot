import modal
import os

app = modal.App("llm-rag-generator")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "transformers",
        "torch",
        "accelerate",
        "qdrant-client",
        "sentence-transformers",
    )
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=300,
    scaledown_window=120,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("qdrant")
    ],
)
@modal.fastapi_endpoint(method="POST")
def generate(request: dict):
    try:
        from transformers import pipeline
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        
        # Get question
        question = request.get("question", "")
        max_results = request.get("max_results", 3)
        
        # Connect to Qdrant
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_key = os.environ.get("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_key:
            return {"error": "Qdrant credentials not configured"}
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        
        # Embed question
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        query_vector = embedder.encode(question).tolist()
       
        
        # Search Qdrant
        results = client.query_points(
            collection_name="hello_world_test",
            query=query_vector,
            limit=max_results
        ).points

        
        # Extract context
        context = "\n\n".join([hit.payload.get("text", "") for hit in results])
        sources = [hit.payload.get("text", "")[:100] + "..." for hit in results]
        
        # Build prompt
        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate with Llama 3.2
        llm = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-3B-Instruct",
            device_map="auto"
        )

        output = llm(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        answer = output[0]["generated_text"].split("Answer:")[-1].strip()
        
        # Clean up - take only first sentence or paragraph
        answer = answer.split("\n\n")[0].strip()
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }


