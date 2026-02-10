from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.core.vector_store import VectorStore
from src.core.generator_template import TemplateGenerator

app = FastAPI(title="SLM RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize components
retriever = VectorStore()
generator = TemplateGenerator()

class Query(BaseModel):
    question: str
    max_results: int = 3

class Response(BaseModel):
    answer: str
    sources: list

@app.post("/query", response_model=Response)
async def query_chatbot(query: Query):
    try:
        # Retrieve context
        results = retriever.search(query.question, top_k=query.max_results)
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant information found")
        
        # Generate response
        answer = generator.generate(query.question, results)
        
        return Response(
            answer=answer,
            sources=[r['text'][:100] + "..." for r in results]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
