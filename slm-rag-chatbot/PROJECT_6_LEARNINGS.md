What are you building?
BUILDING RAG BASED CHATBOT.. PH2/6
(yet to power SLM to it; right now it just picks the relevant chunk and displays. Have created html to have a good user interface)


What worked easily?
Some of the creations of API, Vector DB, linkages etc.

What was confusing?
Debugging was a drain on time and how is each piece contributing to the overall project was unclear unless I started focusing specifically on that aspect.

What would you do differently?
Not much change.. I'll still be holding hand of Perplexity to get work done.


Key technical concepts learned (RAG, embeddings, vector search, CORS, etc.)
These were learnt in Ankit's class not while building.. That session helped while building as I could connect the dots quickly.
RAG, Embeddings, Chunking, Storage in Vector DB, Vector search are all clear. CORS = Cross Origin Resource Sharing (for me I needed to use since browser wouldn't allow the api info since sources are diff)


DETAILS OF LEARNING::


Yesterday vs Today - Project Breakdown
Project: RAG Chatbot = 3 Core Parts
Storage Layer (Vector Database)

Retrieval Layer (Search Engine)

Generation Layer (Response Creator)

YESTERDAY (Setup + Storage):
What we built:

Project structure (folders: src/core, config, etc.)

Qdrant cloud account + API keys

Document processor (takes text → splits into chunks)

Vector store wrapper (sends chunks → Qdrant as embeddings)

Why it matters:
RAG needs a "smart database" that understands meaning, not just keywords. Vector embeddings convert text into numbers that capture semantic meaning. "refund policy" and "money back guarantee" have similar vectors even though words differ.

Analogy: Building a library's cataloging system before adding books.

TODAY (Retrieval + Generation + Interface):
Part 1: Retrieval Layer

Loaded sample documents into Qdrant (test data)

Built search function: question → embedding → find similar docs

Tested: "refund policy" correctly retrieves relevant chunks

Part 2: Generation Layer

Tried Llama 3.2 3B (too slow on CPU - 20 min/response)

Built template generator (fast fallback using retrieved context)

API endpoint combines: search results → template response

Part 3: User Interface

FastAPI backend with /query endpoint

HTML chat UI (browser-based)

CORS fix (let browser talk to API)

End-to-end test: typed question → got answer