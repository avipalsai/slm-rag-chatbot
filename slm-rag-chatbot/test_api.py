import uvicorn
from src.api.routes import app

if __name__ == "__main__":
    print("🚀 Starting API server...")
    print("📍 Test at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
