from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    QDRANT_URL: str = ""
    QDRANT_API_KEY: Optional[str] = None
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"

settings = Settings()
