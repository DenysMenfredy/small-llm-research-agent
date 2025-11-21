from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import logging
import os

load_dotenv()

class Settings(BaseSettings):
    ollama_model: str = "llama3.2"
    chroma_path: str = "./chroma"
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: int = 30

    class Config:
        env_file = ".env"

settings = Settings()

# Setup logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)