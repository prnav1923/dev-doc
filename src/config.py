import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "devdocs-navigator")
    
    # Model Configuration
    MODEL_NAME = "gemini-1.5-flash"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Vector Store Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "faiss_index")
    
    @staticmethod
    def validate():
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
