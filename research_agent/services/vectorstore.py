from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from research_agent.config import settings
import uuid
import logging

logger = logging.getLogger(__name__)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    collection_name="research_memory",
    embedding_function=embeddings,
    persist_directory=settings.chroma_path
)

def embed(text: str):
    """
    Get vector embedding from Ollama.
    """
    try:
        return embeddings.embed_query(text)
    except Exception as e:
        logger.error(f"Embed failed: {e}")
        return []

def embed_and_store(text: str, metadata: dict = None):
    """
    Stores text chunk(s) with vector embeddings into chromadb.
    """
    try:
        vectorstore.add_texts([text], ids=[str(uuid.uuid4())], metadatas=[metadata or {}])
    except Exception as e:
        logger.error(f"Embed and store failed: {e}")

def search(query: str, k: int = 5):
    """
    Retrieve relevant documents from the vector store.
    """
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return [{"document": doc.page_content, "metadata": doc.metadata, "distance": 0} for doc in docs]  # distance not available
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []