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

class Memory:
    """
    Memory module with vector retrieval using LangChain Chroma.
    Stores each interaction as a text document and embeds it.
    """

    def save_interaction(self, query: str, answer: str):
        text = f"QUERY:\n{query}\n\nANSWER:\n{answer}"
        try:
            vectorstore.add_texts([text], ids=[str(uuid.uuid4())], metadatas=[{"type": "interaction"}])
        except Exception as e:
            logger.error(f"Save interaction failed: {e}")

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """
        Retrieve the top-k most relevant past interactions.
        Returns a list of text snippets.
        """
        try:
            docs = vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Retrieve failed: {e}")
            return []