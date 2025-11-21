from langchain.tools import tool
from research_agent.services.web_search import web_search
from research_agent.services.fetch_paper import fetch_paper
from research_agent.services.vectorstore import embed_and_store
import logging

logger = logging.getLogger(__name__)

@tool
def web_search_tool(query: str) -> str:
    """Search the web for information related to the query."""
    try:
        return web_search(query)
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return "Web search failed."

@tool
def fetch_paper_tool(query: str) -> str:
    """Fetch and summarize a research paper based on the query."""
    try:
        return fetch_paper(query)
    except Exception as e:
        logger.error(f"Paper fetch failed: {e}")
        return "Paper fetch failed."

@tool
def embed_store_tool(text: str) -> str:
    """Embed and store text in the vector database."""
    try:
        embed_and_store(text)
        return "Text embedded and stored successfully."
    except Exception as e:
        logger.error(f"Embed store failed: {e}")
        return "Embed store failed."