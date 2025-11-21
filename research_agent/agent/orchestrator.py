from research_agent.graph import compiled_graph
import logging

logger = logging.getLogger(__name__)

class ResearchAgent:
    async def run(self, query: str) -> str:
        try:
            result = await compiled_graph.ainvoke({"query": query})
            return result["final_answer"]
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            return "An error occurred during research."