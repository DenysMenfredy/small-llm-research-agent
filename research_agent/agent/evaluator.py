from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from research_agent.config import settings
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    SYSTEM_PROMPT = """
    You are a critical evaluator.
    Identify missing context, errors, and unclear reasoning.
    Then rewrite the answer with improvements.
    """

    def __init__(self):
        self.llm = ChatOllama(model=settings.ollama_model, temperature=0.1)

    def refine_answer(self, query: str, answer: str) -> str:
        prompt = PromptTemplate.from_template(
            self.SYSTEM_PROMPT + "\n\nQuery: {query}\n\nAnswer: {answer}"
        )
        try:
            response = self.llm.invoke(prompt.format(query=query, answer=answer))
            return response.content.strip()
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return answer