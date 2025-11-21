from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from research_agent.config import settings

class Planner:
    SYSTEM_PROMPT = """
    You are a research planner. Break the user's query into
    3–7 actionable, structured steps. Steps should be concise
    and follow a logical order. Output only the steps, numbered.
    """

    def __init__(self):
        self.llm = ChatOllama(model=settings.ollama_model, temperature=0.1)

    def create_plan(self, query: str) -> str:
        prompt = PromptTemplate.from_template(self.SYSTEM_PROMPT + "\n\nQuery: {query}")
        response = self.llm.invoke(prompt.format(query=query))
        return response.content.strip()