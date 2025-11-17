import ollama
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")

class Planner:
    SYSTEM_PROMPT = """
    You are a research planner. Break the user's query into 
    3–7 actionable, structured steps. Steps should be concise 
    and follow a logical order.
    """

    def create_plan(self, query: str):
        response = ollama.chat(
            model = OLLAMA_MODEL,
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {query}"}
            ]
        )

        return response["message"]["content"]
