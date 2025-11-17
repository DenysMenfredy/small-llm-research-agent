import ollama
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")

class Evaluator:
    SYSTEM_PROMPT = """
    You are a critical evaluator. 
    Identify missing context, errors, and unclear reasoning.
    Then rewrite the answer with improvements.
    """

    def refine_answer(self, query: str, answer: str):
        resp = ollama.chat(
            model = OLLAMA_MODEL, 
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Query:\n{query}\n\nAnswer:\n{answer}"}
            ]
        )

        return resp["message"]["content"]
