import ollama

from services.web_search import web_search
from services.fetch_paper import fetch_paper
from services.vectorstore import embed_and_store

from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")

class Researcher:
    def execute(self, plan: str):
        steps = plan.split("\n")

        notes = []

        for step in steps:
            low = step.lower()

            if "search" in low:
                notes.append(web_search(step))

            elif "paper" in low or "arxiv" in low:
                notes.append(fetch_paper(step))

            elif "embed" in low or "store" in low:
                embed_and_store("\n".join(notes))

        # Summarize and synthesize
        resp = ollama.chat(
            model = OLLAMA_MODEL, 
            messages = [
                {"role": "system", "content": "Summarize and synthesize the findings."},
                {"role": "user", "content": "\n\n".join(notes)}
            ]
        )
        
        return resp["message"]["content"]

