from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from research_agent.agent.tools import web_search_tool, fetch_paper_tool, embed_store_tool
from research_agent.config import settings
import logging

logger = logging.getLogger(__name__)

class Researcher:
    SYSTEM_PROMPT = """
    You are a research assistant. Based on the plan, decide which tool to use and what query to pass.
    Output the tool name and the query, separated by a colon.
    Available tools: web_search, fetch_paper, embed_store
    """

    def __init__(self):
        self.llm = ChatOllama(model=settings.ollama_model, temperature=0.2)
        self.tools = {
            "web_search": web_search_tool,
            "fetch_paper": fetch_paper_tool,
            "embed_store": embed_store_tool
        }

    def execute(self, plan: str) -> str:
        notes = []
        steps = plan.split("\n")
        for step in steps:
            if not step.strip():
                continue
            prompt = PromptTemplate.from_template(self.SYSTEM_PROMPT + "\n\nStep: {step}")
            try:
                decision = self.llm.invoke(prompt.format(step=step)).content.strip()
                if ":" in decision:
                    tool_name, query = decision.split(":", 1)
                    tool_name = tool_name.strip()
                    query = query.strip()
                    if tool_name in self.tools:
                        result = self.tools[tool_name].invoke({"input": query})
                        notes.append(result)
                    else:
                        notes.append(f"Unknown tool: {tool_name}")
                else:
                    notes.append(f"Could not parse decision: {decision}")
            except Exception as e:
                logger.error(f"Error in step '{step}': {e}")
                notes.append(f"Error in step: {step}")

        # Summarize
        summary_prompt = PromptTemplate.from_template("Summarize and synthesize the following research notes:\n\n{notes}")
        try:
            summary = self.llm.invoke(summary_prompt.format(notes="\n\n".join(notes))).content.strip()
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "\n\n".join(notes)
