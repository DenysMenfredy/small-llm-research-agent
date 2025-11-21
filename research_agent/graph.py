from langgraph.graph import StateGraph, END
from typing import TypedDict
from research_agent.agent.planner import Planner
from research_agent.agent.researcher import Researcher
from research_agent.agent.evaluator import Evaluator
from research_agent.agent.memory import Memory
import logging
import asyncio

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    query: str
    memory: list[str]
    plan: str
    results: str
    final_answer: str

async def retrieve_memory(state: AgentState) -> AgentState:
    memory = Memory()
    recalled = await asyncio.get_event_loop().run_in_executor(None, memory.retrieve, state["query"])
    state["memory"] = recalled
    return state

async def plan_step(state: AgentState) -> AgentState:
    planner = Planner()
    context = f"{state['query']}\n\nRelevant past knowledge:\n" + "\n\n--- MEMORY ---\n\n".join(state["memory"]) if state["memory"] else "No relevant memory found."
    plan = await asyncio.get_event_loop().run_in_executor(None, planner.create_plan, context)
    state["plan"] = plan
    logger.info(f"Plan: {plan}")
    return state

async def research_step(state: AgentState) -> AgentState:
    researcher = Researcher()
    results = await asyncio.get_event_loop().run_in_executor(None, researcher.execute, state["plan"])
    state["results"] = results
    return state

async def evaluate_step(state: AgentState) -> AgentState:
    evaluator = Evaluator()
    improved = await asyncio.get_event_loop().run_in_executor(None, evaluator.refine_answer, state["query"], state["results"])
    state["final_answer"] = improved
    return state

async def save_memory(state: AgentState) -> AgentState:
    memory = Memory()
    await asyncio.get_event_loop().run_in_executor(None, memory.save_interaction, state["query"], state["final_answer"])
    return state

# Define the graph
graph = StateGraph(AgentState)
graph.add_node("retrieve_memory", retrieve_memory)
graph.add_node("plan", plan_step)
graph.add_node("research", research_step)
graph.add_node("evaluate", evaluate_step)
graph.add_node("save_memory", save_memory)

graph.set_entry_point("retrieve_memory")
graph.add_edge("retrieve_memory", "plan")
graph.add_edge("plan", "research")
graph.add_edge("research", "evaluate")
graph.add_edge("evaluate", "save_memory")
graph.add_edge("save_memory", END)

compiled_graph = graph.compile()