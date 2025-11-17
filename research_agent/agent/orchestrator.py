from agent.planner import Planner
from agent.researcher import Researcher
from agent.evaluator import Evaluator
from agent.memory import Memory

class ResearchAgent:
    def __init__(self):
        self.planner = Planner()
        self.researcher = Researcher()
        self.evaluator = Evaluator()
        self.memory = Memory()

    def run(self, query: str):
        # 1. Retrieve relevant memory
        recalled = self.memory.retrieve(query)
        memory_text = "\n\n--- MEMORY ---\n\n".join(recalled) if recalled else "No relevant memory found."

        # 2. Generate plan using query + memory context
        plan = self.planner.create_plan(
            f"{query}\n\nRelevant past knowledge:\n{memory_text}"
        )
        print("PLAN:", plan)

        # 3. Execute research steps
        raw_results = self.researcher.execute(plan)

        # 4. Evaluate and improve results
        improved_results = self.evaluator.refine_answer(query, raw_results)

        # 5. Store the new result in vector memory
        self.memory.save_interaction(query, improved_results)

        return improved_results
