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
            plan = self.planner.create_plan(query)
            print("PLAN":, plan)

            raw_results = self.researcher.execute(plan)
            improved_results = self.evaluator.refine_answer(query, raw_results)

            self.memory.save_interaction(query, improved_results)

            return improved_results
