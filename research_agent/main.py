from agent.orchestrator import ResearchAgent


if __name__ == "__main__":
    agent = ResearchAgent()
    result = agent.run("Give me a summary of the latest research on small language models for edge deployment.")
    print(result)

