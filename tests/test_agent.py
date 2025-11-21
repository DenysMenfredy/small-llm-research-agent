from research_agent.agent.orchestrator import ResearchAgent

def test_agent_run():
    agent = ResearchAgent()
    # Mock or skip if no Ollama
    result = agent.run("Test query")
    assert isinstance(result, str)