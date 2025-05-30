import os

from agentsystem import AgentSystem

if __name__ == "__main__":
    agent = AgentSystem(
        model="qwen/qwen3-235b-a22b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTEREGORGOOGLE"),
        temperature=0.7,
        top_p=0.8,
    )
    query = "Сделай грустную историю о потерянной любви"

    story = agent.process_agent_system(
        query=query,
    )
    print(f"Generated Story: {story}")
