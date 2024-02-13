from algorithms.RANDOM import RandomAgent
def agent_selection(agent: str) -> object:
    if agent == "random":
        return RandomAgent()

