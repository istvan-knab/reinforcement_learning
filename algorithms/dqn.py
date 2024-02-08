import torch
from algorithms.components.epsilon_greedy import EpsilonGreedy
from algorithms.components.replay_memory import ReplayMemory

class DQN(object):
    def __init__(self, config: dict, device: str , env: object) -> None:
        self.config = config
        self.env = env
        self.device = device
        self.action_selection = EpsilonGreedy(config)
        self.memory = ReplayMemory(config["MEMORY_SIZE"], config["BATCH_SIZE"])

    def training_step(self, state):
        action = self.action_selection.epsilon_greedy_selection()
        observation, reward, terminated, truncated, _ = self.env.step(action)
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.memory.push(state, action, next_state, reward)

        self.env.render()



    def fit_model(self):
        pass