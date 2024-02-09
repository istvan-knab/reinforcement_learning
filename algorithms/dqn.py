import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algorithms.components.epsilon_greedy import EpsilonGreedy
from algorithms.components.replay_memory import ReplayMemory


class NN(nn.Module):
    """
    Neural Network todo custom implementation
    """
    def __init__(self, config):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(config["state_size"], 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, config["action_size"])

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN(object):
    def __init__(self, config: dict, device: str , env: object) -> None:
        self.config = config
        self.env = env
        self.device = device
        self.action_selection = EpsilonGreedy(config)
        self.memory = ReplayMemory(config["MEMORY_SIZE"], config["BATCH_SIZE"])
        self.model = NN(config).to(self.device)
        self.target = NN(config).to(self.device)

    def training_step(self, state):
        action = self.action_selection.epsilon_greedy_selection()
        observation, reward, terminated, truncated, _ = self.env.step(action)
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated

        if terminated:
            next_state = None
            return True
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.memory.push(state, action, next_state, reward)
        state = next_state

        self.fit_model()

        target_net_state_dict = self.target.state_dict()
        model_net_state_dict = self.model.state_dict()
        for key in model_net_state_dict.keys():
            target_net_state_dict[key] = (model_net_state_dict[key] * self.config["TAU"] +
                                          target_net_state_dict[key] * (1 - self.config["TAU"]))
        self.target.load_state_dict(target_net_state_dict)




    def fit_model(self):
        pass

