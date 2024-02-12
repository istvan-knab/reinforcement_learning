import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

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
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["ALPHA"], amsgrad=True)
        self.Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))

    def training_step(self, state, env, sum_reward, time_step):
        action = self.action_selection.epsilon_greedy_selection(self.model, state)
        observation, reward, terminated, truncated, _ = self.env.step(action)
        reward = torch.tensor([reward], device=self.device)
        sum_reward += reward
        done = terminated or truncated

        if done:
            next_state = None
            return True, sum_reward
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor([[action]], dtype=torch.long, device=self.device)
        self.memory.push(state, action, next_state, reward)
        state = next_state

        self.fit_model()
        if time_step % self.config["TAU"] == 0:
            self.update_target()


        return False, sum_reward


    def update_target(self):
        target_net_state_dict = self.target.state_dict()
        model_net_state_dict = self.model.state_dict()
        for key in model_net_state_dict.keys():
            target_net_state_dict[key] = (model_net_state_dict[key] * self.config["TAU"] +
                                          target_net_state_dict[key] * (1 - self.config["TAU"]))
        self.target.load_state_dict(target_net_state_dict)

    def fit_model(self):
        if len(self.memory) < self.config["BATCH_SIZE"]:
            return
        transitions = self.memory.sample()

        batch = self.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.model(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.config["BATCH_SIZE"], device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.config["GAMMA"]) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

