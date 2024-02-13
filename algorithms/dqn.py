import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from collections import OrderedDict

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
        self.criterion = nn.MSELoss()
        self.action_selection = EpsilonGreedy(config)
        self.memory = ReplayMemory(config["MEMORY_SIZE"], config["BATCH_SIZE"])
        self.model = NN(config).to(self.device)
        self.target = NN(config).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["ALPHA"], amsgrad=True)
        self.Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward', 'done'))

    def training_step(self, state, env, sum_reward, time_step):
        action = self.action_selection.epsilon_greedy_selection(self.model, state)
        observation, reward, terminated, truncated, _ = self.env.step(action)
        reward = torch.tensor([reward], device=self.device)
        sum_reward += reward
        done = torch.tensor([int(terminated or truncated)], device=self.device)

        if done:
            next_state = None
            return True, sum_reward
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor([[action]], dtype=torch.long, device=self.device)
        self.memory.push(state, action, next_state, reward, done)
        state = next_state
        if time_step % self.config["TAU"] == 0:
            self.fit_model()



        return False, sum_reward


    def update_target(self):
        self.target.load_state_dict(OrderedDict(self.model.state_dict()))

    def fit_model(self):
        if len(self.memory) < self.config["BATCH_SIZE"]:
            return
        sample = self.memory.sample()
        batch = self.Transition(*zip(*sample))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        with torch.no_grad():
            output_next_state_batch = self.target(next_state_batch).detach()
            output_next_state_batch = torch.max(output_next_state_batch, 1)[0].detach()
            output_next_state_batch = torch.reshape(output_next_state_batch,
                                                    (self.config["BATCH_SIZE"], -1)).detach()

        y_batch = reward_batch + self.config['GAMMA'] * output_next_state_batch * (1 - done_batch)
        output = torch.reshape(self.model(state_batch), (self.config["BATCH_SIZE"], -1))
        q_values = torch.gather(output, 1, action_batch)

        loss = self.criterion(q_values, y_batch)
        self.loss = float(loss)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()



