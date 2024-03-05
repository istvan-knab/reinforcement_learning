import torch
import gymnasium as gym

from algorithms.DQN.epsilon_greedy import EpsilonGreedy
from algorithms.logger import Logger
from algorithms.io import IO

class PPOAgent(object):
    def __init__(self, config):
        self.env = gym.make(config["environment"], render_mode=config["RENDER_MODE"])
        self.action_selection = EpsilonGreedy(config, self.env)

    def train(self, config):
        for episode in range(config['episodes']):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=config["DEVICE"]).unsqueeze(0)
            done = False
            self.action_selection.epsilon_update()
            episode_reward = 0
            episode_loss = 0
            while not done:
                pass
