import torch
import gymnasium as gym
import yaml
import numpy as np

from algorithms.DQN.epsilon_greedy import EpsilonGreedy
from algorithms.neural_networks.actor_critic import ActorNetwork, CriticNetwork
from algorithms.PPO.ppo_memory import Memory
from algorithms.logger import Logger
from algorithms.io import IO

class PPOAgent(object):
    def __init__(self, config):
        with open('algorithms/PPO/ppo_config.yaml', 'r') as file:
            self.ppo_config = yaml.safe_load(file)
        self.env = gym.make(config["environment"], render_mode=config["RENDER_MODE"])
        self.action_selection = EpsilonGreedy(config, self.env)
        self.config = config
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.actor = ActorNetwork(state_size, action_size, config)
        self.critic = CriticNetwork(state_size, config)
        self.memory = Memory(self.ppo_config["BATCH_SIZE"])


    def train(self, config):
        score_history = []
        avg_score = []
        number_of_steps = 0
        for episode in range(config["EPISODES"]):
            state, _  = self.env.reset()
            done = False
            score = 0
            while not done:
                number_of_steps += 1
                action, probability, value = self.choose_action(state)
                next_state, reward, done, terminated, truncated = self.env.step(action)
                # TODO
                self.memory.push(state, action, probability, value, reward, done)
                if number_of_steps % self.ppo_config["TAU"]:
                    self.fit()
                state = next_state
                if terminated or truncated:
                    done = True
                    score_history.append(score)
                    avg_score.append(np.mean(score_history[100:]))



    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float32, device=self.config["DEVICE"])
        distribution = self.actor(state)
        value = self.critic(state)
        action = distribution.sample()

        probabilities = torch.squeeze(distribution.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probabilities, value

    def fit(self):
        pass
    def run_policy(self):
        pass

    def compute_advantage_estimates(self):
        pass

    def optimize_policy(self):
        pass
