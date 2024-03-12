import torch
import gymnasium as gym
import yaml
import numpy as np
from collections import namedtuple

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
        self.logger = Logger(config)
        #TODO:transition also in memory
        #self.transition = namedtuple('Transition', ('state', 'action', ''))


    def train(self, config):
        score_history = []
        avg_score = []
        number_of_steps = 0
        last = 0
        for episode in range(config["EPISODES"]):
            state, _  = self.env.reset()
            done = False
            score = 0
            while not done:
                number_of_steps += 1
                action, probability, value = self.choose_action(state)
                next_state, reward, done, terminated, truncated = self.env.step(action)
                self.memory.push(state, action, probability, value, reward, done)
                score += reward
                if number_of_steps % self.ppo_config["TAU"]:
                    # TODO
                    losses = self.fit()
                state = next_state
                if terminated or truncated:
                    done = True
                    score_history.append(score)
                    last = avg_score.append(np.mean(score_history[100:]))
            self.logger.step(episode)



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
        losses = 0
        for run_policy_iteration in range(self.ppo_config["POLICY_ITERATIONS"]):
            states, actions, probabilities, values,\
            rewards, dones, batches = self.memory.generate_batch()
            advantages = np.zeros(len(rewards), dtype=np.float32)
            for t in range(len(rewards)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards)-1):
                    a_t += discount * (rewards[k] + self.config["GAMMA"] * values[k + 1] * (1 - int(dones[k])) - values[k])
                    discount = self.config["GAMMA"] * self.ppo_config["LAMBDA"]
                advantages[t] = a_t
            advantages = torch.tensor(advantages, dtype=torch.float32, device=self.config["DEVICE"])
            values = torch.tensor(values, dtype=torch.float32, device=self.config["DEVICE"])
            for batch in batches:
                states = torch.tensor(states[batch], dtype=torch.float32, device=self.config["DEVICE"])
                old_probabilities = torch.tensor(probabilities[batch], dtype=torch.float32, device=self.config["DEVICE"])
                actions = torch.tensor(actions[batch], dtype=torch.int32, device=self.config["DEVICE"])

                distribution = self.actor(states)
                critic_value = self.critic(states)
                new_probabilities = distribution.log_prob(actions)
                probability_ratio = new_probabilities.exp() / old_probabilities.exp()
                weighted_probabilities = advantages[batch] * probability_ratio
                weighted_clipped_probabilities = torch.clamp(probability_ratio, 1 - self.ppo_config["CLIP_PROBABILITY"],
                                                             1 + self.ppo_config["CLIP_PROBABILITY"])  * advantages[batch]
                actor_loss = - torch.min(weighted_probabilities, weighted_clipped_probabilities).mean()

                returns = advantages[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss

                self. actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                losses += total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear()
        return losses




