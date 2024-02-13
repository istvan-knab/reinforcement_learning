import yaml
import torch
import gymnasium as gym

from algorithms.dqn import DQN
from algorithms.ddpg import DDPG
from algorithms.td3 import TD3
from algorithms.tabular_q_learning import TabularQLearning
from run.log import Logger


# Read Parameters
def read_parameters() -> dict:

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(config: dict) -> None:
    # determine environment
    env = gym.make(config['ENVIRONMENT'], render_mode=config['RENDER_MODE'])

    # determine algorithm
    agent = determine_algorithm(config, env)

    for episode in range(config["EPISODES"]):
        done = False
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=config["DEVICE"]).unsqueeze(0)
        agent.action_selection.epsilon_update()
        sum_reward = 0
        time_step = 0
        while not done:
            done, sum_reward = agent.training_step(state, env, sum_reward, time_step)
            if done:
                print(sum_reward, agent.action_selection.config["EPSILON"])

            time_step += 1


        agent.update_target()




def test_model(config: dict) -> None:
    # code for training the model
    for i in range(config["EPISODES"]):
        done = False
        while not done:
            pass

def determine_algorithm(config: dict, env) -> object:
    if config["ALGORITHM"] == "DDPG": agent = DDPG(config, device=config["DEVICE"], env= env)
    elif config["ALGORITHM"] == "DQN": agent = DQN(config, device=config["DEVICE"], env= env)
    elif config["ALGORITHM"] == "TD3": agent = TD3(config, device=config["DEVICE"], env= env)
    elif config["ALGORITHM"] == "Tabular_Q_Learning":agent = TabularQLearning(config, device=config["DEVICE"],
                                                                              env= env)

    return agent