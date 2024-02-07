import yaml

from algorithms.dqn import DQN
from algorithms.ddpg import DDPG
from algorithms.td3 import TD3
from run.log import Logger


# Read Parameters
def read_parameters() -> dict:

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        print(type(config))
    return config

def train_model(config: dict) -> None:
    # determine algorithm
    agent = determine_algorithm(config)
    for i in range(config["EPISODES"]):
        done = False
        while not done:
            agent.training_step()

def test_model(config: dict) -> None:
    # code for training the model
    for i in range(config["EPISODES"]):
        done = False
        while not done:
            pass

def determine_algorithm(config: dict) -> object:
    if config["ALGORITHM"] == "DDPG": agent = DDPG(config)
    elif config["ALGORITHM"] == "DQN": agent = DQN(config)
    elif config["ALGORITHM"] == "TD3": agent = TD3(config)

    return agent