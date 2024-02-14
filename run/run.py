import yaml

from algorithms.DQN.dqn import DQNAgent
from algorithms.DDPG.ddpg import DDPGAgent

def determine_agent(config) -> object:
    if config['algorithm'] == 'DQN':
        return DQNAgent(config)
    elif config['algorithm'] == 'DDPG':
        pass
    else:
        raise NotImplementedError

def parameters() -> dict:

    with open('run/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def test():
    pass

if __name__ == '__main__':
    config = parameters()
    agent = determine_agent(config)

    if config["mode"] == "train":
        agent.train(config)
    else:
        test(config)

    print("Done.........")





