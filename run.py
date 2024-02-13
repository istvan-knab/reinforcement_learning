import yaml

from algorithms.DQN.dqn import DQNAgent

def determine_agent(config) -> object:
    if config['algorithm'] == 'DQN':
        return DQNAgent(config)
    elif config['algorithm'] == 'DDPG':
        pass
    else:
        raise NotImplementedError

def parameters() -> dict:

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def test():
    pass

if __name__ == '__main__':
    config = parameters()
    determine_agent(config)

    if config["mode"] == "train":
        agent.train(env, config)
    else:
        test(env, config)

    print("Done.........")





