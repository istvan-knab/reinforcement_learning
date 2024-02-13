import gymnasium as gym
import yaml

def determine_agent(config) -> object:
    if config['algorithm'] == 'DQN':
        pass
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
    env = gym.make(config["environment"])
    determine_agent()

    if config["mode"] == "train":
        agent.train(env, config)
    else:
        test(env, config)

    print("Done.........")





