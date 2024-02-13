import gym
import yaml
from run.train import train
from run.test import test


def parameters() -> dict:

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    config = parameters()
    env = gym.make(config["environment"])
    if config["mode"] == "train":
        train(env, config)
    else:
        test(env, config)

    print("Done.........")





