import gym


def main() -> dict:
    pass

if __name__ == '__main__':
    config = main()
    env = gym.make(config["environment"])
    for episode in range(config['EPISODES']):
        ...

