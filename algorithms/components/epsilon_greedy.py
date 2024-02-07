import random
class EpsilonGreedy:
    def __init__(self, config):
        self.config = config

    def epsilon_greedy_selection(self):

        if random.random() < self.config['EPSILON']:
            "Random exploratory step"
            return random.choice(self.config['ACTIONS'])
        else:
            "Greedy exploitation step"
            return self.config['ACTIONS'][self.config['ACTIONS'].index(max(self.config['ACTIONS']))]

    def epsilon_update(self):
        self.config["EPSILON"] *= self.config["EPSILON_DECAY"]