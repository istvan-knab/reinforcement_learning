import random
import torch
class EpsilonGreedy:
    def __init__(self, config):
        self.config = config

    def epsilon_greedy_selection(self, model: object, state):

        if random.random() < self.config['EPSILON']:
            "Random exploratory step"
            action = random.choice(self.config['ACTIONS'])

            return action
        else:
            "Greedy exploitation step"
            with torch.no_grad():
                model.eval()
                q_calc = model(state)
                model.train()
                action = int(torch.argmax(q_calc))

                return action

    def epsilon_update(self):
        self.config["EPSILON"] *= self.config["EPSILON_DECAY"]
