import torch

class DQN(object):
    def __init__(self, config, env, device):
        self.config = config
        self.env = env
        self.device = device

    def training_step(self):

        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def fit_model(self):
        pass