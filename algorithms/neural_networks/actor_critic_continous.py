import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

class ActorNetworkContinous(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(ActorNetworkContinous, self).__init__()
        hidden_size = 256
        self.actor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, output_size*2))  # 2*output_size for mean and std_dev
        self.optimizer = optim.Adam(self.parameters(), lr = config["ALPHA"])

    def forward(self, x):
        mean, log_std = torch.chunk(self.actor(x), 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)  # to prevent std from going to zero or very high
        std = torch.exp(log_std)
        distribution = Normal(mean, std)
        return distribution