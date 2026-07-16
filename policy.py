import argparse
import random
from collections import deque, namedtuple
 
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 120):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    

class ReplayBuffer:
    """Fixed-size cyclic buffer storing past transitions for experience replay."""
 
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
 
    def push(self, *args):
        self.buffer.append(Transition(*args))
 
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
 
    def __len__(self):
        return len(self.buffer)
    


class DQNAgent:
    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 device: torch.device,
                 lr: float=1e-3,
                 gamma: float = 0.99,
                 buffer_capacity: int = 50_000,
                 batch_size: int = 64,
                 eps_start: float = 1.0,
                 eps_end: float = 0.02,
                 eps_decay: float = 0.995,
                 target_update_freq: int = 10,
    ):
        
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
 
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
 
        self.target_update_freq = target_update_freq
        self.update_count = 0
 
        self.policy_net = QNetwork(state_dim, n_actions).to(device)
        self.target_net = QNetwork(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
 
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
 
        self.memory = ReplayBuffer(buffer_capacity)
