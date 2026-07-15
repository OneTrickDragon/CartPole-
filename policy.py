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
    
