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

 
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())
        
    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
 
    def store(self, *args):
        self.memory.push(*args)

 
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return None
 
        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))
 
        states = torch.as_tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
 
        # Q(s, a) for the actions actually taken
        q_values = self.policy_net(states).gather(1, actions)
 
        # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)
 
        loss = self.loss_fn(q_values, targets)
 
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
 
        return loss.item()
 
    def maybe_update_target(self, episode: int):
        if episode % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

 
def train(
    n_episodes: int = 400,
    max_steps: int = 500,
    solved_reward: float = 475.0,
    solved_window: int = 20,
    seed: int = 0,
):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, n_actions, device)
 
    episode_rewards = []
 
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
 
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
 
            agent.store(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
 
            agent.optimize()
 
            if done:
                break
 
        agent.decay_epsilon()
        agent.maybe_update_target(episode)
        episode_rewards.append(episode_reward)
 
        avg_recent = np.mean(episode_rewards[-solved_window:])
        if episode % 10 == 0 or episode == 1:
            print(
                f"Episode {episode:4d} | Reward: {episode_reward:6.1f} | "
                f"Avg({solved_window}): {avg_recent:6.1f} | Epsilon: {agent.eps:.3f}"
            )
 
        if len(episode_rewards) >= solved_window and avg_recent >= solved_reward:
            print(f"\nSolved at episode {episode}! Average reward: {avg_recent:.1f}")
            break
 
    env.close()
 
    torch.save(agent.policy_net.state_dict(), "/mnt/user-data/outputs/dqn_cartpole_weights.pt")
 
    plt.figure(figsize=(9, 5))
    plt.plot(episode_rewards, alpha=0.4, label="Episode reward")
    if len(episode_rewards) >= solved_window:
        moving_avg = np.convolve(
            episode_rewards, np.ones(solved_window) / solved_window, mode="valid"
        )
        plt.plot(
            range(solved_window - 1, len(episode_rewards)),
            moving_avg,
            label=f"{solved_window}-episode moving average",
            linewidth=2,
        )
    plt.axhline(solved_reward, color="red", linestyle="--", label="Solved threshold")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("DQN on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/training_rewards.png", dpi=120)
 
    return agent, episode_rewards
 
 
def watch(agent: DQNAgent, n_episodes: int = 3):
    env = gym.make("CartPole-v1", render_mode="human")
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"[Render] Episode {ep + 1}: reward = {total_reward}")
    env.close()
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent on CartPole-v1")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--render", action="store_true", help="Render a few episodes after training")
    args = parser.parse_args()
 
    trained_agent, rewards = train(n_episodes=args.episodes)
 
    if args.render:
        watch(trained_agent)