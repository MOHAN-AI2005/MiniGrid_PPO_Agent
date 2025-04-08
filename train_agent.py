import gym
import torch
import numpy as np
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

env = ImgObsWrapper(RGBImgPartialObsWrapper(gym.make("MiniGrid-Empty-5x5-v0")))
obs_shape = env.observation_space.shape
n_actions = env.action_space.n

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape), 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
writer = SummaryWriter()

for episode in range(1000):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = policy(obs_tensor)
        probs = Categorical(logits=logits)
        action = probs.sample().item()
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
    writer.add_scalar("Reward", total_reward, episode)
    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")
writer.close()
