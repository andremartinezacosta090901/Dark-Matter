import torch
import torch.nn.functional as F
import gymnasium as gym
from Dark_Matter.utils.networks import LSSM

"""
Goals:
1. Smooth trajectory
2. Physics preserved
3. Small drift
"""
model = LSSM(
    action_dim=2,
    embed_dim=4,
    stoch_dim=32,
    discrete_dim=32,
    deter_dim=200
)

env = gym.make('CartPole-v1', render_mode=None)
obs, info = env.reset()
episodes = 5
for _ in range(1, episodes+1):
    obs, info = env.reset()

    batch_size = 1
    z_size = 32*32 #stoch_dim*discrete_dim
    h_prev = torch.zeros(batch_size, 200)
    z_prev = torch.zeros(batch_size, z_size)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        action_tensor = F.one_hot(torch.tensor(action), num_classes=2).float().unsqueeze(0)
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        h, z_post, prior_dist, post_dist, mismatch, avg_conf = model.observe_step(
            prev_h=h_prev,
            prev_z=z_prev,
            action=action_tensor,
            observation_embed=obs_tensor,
            t=1
        )
        prev_h = h.detach()
        prev_z = z_post.detach()
        for t in range(15):
            z_prior, h, prior_dist = model.imagination_step(
                prev_h=prev_h,
                prev_z=prev_z,
                action=action_tensor
            )
            if t ==14:
                prev_h = h.detach()
                prev_z = z_prior.detach()
                print(f"prev_h: {prev_h}")
                print(f"prev_z: {prev_z}")
                break
env.close()
