import torch
import torch.nn.functional as F
from pytorch_optimizer import Muon
from torch.distributions import kl_divergence
import gymnasium as gym
from Dark_Matter.utils.networks import LSSM, Buffer_without_lambda

"""
Goals:
1. Smooth trajectory STATUS: PASSED
2. Physics preserved STATUS: PASSED
3. Small drift STATUS: PASSED 
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Device: {device}\n")

"""Liquid State Space Model"""
model = LSSM(action_dim=2,
            embed_dim=4,
            stoch_dim=32,
            discrete_dim=32,
            deter_dim=512
).to(device)

"""Prioritized buffer guided just by mismatch (KL Divergence) from the agent"""
buffer = Buffer_without_lambda(action=1,
                               obs_dim=4,
                               capacity=2000)

paramsM = [p for p in model.parameters() if p.ndim >= 2]

#Experiment testing using AdamW + Muon
A_optimizer = torch.optim.AdamW(list(model.parameters()),
                                lr=2e-4,
                                weight_decay=1e-1)

M_optimizer = Muon([{'params':paramsM, 'use_muon': True}],
                   lr=2e-3,
                   momentum=0.95,
                   Nesterov=True)

env = gym.make('CartPole-v1')
episodes = 200
imagination_step = 15
h_prev = torch.zeros(1, 512).to(device)
z_prev = torch.zeros(1, 32 * 32).to(device)  # stoch_dim*discrete_dim

for i in range(1, episodes+1):
    obs, info = env.reset()

    # 1. Data collection (From the real environment)
    while True:

        # Random action from CartPole-v1
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        transition = (action,
                      reward,
                      terminated,
                      observation)

        # Check if it is over
        if terminated or truncated:
            break

        # Make action from environment OneHot
        action_tensor = F.one_hot(torch.tensor(action), num_classes=2).float().unsqueeze(0).to(device)

        # Make observation from environment a tensor
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

        # Extract memory (h), post (Observation to the real env), and mismatch (The difference between the reality and the belief of the environment from the agent)
        h, z_post, _, _, mismatch, _ = model.observe_step(h_prev,
                                                          z_prev,
                                                          action_tensor,
                                                          observation_tensor,
                                                          t=1)

        # Save transition and mismatch
        buffer.add(transition, mismatch.item())

        #Upload the old h, and z with memory (h), post (Observation to the real env) from the agent
        h_prev, z_prev = h.detach(), z_post.detach()

    # 2.- Imagination (Rollout using brief from the environment)
    if i > 10:
        total_mismatch = 0
        loss = 0
        weights, tree_idx, batch = buffer.sample(batch_size=1)
        h_post, z_post = torch.zeros(1, 512).to(device), torch.zeros(1, 32 * 32).to(device)

        for t in range(imagination_step):
            # Uses the mentioned samples from the buffer to train the agent
            obs, action, reward, continued = batch

            # Preparing data
            act = action[0].item()
            act_tensor = F.one_hot(torch.tensor(act), num_classes=2).float().unsqueeze(0).to(device)
            obs_tensor = (obs.detach().clone()).to(device)

            """MODEL OBS_STEP"""
            _, _, _, post_dist, mismatch, _ = model.observe_step(h_post,
                                                                 z_post,
                                                                 act_tensor,
                                                                 obs_tensor,
                                                                 t=1)
            """MODEL IMG_STEP"""
            z_prior, h_imagination, prior_dist_ = model.imagination_step(h_post,
                                                                         z_post,
                                                                         act_tensor)
            # This time mismatch is between post_dist (which represent the reality), and prior_dist_ (which represent the brief that the agent have to the env)
            mismatch_ = kl_divergence(post_dist, prior_dist_).mean()
            loss += mismatch_
            h_post, z_post = h_imagination, z_prior

            # Parameter used to calculate the real mismatch of this whole loop
            total_mismatch += mismatch.item()

            # Buffer update the priorities based on the mismatch, then it search for the ubication of the "memory"
            buffer.update(tree_idx[0], mismatch.item())

        avg_mismatch = total_mismatch / imagination_step

        # Finally the loss and optimizer based on data improve the agent.
        M_optimizer.zero_grad()
        A_optimizer.zero_grad()
        loss.backward()
        M_optimizer.step()
        A_optimizer.step()

        #Advice: print issue fixed!!
        print(f"Episode: {i} // Mismatch: {avg_mismatch:.5f} // Loss: {loss:.6f}")


env.close()
