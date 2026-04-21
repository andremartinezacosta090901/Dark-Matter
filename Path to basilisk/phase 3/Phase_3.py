import gymnasium as gym
import time
import torch
import torch.nn.functional as F
from COSMOS.COSMOS_optim import COSMOS
from torch.distributions import kl_divergence
from Dark_Matter.utils.networks import LSSM, Buffer, PolicyHead, ValueHead, RewardHead
from Dark_Matter.utils.utils import ComputeLambdaValues
"""
Goals:
1. Train actor-critic just using imagination 
2. Fix reset bug from GLU
3. Fix math error bug 
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Device: {device}\n")

"""Liquid State Space Model"""
model = LSSM(action_dim=2,
            embedding_dim=4,
            stoch_dim=32,
            discrete_dim=32,
            deter_dim=1024).to(device)

"""Policy head"""
actor = PolicyHead(
                feature_dim=512,
                action_dim=2,
                hidden_dim=[256],
                unimix=0.01).to(device)

"""Value head"""
critic = ValueHead(
                deter_dim=512,
                stoch_dim=32,
                classes=32,
                embedding_dim=4).to(device)

"""Reward head"""
reward_head = RewardHead(
                deter_dim=512,
                stoch_dim=32,
                classes=32,
                embedding_dim=4,
                bins=255).to(device)

"""Prioritized buffer guided just by mismatch (KL Divergence) and lambda return from the agent"""
buffer = Buffer(action=1,
                obs_dim=4, #Cartpole =4
                capacity=2000)

all_params = (list(model.parameters()) +
              list(actor.parameters()) +
              list(critic.parameters()) +
              list(reward_head.parameters()))

#Experiment testing using LaProp + Muon
optimizer = COSMOS([
    {'params': model.parameters(), 'lr': 2e-3},
    {'params': actor.parameters(), 'lr': 1e-4},
    {'params': critic.parameters(), 'lr': 1e-4},
    {'params': reward_head.parameters(), 'lr': 1e-4}],
    betas=(0.95, 0.95),
    rank=32,
    nestrov=True,
    weight_decay=0.1)

env = gym.make('CartPole-v1')
episodes = 50
imagination_step = 100
h_prev = torch.zeros(1, 512).to(device)
z_prev = torch.zeros(1, 32 * 32).to(device)  # stoch_dim*discrete_dim
prev_time = time.time()

"""COFIGURATION:"""
Beta = 0.1
Entropy = 1e-4
reset = 0

for i in range(1, episodes+1):
    obs, info = env.reset()

    # 1. Data collection (From the real environment)
    while True:
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        # Random action from CartPole-v1
        action, dist = actor(h_prev)
        action_ = action.argmax(dim=1).item()
        observation, reward, terminated, truncated, _ = env.step(action_)
        transition = (action_,
                      reward,
                      terminated,
                      observation)

        # Check if it is over
        if terminated or truncated:
            reset = 1
            break

        # Make observation from environment a tensor
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

        # Extract memory (h), post (Observation to the real env), and mismatch (The difference between the reality and the belief of the environment from the agent)
        h, z_post, _, _, mismatch, _ = model.observe_step(reset,
                                                        h_prev,
                                                        z_prev,
                                                        action,
                                                        observation_tensor,
                                                        t=dt)

        # Save transition and mismatch
        buffer.add(transition, mismatch.item(), 0)

        #Upload the old h, and z with memory (h), post (Observation to the real env) from the agent
        h_prev, z_prev = h.detach(), z_post.detach()

    # 2.- Imagination (Rollout using brief from the environment)
    if i > 10:
        total_mismatch = 0
        loss = 0
        weights, tree_idx, batch = buffer.sample(batch_size=1)

        # Uses the mentioned samples from the buffer to train the agent
        obs, action_sample, reward_sample, continued_sample = [b.to(device) for b in batch]
        h_post, z_post = torch.zeros(1, 512).to(device), torch.zeros(1, 32 * 32).to(device)

        for t in range(imagination_step):

            # Preparing data
            obs_tensor = (obs.detach().clone()).to(device)

            """MODEL OBS_STEP"""
            _, _, _, post_dist, mismatch, _ = model.observe_step(reset,
                                                                h_post,
                                                                z_post,
                                                                action,
                                                                obs_tensor,
                                                                t=dt)



            """MODEL IMG_STEP"""
            z_prior, h_imagination, prior_dist_ = model.imagination_step(h_post,
                                                                         z_post,
                                                                         action,
                                                                         t=dt)

            print(f"DEBUG SHAPES: z_post={z_post.shape}, h_imag={h_imagination.shape}")
            value, _ = critic(z_post.detach(), h_imagination.detach(), t=dt)
            logits, _ = reward_head(z_post, h_imagination)
            lambda_return = ComputeLambdaValues(reward_sample, value, continued_sample)


            # This time mismatch is between post_dist (which represent the reality), and prior_dist_ (which represent the brief that the agent have to the env)
            mismatch_ = kl_divergence(post_dist, prior_dist_).mean()

            reward_target = reward_head.TwoHotDistribution(reward_sample)
            reward_loss = -torch.sum(reward_target * F.log_softmax(logits, dim=-1), dim=-1).mean()

            wm_loss = (Beta * mismatch_) + reward_loss

            critic_loss = F.mse_loss(value, lambda_return.detach())
            action, dist = actor(h_imagination.detach())
            entropy = dist.entropy().mean()
            actor_loss = -lambda_return.mean() - (Entropy * entropy)
            loss += (wm_loss + critic_loss + actor_loss)

            h_post, z_post = h_imagination, z_prior

            # Parameter used to calculate the real mismatch of this whole loop
            total_mismatch += mismatch.item()

            # Buffer update the priorities based on the mismatch, then it search for the ubication of the "memory"
            buffer.update(tree_idx[0], mismatch.item(), lambda_return)

        avg_mismatch = total_mismatch / imagination_step

        # Finally the loss and optimizer based on data improve the agent.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Advice: print issue fixed!!
        print(f"Episode: {i} // Reward: {reward:.4f} // Mismatch: {avg_mismatch:.4f} // Loss: {loss:.3f}")

env.close()