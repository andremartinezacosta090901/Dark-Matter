import torch 
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import kl_divergence
from torch.distributions import OneHotCategorical
import lpips
import unittest
from unittest import TestCase
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Dark_Matter.utils.utils import GLU, model_builder_layer, StochasticNetwork, CFC, symexp, symlog, ExponentialMovingAverage, Sum_Tree, device

class Encoder(nn.Module): #WORKS perfectly
    def __init__(self, 
                input_channels, 
                hidden_layers, 
                use_norm=True, 
                activation='ReLU'):
        super().__init__()
        layers = []
        current_channels = input_channels
        for hidden_layer in hidden_layers:
            conv_layer = nn.Conv2d(current_channels,
                                    hidden_layer,
                                    kernel_size=5, #Change
                                    stride=2,
                                    padding= 2
                                    )
            nn.init.kaiming_normal_(conv_layer.weight, nonlinearity=activation.lower())
            if conv_layer.bias is not None:
                nn.init.constant_(conv_layer.bias, 0.0)
            
            layers.append(conv_layer)
            if use_norm:
                layers.append(nn.BatchNorm2d(hidden_layer))
            
            layers.append(getattr(nn, activation)())
            
            current_channels = hidden_layer
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        features = self.layers(x)
        return features
    
class Decoder(nn.Module): #WORKS perfectly
    def __init__(self, 
                input_channels, 
                hidden_layers, 
                use_norm=True, 
                activation='ReLU'):
        super().__init__()
        layers = []
        current_channels = input_channels
        for i, hidden_layer in enumerate(hidden_layers):
            convT_layer = nn.ConvTranspose2d(current_channels,
                                            hidden_layer,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1)
            nn.init.kaiming_normal_(convT_layer.weight, nonlinearity=activation.lower())
            if convT_layer.bias is not None:
                nn.init.constant_(convT_layer.bias, 0.0)
                
            layers.append(convT_layer)
            verify_last_layer = i == len(hidden_layers) - 1
            
            if not verify_last_layer:
                if use_norm:
                    layers.append(nn.BatchNorm2d(hidden_layer))
                layers.append(getattr(nn, activation)())
                
            else:
                layers.append(nn.Sigmoid()) #Change
            current_channels = hidden_layer
            
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Vector_Quantizer(nn.Module): #WORKS perfectly
    def __init__(self, 
                num_embeddings,
                embedding_dim, 
                useEMA=True, 
                
                decay=0.8, #Change 
                
                epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.useEMA = useEMA
        self.decay = decay
        self.epsilon = epsilon
        limit = 3**0.5
        self.register_buffer('embedding', torch.zeros(num_embeddings, embedding_dim))
        self.embedding.data.uniform_(-limit, limit)
        self.times_used = ExponentialMovingAverage(decay, (num_embeddings,))
        self.average = ExponentialMovingAverage(decay, (num_embeddings, embedding_dim))
    
    def forward(self, x):        
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = ((flat_x**2).sum(1, keepdim=True) -2 * flat_x @ self.embedding.t() + 
                    (self.embedding.t()**2).sum(0, keepdim=True)
                    )
        encoding_indices = distances.argmin(1)
        quantized = F.embedding(encoding_indices, self.embedding)
        quantized = quantized.view(x.shape[0], x.shape[2], x.shape[3], x.shape[1]).permute(0, 3, 1, 2)
        
        if self.useEMA and self.training:
            encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
            encodings_sum = encodings_one_hot.sum(0)
            self.times_used.calculation(encodings_sum)
            vector_sum = encodings_one_hot.t() @ flat_x
            self.average.calculation(vector_sum)
            
            n_i_stable = self.times_used.avg.sum()
            cluster_size = ((self.times_used.avg + self.epsilon)
            / (n_i_stable + self.num_embeddings * self.epsilon)
            * n_i_stable
            )
            self.embedding.data.copy_(self.average.avg / cluster_size.unsqueeze(1))
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        
        loss = 0.25 * e_latent_loss #Change 0.25 -> 1
        
        quantized = x + (quantized - x).detach()
        
        return loss, quantized, encoding_indices
    
    @torch.no_grad()
    def restart_dead_codes(self, encoding_indices, z_e):
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        used = encodings.sum(0)
        dead_indices = (used == 0).nonzero(as_tuple=True)[0]
        dead_codes = dead_indices.size(0)
        
        if dead_codes > 0:
            z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
            
            random_index = torch.randint(0, z_e_flat.size(0), (dead_codes,)).to(z_e.device)
            replacement = z_e_flat[random_index]
            self.embedding.data[dead_indices] = replacement
            
            return dead_codes
        
        return 0


class FSQ_(nn.Module): #Made by me
    def __init__(self, channels, levels):
        super().__init__()
        self.levels = torch.tensor(levels)
        self._dim = len(levels)
        self.bottleneck = nn.Linear(channels, self._dim)
        self.expansion = nn.Linear(self._dim, channels)
        
        basis = [1]
        for i in levels[:-1]:
            basis.append(basis[-1] * i)
        self.register_buffer('basis', torch.tensor(basis))
        self.register_buffer('levels_buffer', self.levels)
        
    def forward(self, z_e):
        z = z_e.permute(0, 2, 3, 1)
        z = self.bottleneck(z)
        z_bound = torch.tanh(z)
        s = (self.levels_buffer -1) /2
        z_scaled = z_bound * s
        z_scaled = torch.round(z_scaled)
        z_quantized = z_scaled / s
        z_ste = z_bound + (z_quantized - z_bound).detach()
        indices = torch.sum((z_scaled+s)* self.basis, dim=-1).long()
        z_out = self.expansion(z_ste)
        z_out = z_out.permute(0, 3, 1, 2)
        return z_out, indices

class FSQ(nn.Module): #Version upgraded highly inspired in https://github.com/Nikolai10/FSQ
    def __init__(self, levels: list[int], eps: float = 1e-3):
        super().__init__()
        self.levels = levels
        self.eps = eps
        
        levels_ = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer('levels_buffer', levels_)
        
        basis_ = torch.tensor(
            np.concatenate(([1], np.cumprod(levels[:-1]))).astype(np.int32), 
            dtype=torch.long)
        self.register_buffer('basis', basis_)
        
        off_sets = torch.where(levels_ % 2 == 1, 0.0, 0.5)
        self.register_buffer('offsets', off_sets)
        
    def bound(self, z):
        levels = self.levels_buffer.view(1, -1, 1, 1)
        offsets = self.offsets.view(1, -1, 1, 1)
        half_level = (levels - 1) * (1 - self.eps) / 2
        shift = torch.atanh(offsets / (half_level + self.eps))
        return torch.tanh(z + shift) * half_level - offsets
    
    def quantize(self, z):
        z_bound = self.bound(z)
        z_round = torch.round(z_bound)
        z_ste = z_bound + (z_round - z_bound).detach()
        half_width = (self.levels_buffer.view(1, -1, 1, 1) // 2)
        return z_ste / half_width
    
    def codes_to_indices(self, codes):
        """Use Nograd when indices are used for AI
            example:
            with torch.no_grad():
                z = model.encoder(current_frame)
                zhat = model.quantizer(z)
                indices = model.quantizer.codes_to_indices(zhat)"""
                
        half_width = self.levels_buffer // 2
        zhat = (codes * half_width) + half_width
        zhat = torch.round(zhat).long()
        return (zhat* self.basis).sum(-1)
    
    def forward(self, z_e):
        return self.quantize(z_e)

class VQ_VAE(nn.Module):  #WORKS perfectly
    def __init__(self, 
                input_channels, 
                hidden_layers, 
                embedding_dim, 
                num_embeddings):
        super().__init__()
        assert hidden_layers[-1] == embedding_dim, \
            f"Error!!!: Encoder output ({hidden_layers[-1]}) it does not match Embedding Dim ({embedding_dim})"
        self.encoder = Encoder(
            input_channels=input_channels,
            hidden_layers=hidden_layers
        )
        self.quantizer = Vector_Quantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )
        
        layer_inverted = hidden_layers[::-1]
        decoder_layers = layer_inverted[1:] + [input_channels]
        
        self.decoder = Decoder(
            input_channels = embedding_dim,
            hidden_layers = decoder_layers
        )
    
    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, encoding_indices = self.quantizer(z)
        recon = self.decoder(quantized)
        
        return {'quantized': quantized,
                'recon': recon,
                'loss': loss,
                'encoding_indices': encoding_indices
                }

class LSSM(nn.Module): #Liquid State Space Model and WORKS PERFECTLY
    def __init__(self, 
                action_dim,
                embed_dim,
                stoch_dim=32,
                discrete_dim=32,
                deter_dim=200,
                hidden_dim=200,
                activation='ReLU'):
        
        super().__init__()
        z_size = stoch_dim * discrete_dim
        self.deter_dim = deter_dim
        
        self.in_layer = model_builder_layer(
            in_channel=action_dim + z_size,
            output_channel=hidden_dim,
            hidden_lay=[hidden_dim],
            norm=True,
            zeroLastLayer=False,
            activation=activation
        )
        
        self.glu = GLU(
            input_size=hidden_dim,
            size=deter_dim,
            norm=True,
            activation=activation,
            zeroLastLayer=True
        )
        
        self.prior = StochasticNetwork(
            in_channel=deter_dim,
            stoch_dim=stoch_dim,
            discrete_dim=discrete_dim,
            hidden_layers=hidden_dim
        )
        
        self.post = StochasticNetwork(
            in_channel=deter_dim + embed_dim,
            stoch_dim=stoch_dim,
            discrete_dim=discrete_dim,
            hidden_layers=hidden_dim
        )
        
        
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('count', torch.tensor(1e-6))
    
    def update_stats(self, x):
        val = x.detach()
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        return self.mean
    
    def imagination_step(self, prev_h, prev_z, action, t=1):
        fused_input = torch.cat([action, prev_z], dim=-1)
        x = self.in_layer(fused_input)
        x_output,h = self.glu(x, prev_h, t=t)
        z_prior, prior_dist = self.prior(x_output)
        return z_prior, h, prior_dist
    
    def observe_step(self, prev_h, prev_z, action, observation_embed, t=1):
        fused_input = torch.cat([action, prev_z], dim=-1)
        x = self.in_layer(fused_input)
        x_output,h = self.glu(x, prev_h, t=t)
        _, prior_dist = self.prior(x_output)
        post_input = torch.cat([h, observation_embed], dim=-1)
        z_post, post_dist = self.post(post_input)
        mismatch = kl_divergence(post_dist, prior_dist).mean()
        #mismatch = torch.maximum(mismatch, torch.tensor(1.0))
        avg_confussion = self.update_stats(mismatch)
        return h, z_post, prior_dist, post_dist, mismatch, avg_confussion

class RewardHead(nn.Module): #Works perfectly
    def __init__(self, 
                deter_dim=4096, 
                stoch_dim=32, 
                classes=32, 
                embedding_dim=256, 
                bins=255):
        super().__init__()
        x_dim = deter_dim + (stoch_dim*classes)
        self.reward_head = model_builder_layer(in_channel=x_dim, hidden_lay=[embedding_dim], output_channel=bins, norm=True, zeroLastLayer=True, activation='ReLU')
        self.register_buffer('buckets', torch.linspace(-20, 20, bins))
        
    def get_probs(self, x):
        probs = F.softmax(x, dim=-1)
        v_sym = (probs * self.buckets).sum(dim=-1, keepdim=True) 
        return symexp(v_sym)
    
    def forward(self, z , h):
        z_flat = z.view(z.shape[0], -1)
        features = torch.cat([h, z_flat], dim=1)
        logits = self.reward_head(features)
        reward = self.get_probs(logits)
        return logits, reward
    
    def TwoHotDistribution(self, real_rewards):
        target = symlog(real_rewards) 
        indices = torch.searchsorted(self.buckets, target) - 1
        indices = torch.clamp(indices, 0, len(self.buckets) - 2)
        
        bin_low = self.buckets[indices]
        bin_high = self.buckets[indices + 1]
        weight_high = (target - bin_low) / (bin_high - bin_low)
        weight_high = torch.clamp(weight_high, 0.0, 1.0)
        target_dist = torch.zeros(target.shape[0], len(self.buckets), device=target.device)
        target_dist.scatter_(1, indices.unsqueeze(1), 1.0 - weight_high.unsqueeze(1))
        target_dist.scatter_(1, (indices + 1).unsqueeze(1), weight_high.unsqueeze(1))
        
        return target_dist

class ValueHead(nn.Module): #WORKS Perfectly
    def __init__(self, 
                deter_dim=4096, 
                stoch_dim=32, 
                classes=32, 
                embedding_dim=256):
        super().__init__()
        x_dim = deter_dim + (stoch_dim * classes)
        self.value_head = CFC(
            input_size=x_dim,
            size=embedding_dim,
            norm=True,
            activation='ReLU',
            zeroLastLayer=True
        )
        self.to_value = nn.Linear(embedding_dim, 1)

    def forward(self, x, h, t=1):
        h_next = self.value_head(x, h, t=t)
        value = self.to_value(h_next)
        print("x:", x.shape)
        print("h:", h.shape)
        return value, h_next

class PolicyHead(nn.Module): #WORKS perfectly
    def __init__(self,
                feature_dim, 
                action_dim, 
                hidden_dim=[256],
                unimix=0.01):
        super().__init__()
        self.action_dim = action_dim
        self.unimix = unimix
        self.net = model_builder_layer(in_channel=feature_dim,
                                        hidden_lay=hidden_dim, 
                                        output_channel=action_dim, 
                                        norm=True, 
                                        zeroLastLayer=True, 
                                        activation='ReLU')

    def forward(self, features):
        logits = self.net(features)
        if self.unimix > 0:
            probs = torch.softmax(logits, dim=-1)
            probs = (1.0 - self.unimix) * probs + (self.unimix / self.action_dim)
            logits = torch.log(probs) 
            
        dist = OneHotCategorical(logits=logits)
        sample = dist.sample()
        
        action = sample + (dist.probs - dist.probs.detach())
        return action, dist

class Buffer: #WORK in progress
    def __init__(self, action, capacity, epsilon=1e-2, alpha=0.1, beta=0.1):
        self.sumtree = Sum_Tree(capacity)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        
        # transition: state, action, reward, next_state, done
        self.action = torch.empty(capacity, action, dtype=torch.long)
        self.reward = torch.empty(capacity, dtype=torch.float)
        self.continued = torch.empty(capacity, dtype=torch.bool)
        self.obs = torch.empty(capacity, dtype=torch.float)
        
        self.count = 0
        self.real_size = 0
        self.size = capacity
        
    def get_priority(self, lambda_, mismatch_score):
        priority = abs(
            (self.alpha * mismatch_score + self.beta * lambda_) + self.epsilon
        )
        return float(priority)
    
    def add(self, transition, lambda_, mismatch_score):
        action, reward, continued, obs = transition
        self.obs[self.count] = torch.as_tensor(obs)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.continued[self.count] = torch.as_tensor(continued)
        priority = self.get_priority(lambda_, mismatch_score)
        self.sumtree.add(priority, self.count)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.real_size + 1, self.size)
        
    def sample(self, batch_size):
        assert self.real_size >= batch_size
        sample_idx = []
        tree_idx = []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)
        segment = self.sumtree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            cumsum = random.uniform(a, b)
            tree_i, p, data_i = self.sumtree.get(cumsum)
            sample_idx.append(data_i)
            tree_idx.append(tree_i)
            priorities[i] = p
        
        probs = priorities / self.sumtree.total()
        weights = (self.real_size * probs) ** -self.beta
        weights /= weights.max()
        
        batch = (
            self.obs[sample_idx].to(device()),
            self.action[sample_idx].to(device()),
            self.reward[sample_idx].to(device()),
            self.continued[sample_idx].to(device())
        )
        
        return weights, tree_idx, batch
    
    def update(self, idx, lambda_, mismatch_score):
        new_priority = self.get_priority(lambda_, mismatch_score)
        self.sumtree.update(idx, new_priority)
        
