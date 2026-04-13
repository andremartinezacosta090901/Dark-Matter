import numpy as np
import torch
import torch.nn as nn
from ncps.torch import CfC
def symlog(x):
    return torch.sign(x) * torch.log(x.abs() + 1)

def symexp(x):
    return torch.sign(x) * (torch.exp(x.abs()) - 1)

def get_stochastic_state(logits, stoch_dim=32, discrete_dim=32, mix_ratio=0.01):
        logits = logits.view(-1, stoch_dim, discrete_dim)
        probs = torch.softmax(logits, dim=-1)
        if mix_ratio > 0:
            uniform = torch.ones_like(probs) / discrete_dim
            probs = (1 - mix_ratio) * probs + mix_ratio * uniform
        dist = torch.distributions.OneHotCategorical(probs=probs)
        sample = dist.sample()
        sample = sample + (probs - probs.detach())
        z_flat = sample.view(logits.shape[0], -1)
        return z_flat, logits, dist
    
class StochasticNetwork(nn.Module):
    def __init__(self, in_channel, stoch_dim=32, discrete_dim=32, hidden_layers=200, activation='ReLU'):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.discrete_dim = discrete_dim
        output = stoch_dim* discrete_dim
        self.net = model_builder_layer(
            in_channel=in_channel,
            hidden_lay=[hidden_layers],
            output_channel=output,
            norm=True,
            zeroLastLayer=False,
            activation=activation
        )
        
    def forward(self, x):
        logits = self.net(x)
        z_sample, _, dist = get_stochastic_state(logits, stoch_dim=self.stoch_dim, discrete_dim=self.discrete_dim)
        return z_sample, dist
    
'This section of code is based on sequentialModel1D of https://github.com/InexperiencedMe/NaturalDreamer/blob/main/utils.py (I am big fan of unexperienced me fr!!)'
def model_builder_layer(in_channel, output_channel, hidden_lay=[64], norm=True, zeroLastLayer=False, activation='relu'):    
    init_mode = activation
    layers = []
    current_channels = in_channel
    
    if isinstance(hidden_lay, int):
        hidden_lay = [hidden_lay]
    
    for hidden_layer in hidden_lay:
        linear_layer = nn.Linear(current_channels, hidden_layer)
        nn.init.kaiming_normal_(linear_layer.weight, nonlinearity=init_mode.lower())
        nn.init.constant_(linear_layer.bias, 0.0)
        layers.append(linear_layer)
        
        if norm:
            layers.append(nn.LayerNorm(hidden_layer))
        
        layers.append(getattr(nn, init_mode)())
        current_channels = hidden_layer
    
    last_layer = nn.Linear(current_channels, output_channel)
    if zeroLastLayer:
        nn.init.uniform_(last_layer.weight, -0.001, 0.001)
        nn.init.uniform_(last_layer.bias, -0.001, 0.001) 
    else:
        nn.init.orthogonal_(last_layer.weight, gain=1)
        nn.init.constant_(last_layer.bias, 0.0)
        
    layers.append(last_layer)

    return nn.Sequential(*layers)

#All this section of code has parts copied from: https://github.com/airalcorn2/vqvae-pytorch/blob/master/vqvae.py#L139
class ExponentialMovingAverage(nn.Module):
    def __init__(self, 
                decay, 
                shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer('hidden', torch.zeros(shape))
        self.register_buffer('avg', torch.zeros(shape))
    
    def calculation(self, x):
        self.counter += 1
        with torch.no_grad():
            delta = (self.hidden - x) * (1-self.decay)
            self.hidden -= delta
            correction_factor = 1 - self.decay ** self.counter
            self.avg = self.hidden / correction_factor

    def forward(self, value):
        self.calculation(value)
        return self.avg

'This section of the code were highly based on https://github.com/NM512/dreamerv3-torch/blob/main/networks.py#L754  and also use the incredible liquid layers work from https://github.com/mlech26l/ncps. Thank you! :D'
#Then I should test: nn.GRU from pytorch VS my GLU
class GLU(nn.Module): #Gated Liquid Unit :)
    def __init__(self, input_size, size, norm=True, activation='ReLU', zeroLastLayer=False):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(size*2, size), nn.Sigmoid())
        nn.init.constant_(self.gate[0].bias, 0.0)
        layers = [nn.Linear(input_size, size)]
        
        if norm:
            layers.append(nn.LayerNorm(size, eps=1e-3)) #random eps must be tested
        layers.append(getattr(nn, activation)())
        self.layers = nn.Sequential(*layers)
        self.liquid_net = CfC(
            input_size=size,
            units=size, 
            proj_size=size,
            return_sequences=True,
            mode='pure', # modes avaible: 'default', 'pure', 'no_gate'
            backbone_layers=2,
            backbone_units=32,
            backbone_dropout=0.05)
        
        if zeroLastLayer:
            #Input layer 
            nn.init.uniform_(self.gate[0].weight, -0.001, 0.001)
            nn.init.uniform_(self.gate[0].bias, 1) 
            
            #Liquid layer
            for name, param in self.liquid_net.named_parameters():
                if 'rnn.projector' in name or 'rnn.output' in name:
                    nn.init.uniform_(param, -0.001, 0.001)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
        else:
            #Input layer
            nn.init.orthogonal_(self.gate[0].weight, gain=1)
            nn.init.constant_(self.gate[0].bias, 0.0)
            
            #Liquid layer
            for name, param in self.liquid_net.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
                    
    def forward(self, x, h, t=1.0, reset=None):
        # x shape: (Batch, Input_Size)
        # h shape: (Batch, Hidden_Size)
        if reset is not None:
            h = h*(1-reset)
        
        features = self.layers(x)
        gate = self.gate(torch.cat([features, h], dim=1))
        B, H = h.shape
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        
        if t.numel() == 1:
            ts = t.expand(B, 1, H)
        elif t.ndim == 1:
            ts = t.unsqueeze(1).expand(B, 1, H)
        else:
            ts = t.view(B, 1, 1).expand(B, 1, H)
                
        gate = gate * torch.clamp(ts[:, 0, 0:1], 1)
        features = features.unsqueeze(1)
        _, h_delta = self.liquid_net(features, hx=h, timespans=ts)
        if h_delta.ndim == 3:
            h_delta = h_delta.squeeze(1)
        h_new = ((1 - gate) * h + gate * h_delta)
        return h_new, h_new

class CFC(nn.Module): #BUILD LIQUID ENHANCED NETWORK FOR CRITIC IT WILL BE ADD WITH COMPUTE LABAMDA
    def __init__(self, input_size, size, norm=True, activation='ReLU', zeroLastLayer=False):
        super().__init__()
        self.in_layers = nn.Linear(input_size, size)
        sequential_block = []
        if norm:
            sequential_block.append(nn.LayerNorm(size, eps=1e-3))#random eps must be tested
        sequential_block.append(getattr(nn, activation)())
        self.features = nn.Sequential(*sequential_block)
        
        self.liquid_net = CfC(
            input_size=size,
            units=size, 
            proj_size=size,
            return_sequences=True,
            mode='pure', # modes avaible: 'default', 'pure', 'no_gate'
            backbone_layers=2,
            backbone_units=32,
            backbone_dropout=0.05)
        
        if zeroLastLayer:
            #Input layer 
            nn.init.uniform_(self.in_layers.weight, -0.001, 0.001)
            nn.init.uniform_(self.in_layers.bias, -0.001, 0.001) 
            
            #Liquid layer
            for name, param in self.liquid_net.named_parameters():
                if 'rnn.projector' in name or 'rnn.output' in name:
                    nn.init.uniform_(param, -0.001, 0.001)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
        else:
            #Input layer
            nn.init.orthogonal_(self.in_layers.weight, gain=1)
            nn.init.constant_(self.in_layers.bias, 0.0)
            
            #Liquid layer
            for name, param in self.liquid_net.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=1)
                
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
            
    def forward(self, x, h, t=1):
        x = self.in_layers(x)
        features = self.features(x)
    
        B, H = h.shape
        features = features.unsqueeze(1) # (B, 1, H)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        if t.numel() == 1:
            ts = t.expand(B, 1, H)
        else:
            ts = t.view(B, 1, 1).expand(B, 1, H)
            
        _, h_next = self.liquid_net(features, hx=h, timespans=ts)
        return h_next
        
'Copy-pasted of Compute Lambda Values from: https://github.com/InexperiencedMe/NaturalDreamer/blob/main/utils.py thanks you so much for teach me a lot of things by your code!!'
def ComputeLambdaValues(rewards, values, continues, lambda_=0.95):
    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]
    for t in reversed(range(rewards.shape[-1])):
        returns[:, t] = rewards[:, t] + continues[:, t] * ((1 - lambda_) * values[:, t] + lambda_ * bootstrap)
        bootstrap = returns[:, t]
    return returns


'Copy-past of Sumtree algorithm implementation from: https://github.com/rlcode/per/blob/master/SumTree.py (thanks for your incredible implementation!!)'
class Sum_Tree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.entries = 0
        
    def propagate(self, idx, change):
        parents = (idx - 1) // 2
        self.tree[parents] += change
        
        if parents != 0:
            self.propagate(parents, change)
            
    def retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])
        
    def total(self):
        return self.tree[0]
    
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        if self.entries < self.capacity:
            self.entries += 1
            
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self.propagate(idx, change)
    
    def get(self, s):
        idx = self.retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]
    
'Copy-pasted of device function from: https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/utils.py'
def device(force_cpu=True):
    return "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"