import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import sys
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

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

#inspired by: https://github.com/Nikolai10/FSQ
class FSQ(nn.Module):
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
class FSQ_(nn.Module):
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
    
"""Margen to pass the next phase (PHASE 1):
    1. Clear reconstruction
    3. latents stable across the frames"""
minimun_perplexity = 512 * 0.3
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((64, 64)),
    T.ToTensor()
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Device: {device}\n")
env = gym.make("CartPole-v1", render_mode="rgb_array")
encoder = Encoder(input_channels=3, hidden_layers=[64, 8]).to(device) #Change "hidden_layers=[128, 256, 256]"  
#vq = Vector_Quantizer(num_embeddings=128, embedding_dim=128).to(device)
fsq = FSQ(levels=[4]*8).to(device)
decoder = Decoder(input_channels = 8, hidden_layers = [64, 3]).to(device)

optimizer=torch.optim.AdamW(list(encoder.parameters())+
                            list(decoder.parameters())+
                            list(fsq.parameters()), 
                            lr=1e-3, 
                            weight_decay=1e-6)
total_epochs = 300
batch_size = 64

for i in range(total_epochs):
    frames = [] #change
    env.reset()
    for step in range(512): #Change
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated: # change
            env.reset()
        
        frames.append(transform(env.render()))
    
    frame = torch.stack(frames) #Change
    loss_ = 0
    for f in range(0, 512, batch_size): #Change
        mini_batch = frame[f : f + batch_size].to(device) #Change
        z_e = encoder(mini_batch)

        #z_e = F.normalize(z_e, dim=1) #REMOVED
        
        #z_e = z_e + torch.randn_like(z_e) * 0.1 #Experiment adding noise from 0.1 to 0.04
        zhat = fsq(z_e)
        recon = decoder(zhat)

        #recon = recon.clamp(0, 1) Removed
        recon_loss = F.mse_loss(recon, mini_batch)
        total_loss = recon_loss 

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loss_ += total_loss.item()
        
    #From google
    #encodings = F.one_hot(encoding_indices, vq.num_embeddings).float()
    #avg_probs = encodings.mean(dim=0)
    #perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    avg_loss = loss_ / (512/batch_size)
    print(f"Epoch: {i}, Loss: {avg_loss:.4f}")
    
    
    if i == 0:
        plt.imshow(frame[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.savefig('Raw_frame_epoch_0.png')
        plt.imshow(recon[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.savefig('reconstruction_epoch_0.png')
        
    elif i == 100:
        plt.imshow(frame[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.savefig('Raw_frame_epoch_100.png')
        plt.imshow(recon[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.savefig('reconstruction_epoch_100.png')
        
    elif i == 200:
        plt.imshow(frame[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.savefig('Raw_frame_epoch_200.png')
        plt.imshow(recon[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.savefig('reconstruction_epoch_200.png')
    
    elif i == total_epochs - 1:
        plt.imshow(frame[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.savefig('Raw_frame_final.png')
        plt.imshow(recon[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.savefig('reconstruction_final.png')
        torch.save(encoder.state_dict(), "encoder.pth")
        torch.save(decoder.state_dict(), "decoder.pth")
        torch.save(fsq.state_dict(), "fsq.pth")
        
"""
PHASE 1 COMPLETED!!!!!!
"""
