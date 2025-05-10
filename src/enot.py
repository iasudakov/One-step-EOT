import torch
import torch.nn as nn
import math
import numpy as np
import pdb
import torch.nn.functional as F
from torch_utils import persistence

from .D import SongUNet_D
from .G import SongUNet_G
from .f import SongUNet_f

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.silu(input)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()

        self.dim = dim
        self.scale = scale

        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        
        input = input*self.scale + 1
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb
    

    
@persistence.persistent_class
class f_wrapper(nn.Module):
    def __init__(self, img_resolution, model_channels, channel_mult, **kwargs):
        
        super().__init__()
        self.denoiser = SongUNet_f(img_resolution, 3, 3, model_channels, channel_mult=channel_mult)        
        self.n_steps = 10
        self.delta_t = 1/self.n_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x0, gamma = 0.0, traj = False):
        x = x0
        t = (torch.zeros(x0.shape[0])).to(self.device)
        trajectory = [x0]
        
        for step in range(self.n_steps):
            if step < self.n_steps - 1:
                x = x + self.delta_t*(self.denoiser(x, t) - x)/(1-torch.tensor(t)[:, None, None, None].cuda()) + torch.randn_like(x)*np.sqrt(gamma*self.delta_t)
            else:
                x = x + self.delta_t*(self.denoiser(x, t) - x)/(1-torch.tensor(t)[:, None, None, None].cuda())
            t += self.delta_t
            trajectory.append(x)
        if traj:
            return x, trajectory
        return x
    

@persistence.persistent_class
class G_wrapper(nn.Module):
    def __init__(self, img_resolution, model_channels, channel_mult, **kwargs):
        super().__init__()
        self.G = SongUNet_G(img_resolution, 4, 3, model_channels, channel_mult=channel_mult)
        
    def forward(self, x0):
        x = torch.cat([x0, torch.randn_like(x0[:, :1, :, :])], dim=1)
        xN = self.G(x)
        return xN
    

@persistence.persistent_class
class D_wrapper(nn.Module):
    def __init__(self, img_resolution, model_channels, channel_mult, **kwargs):
        super().__init__()
        self.D = SongUNet_D(img_resolution, 3, model_channels, channel_mult=channel_mult)
        
    def forward(self, x):
        return self.D(x)
 