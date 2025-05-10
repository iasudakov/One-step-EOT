import torch
import matplotlib.pyplot as plt
import numpy as np


def q_sample(step, p, k, x, sigma_max = 2, sigma_min = 0.2):
    t = 1 - (p*int(step/p)+1)/k
    e = 1 - np.exp(-(sigma_max-sigma_min)*t*t/2 - sigma_min*t)
    
    return np.sqrt(1 - e)*x + np.sqrt(e)*torch.randn_like(x)


def plot_images(T, sampler, n_x, n_samples, gamma=None):
    
    images = next(sampler)[0][:n_x]
    X = images.cuda().to(torch.float32) / 127.5 - 1
    
    if gamma:
        T_X = torch.stack([T(X, gamma) for i in range(n_samples)], dim=1)
    else:
        T_X = torch.stack([T(X) for i in range(n_samples)], dim=1)
    T_X = T_X.to('cpu').detach().permute(0,1,3,4,2).mul(0.5).add(0.5).numpy().clip(0,1)
    X = X.permute(0,2,3,1).mul(0.5).add(0.5).to('cpu').numpy().clip(0,1)
    
    fig, axes = plt.subplots(n_x, n_samples+1, figsize=(n_samples * 1.8, n_x * 1.5), dpi=450)

    
    for i in range(n_x):
        ax = axes[i][0]
        ax.imshow(X[i])
        
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
        if i == 0:
            ax.set_title(rf"$x0$")
    
    for i in range(n_x):
        for j in range(1, n_samples+1):
            ax = axes[i][j]
            ax.imshow(T_X[i][j-1])
            
            ax.get_xaxis().set_visible(False)
            ax.set_yticks([])
            
            if i == 0:
                ax.set_title(rf"$G(x0)$")
            
    fig.tight_layout(pad=0.001)
    return fig


    
def plot_trajectories(T, gamma, sampler, n_x):
    n_samples = 10
    fig, axes = plt.subplots(n_x, n_samples+1, figsize=(6.7, 2), dpi=400)
    
    for i in range(n_x):
        images = next(sampler)[0][:1]
        X = images.cuda().to(torch.float32) / 127.5 - 1
        T_X = torch.stack(T(X, gamma, traj=True)[1], dim=1)[:, 1:]
        T_X = T_X.to('cpu').detach().permute(0,1,3,4,2).mul(0.5).add(0.5).numpy().clip(0,1)
        X_img = X.permute(0,2,3,1).mul(0.5).add(0.5).to('cpu').numpy().clip(0,1)
        
        ax = axes[i][0]
        ax.imshow(X_img[0])
        
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
        if i == 0:
            ax.set_title(r"$X \sim \mathbb{P}_0$")
        
        for j in range(0, n_samples):
            ax = axes[i][j+1]
            ax.imshow(T_X[0][j])
            
            ax.get_xaxis().set_visible(False)
            ax.set_yticks([])
            
            if i == 0:
                ax.set_title(fr"t={round(0.1*(j+1), 1)}")
            
    fig.tight_layout(pad=0.001)
    return fig