import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc


from IPython.display import clear_output
from matplotlib import pyplot as plt
from src.plotters import plot_trajectories, plot_images
import torch.nn.functional as F
import wandb
import gc
from fid import calculate_inception_stats, calculate_fid_from_inception_stats
from dnnlib.util import open_url
from src.enot import G_wrapper, D_wrapper, f_wrapper

from src.fid import save_model_samples


def calc_fid(image_path, ref_path, batch):
    with open_url(ref_path) as f:
        ref = dict(np.load(f))

    mu, sigma = calculate_inception_stats(image_path=image_path, max_batch_size=batch)
    fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
    return fid

#----------------------------------------------------------------------------

def training_loop(
    name                      = '', # Run name
    dataset1train_kwargs      = {}, # Options for training set.
    data_loader1train_kwargs  = {}, # Options for torch.utils.data.DataLoader.
    dataset2train_kwargs      = {}, # Options for training set.
    data_loader2train_kwargs  = {}, # Options for torch.utils.data.DataLoader.
    dataset1test_kwargs      = {},  # Options for training set.
    data_loader1test_kwargs  = {},  # Options for torch.utils.data.DataLoader.
    dataset2test_kwargs      = {},  # Options for training set.
    data_loader2test_kwargs  = {},  # Options for torch.utils.data.DataLoader.
    dataset1stats_path       = '',
    dataset2stats_path       = '',
    
    samples_dir_SDE          = 'samples_SDE',
    samples_dir_G            = 'samples_G',
    
    network_G_kwargs      = {},      # Options for model and preconditioning.
    optimizer_G_kwargs    = {},      # Options for optimizer.
    network_D_kwargs      = {},      # Options for model and preconditioning.
    optimizer_D_kwargs    = {},      # Options for optimizer.
    network_f_kwargs      = {},      # Options for model and preconditioning.
    optimizer_f_kwargs    = {},      # Options for optimizer.
    
    gamma                 = 0.0,
    
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    
    G_iters             = 10,
    D_iters             = 1,
    f_iters             = 2,
    
    
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    **kwargs
):
    # Initialize.
    run_dir = '.'
    best_fid = 1000
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset1train_obj = dnnlib.util.construct_class_by_name(**dataset1train_kwargs) # subclass of training.dataset.Dataset
    dataset1train_sampler = misc.InfiniteSampler(dataset=dataset1train_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset1train_iterator = iter(torch.utils.data.DataLoader(dataset=dataset1train_obj, sampler=dataset1train_sampler, batch_size=batch_gpu, **data_loader1train_kwargs))
    dataset1train_loader = torch.utils.data.DataLoader(dataset=dataset1train_obj, batch_size=batch_gpu, **data_loader1train_kwargs)


    dataset2train_obj = dnnlib.util.construct_class_by_name(**dataset2train_kwargs) # subclass of training.dataset.Dataset
    dataset2train_sampler = misc.InfiniteSampler(dataset=dataset2train_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset2train_iterator = iter(torch.utils.data.DataLoader(dataset=dataset2train_obj, sampler=dataset2train_sampler, batch_size=batch_gpu, **data_loader2train_kwargs))
    dataset2train_loader = torch.utils.data.DataLoader(dataset=dataset2train_obj, batch_size=batch_gpu, **data_loader2train_kwargs)


    dataset1test_obj = dnnlib.util.construct_class_by_name(**dataset1test_kwargs) # subclass of training.dataset.Dataset
    dataset1test_loader = torch.utils.data.DataLoader(dataset=dataset1test_obj, batch_size=batch_gpu, **data_loader1test_kwargs)

    dataset2test_obj = dnnlib.util.construct_class_by_name(**dataset2test_kwargs) # subclass of training.dataset.Dataset
    dataset2test_loader = torch.utils.data.DataLoader(dataset=dataset2test_obj, batch_size=batch_gpu, **data_loader2test_kwargs)



    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs_G = dict(img_resolution=dataset1train_obj.resolution, img_channels=dataset1train_obj.num_channels, label_dim=dataset1train_obj.label_dim)
    net_G = G_wrapper(**network_G_kwargs, **interface_kwargs_G)
    net_G.train().requires_grad_(True).to(device)

    interface_kwargs_D = dict(img_resolution=dataset1train_obj.resolution, img_channels=dataset1train_obj.num_channels, label_dim=dataset2train_obj.label_dim)
    net_D = D_wrapper(**network_D_kwargs, **interface_kwargs_D)
    net_D.train().requires_grad_(True).to(device)
    
    interface_kwargs_f = dict(img_resolution=dataset1train_obj.resolution, img_channels=dataset1train_obj.num_channels, label_dim=dataset2train_obj.label_dim)
    net_f = f_wrapper(**network_f_kwargs, **interface_kwargs_f)
    net_f.train().requires_grad_(True).to(device)
    


    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    optimizer_G = dnnlib.util.construct_class_by_name(params=net_G.parameters(), **optimizer_G_kwargs)
    ddp_G = torch.nn.parallel.DistributedDataParallel(net_G, device_ids=[device])
    ema_G = copy.deepcopy(net_G).eval().requires_grad_(False)
    
    optimizer_D = dnnlib.util.construct_class_by_name(params=net_D.parameters(), **optimizer_D_kwargs)
    ddp_D = torch.nn.parallel.DistributedDataParallel(net_D, device_ids=[device])
    ema_D = copy.deepcopy(net_D).eval().requires_grad_(False)
    
    optimizer_f = dnnlib.util.construct_class_by_name(params=net_f.parameters(), **optimizer_f_kwargs)
    ddp_f = torch.nn.parallel.DistributedDataParallel(net_f, device_ids=[device])
    ema_f = copy.deepcopy(net_f).eval().requires_grad_(False)
    
    
    size_model = 0
    for param in ema_D.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model D size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
    
    size_model = 0
    for param in ema_G.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model G size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
    
    size_model = 0
    for param in ema_f.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model f size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
    
    print(name)
    
    if dist.get_rank() == 0:
        wandb.init(project='One-step_EOT', name=name)
    
    step = 0
    cur_nimg = 0
    optimizer_G.zero_grad(set_to_none=True)
    optimizer_D.zero_grad(set_to_none=True)
    optimizer_f.zero_grad(set_to_none=True)
    while True:
        
        for G_iter in range(G_iters):
            
            for f_iter in range(f_iters):
                optimizer_f.zero_grad(set_to_none=True)
                for round_idx in range(num_accumulation_rounds):
                    with misc.ddp_sync(ddp_f, (round_idx == num_accumulation_rounds - 1)):
                        images, _ = next(dataset1train_iterator)
                        x0 = images.to(device).to(torch.float32) / 127.5 - 1
                        xN = ddp_G(x0)
                        
                        t = torch.rand(x0.shape[0]).to(device)
                        xt = x0 + (xN - x0) * t[:, None, None, None] + torch.randn_like(x0)*torch.sqrt(t*(1-t)*gamma)[:, None, None, None]
        
                        f_loss = ((net_f.denoiser(xt, t) - xN) ** 2).mean()
                        f_loss.mul(1 / num_accumulation_rounds).backward()
                        
                for g in optimizer_f.param_groups:
                    g['lr'] = optimizer_f_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
                for param in net_f.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                        param.grad = torch.clip(param.grad, -1.0, 1.0)
                optimizer_f.step()
            
            optimizer_G.zero_grad(set_to_none=True)
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(ddp_G, (round_idx == num_accumulation_rounds - 1)):
                    images, _ = next(dataset1train_iterator)
                    x0 = images.to(device).to(torch.float32) / 127.5 - 1
                    xN = ddp_G(x0)
                    
                    t = torch.rand(x0.shape[0]).to(device)
                    xt = x0 + (xN - x0) * t[:, None, None, None] + torch.randn_like(x0)*torch.sqrt(t*(1-t)*gamma)[:, None, None, None]
                    
                    f_x_t = (net_f.denoiser(xt, t) - xt)
                    E = (xN - xt)
    
                    G_loss = ((f_x_t*E).mean() - (f_x_t*f_x_t).mean()/2)*2 - net_D(xN).mean()
                    G_loss.mul(1 / num_accumulation_rounds).backward()
                    
            for g in optimizer_G.param_groups:
                g['lr'] = optimizer_G_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
            for param in net_G.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                    param.grad = torch.clip(param.grad, -1.0, 1.0)
            optimizer_G.step()
    

        for D_iter in range(D_iters):
            optimizer_D.zero_grad(set_to_none=True)
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(ddp_D, (round_idx == num_accumulation_rounds - 1)):
                    images, _ = next(dataset1train_iterator)
                    x0 = images.to(device).to(torch.float32) / 127.5 - 1
                    images, _ = next(dataset2train_iterator)
                    x1 = images.to(device).to(torch.float32) / 127.5 - 1
                    
                    xN = ddp_G(x0)
                    D_loss = (- net_D(x1) + net_D(xN)).mean()
                    D_loss.mul(1 / num_accumulation_rounds).backward()
                    
            for g in optimizer_D.param_groups:
                g['lr'] = optimizer_D_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
            for param in net_D.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                    param.grad = torch.clip(param.grad, -1.0, 1.0)
            optimizer_D.step()


        if dist.get_rank() == 0:
            wandb.log({f'f_loss' : f_loss.item()}, step=step)
            wandb.log({f'G_loss' : G_loss.item()}, step=step)
            wandb.log({f'D_loss' : D_loss.item()}, step=step)
            
            
        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema_G.parameters(), net_G.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
        for p_ema, p_net in zip(ema_D.parameters(), net_D.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
        for p_ema, p_net in zip(ema_f.parameters(), net_f.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
            
        
        cur_nimg += batch_size

        if dist.get_rank() == 0 and step % 250 == 0:
            clear_output(wait=True)
            
            with torch.no_grad():
                images, _ = next(dataset1train_iterator)
                X = images[:batch_gpu].to(device).to(torch.float32) / 127.5 - 1

                T_XZ_np = []
                for i in range(10):
                    T_XZ_np.append(ema_G(X).cpu().numpy())
                T_XZ_np = np.array(T_XZ_np)
                wandb.log({f'G var' : T_XZ_np.var(axis=0).mean().item()}, step=step)

                T_X_np = []
                for i in range(10):
                    T_X_np.append(ema_f(X, gamma).cpu().numpy())
                T_X_np = np.array(T_X_np)
                wandb.log({f'sde var' : T_X_np.var(axis=0).mean().item()}, step=step)
                
                G_dataset = ema_G(X).detach()
                f_dataset = ema_f(X).detach()
                wandb.log({f'G L2' : torch.sqrt(F.mse_loss(X.detach(), G_dataset)).item()}, step=step)
                wandb.log({f'sde L2' : torch.sqrt(F.mse_loss(X.detach(), f_dataset)).item()}, step=step)
                
                fig1 = plot_trajectories(ema_f, gamma, dataset1train_iterator, 3)
                wandb.log({"trajectories": wandb.Image(fig1)}, step=step)
                plt.close(fig1)
                torch.cuda.empty_cache(); gc.collect()

                fig2 = plot_images(ema_G, dataset1train_iterator, 4, 4)
                wandb.log({"G_images": wandb.Image(fig2)}, step=step)
                plt.close(fig2)
                torch.cuda.empty_cache(); gc.collect()
                
                fig3 = plot_images(ema_f, dataset1train_iterator, 4, 4, gamma)
                wandb.log({"SDE_images": wandb.Image(fig3)}, step=step)
                plt.close(fig3)
                torch.cuda.empty_cache(); gc.collect()
                
                save_model_samples(f'samples/{name}_{samples_dir_G}_test', f'samples/{name}_{samples_dir_SDE}_test', ema_G, ema_f, dataset1test_loader, gamma)
                fid_G = calc_fid(f'samples/{name}_{samples_dir_G}_test', dataset2stats_path, batch_gpu)
                wandb.log({f'FID_G_test' : fid_G}, step=step)
                fid_SDE = calc_fid(f'samples/{name}_{samples_dir_SDE}_test', dataset2stats_path, batch_gpu)
                wandb.log({f'FID_SDE_test' : fid_SDE}, step=step)
                
                save_model_samples(f'samples/{name}_{samples_dir_G}_train', f'samples/{name}_{samples_dir_SDE}_train', ema_G, ema_f, dataset1train_loader, gamma)
                fid_G = calc_fid(f'samples/{name}_{samples_dir_G}_train', dataset2stats_path, batch_gpu)
                wandb.log({f'FID_G_train' : fid_G}, step=step)
                fid_SDE = calc_fid(f'samples/{name}_{samples_dir_SDE}_train', dataset2stats_path, batch_gpu)
                wandb.log({f'FID_SDE_train' : fid_SDE}, step=step)
                
                if fid_G < best_fid:
                    best_fid = fid_G
                    data = dict(ema_f=ema_f, ema_G=ema_G, ema_D=ema_D)
                    for key, value in data.items():
                        if isinstance(value, torch.nn.Module):
                            value = copy.deepcopy(value).eval().requires_grad_(False)
                            data[key] = value.cpu()
                        del value # conserve memory
                    if dist.get_rank() == 0:
                        with open(os.path.join(run_dir, f'network-snapshot-{gamma}_{name}-{step//1000:06d}.pkl'), 'wb') as f:
                            pickle.dump(data, f)
                    del data # conserve memory
                
                torch.cuda.empty_cache(); gc.collect()
                
            
        gc.collect(); torch.cuda.empty_cache()
            
        if dist.get_rank() == 0:        
            step+=1
            print(step)
        
#----------------------------------------------------------------------------
