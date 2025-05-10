import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--name',          help='Run name', metavar='STR',                                    type=str, required=True)
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data1train',    help='Path to the dataset1', metavar='ZIP|DIR',                    type=str, required=True)
@click.option('--data2train',    help='Path to the dataset2', metavar='ZIP|DIR',                    type=str, required=True)
@click.option('--data1test',     help='Path to the dataset1', metavar='ZIP|DIR',                    type=str, required=True)
@click.option('--data2test',     help='Path to the dataset2', metavar='ZIP|DIR',                    type=str, required=True)
@click.option('--data1stats',    help='Path to the dataset1', metavar='ZIP|DIR',                    type=str, required=True)
@click.option('--data2stats',    help='Path to the dataset2', metavar='ZIP|DIR',                    type=str, required=True)
@click.option('--samples_dir_SDE',   help='Directory to save samples of SDE', metavar='STR',        type=str, default='samples', show_default=True)
@click.option('--samples_dir_G',   help='Directory to save samples of G', metavar='STR',            type=str, default='samples', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--G_iters',       help='G iters', metavar='INT',                                     type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--D_iters',       help='D iters', metavar='INT',                                     type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--f_iters',       help='f iters', metavar='INT',                                     type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--model_channels',       help='model channels', metavar='INT',                       type=click.IntRange(min=1), default=96, show_default=True)
@click.option('--gamma',         help='GAMMA', metavar='FLOAT',                                     type=click.FloatRange(min=-1.0, min_open=True), default=0.0, show_default=True)

@click.option('--lr_g',          help='Learning rate G', metavar='FLOAT',                           type=click.FloatRange(min=0, min_open=True), default=1e-4, show_default=True)
@click.option('--lr_D',          help='Learning rate D', metavar='FLOAT',                           type=click.FloatRange(min=0, min_open=True), default=1e-4, show_default=True)
@click.option('--lr_f',          help='Learning rate f', metavar='FLOAT',                           type=click.FloatRange(min=0, min_open=True), default=1e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)

# Performance-related.
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    
    c.name = opts.name
    c.G_iters = opts.g_iters
    c.D_iters = opts.d_iters
    c.f_iters = opts.f_iters
    c.model_channels = opts.model_channels
    c.gamma = opts.gamma
    c.samples_dir_SDE = opts.samples_dir_sde
    c.samples_dir_G = opts.samples_dir_g
    
    c.dataset1train_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data1train, use_labels=False, xflip=False, cache=opts.cache)
    c.dataset2train_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data2train, use_labels=False, xflip=False, cache=opts.cache)
    c.dataset1test_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data1test, use_labels=False, xflip=False, cache=opts.cache)
    c.dataset2test_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data2test, use_labels=False, xflip=False, cache=opts.cache)
    c.data_loader1train_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.data_loader2train_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.data_loader1test_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.data_loader2test_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2) 
    
    c.dataset1stats_path=opts.data1stats
    c.dataset2stats_path=opts.data2stats

    c.network_G_kwargs = dnnlib.EasyDict()
    c.optimizer_G_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr_g, betas=[0.9,0.999], eps=1e-8)

    c.network_D_kwargs = dnnlib.EasyDict()
    c.optimizer_D_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr_d, betas=[0.9,0.999], eps=1e-8)
    
    c.network_f_kwargs = dnnlib.EasyDict()
    c.optimizer_f_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr_f, betas=[0.9,0.999], eps=1e-8)

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset1train_kwargs)
        c.dataset1train_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset1train_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset2train_kwargs)
        c.dataset2train_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset2train_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset1test_kwargs)
        c.dataset1test_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset1test_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset2test_kwargs)
        c.dataset2test_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset2test_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    c.network_G_kwargs.update(model_type='SongUNet_G', encoder_type='standard', decoder_type='standard')
    c.network_G_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=opts.model_channels, channel_mult=[2,2,2])

    c.network_D_kwargs.update(model_type='SongUNet_D', encoder_type='standard', decoder_type='standard')
    c.network_D_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=int(opts.model_channels//32 + 1)*32, channel_mult=[2,2,2])
    
    c.network_f_kwargs.update(model_type='SongUNet_f', encoder_type='standard', decoder_type='standard')
    c.network_f_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=opts.model_channels, channel_mult=[2,2,2])

    # Preconditioning & loss function.
    c.network_G_kwargs.class_name = 'src.enot.G_wrapper'
    c.network_D_kwargs.class_name = 'src.enot.D_wrapper'
    c.network_f_kwargs.class_name = 'src.enot.f_wrapper'
    
    
    

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    if opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    desc = f'gpus{dist.get_world_size():d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------