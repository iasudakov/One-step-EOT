## One-step Entropic Optimal Transport for Unpaired Image-to-Image Translation <br><sub>Official PyTorch implementation</sub>

<!-- ![Teaser image](./docs/teaser-1920x640.jpg)

**Elucidating the Design Space of Diffusion-Based Generative Models**<br>
Tero Karras, Miika Aittala, Timo Aila, Samuli Laine
<br>https://arxiv.org/abs/2206.00364<br> -->

Abstract: *We introduce a novel neural network-based approach for solving the entropy-regularized optimal transport
(EOT) problem between two continuous distributions in a single
step. Constructing such a transport map has broad applications
in machine learning, including generative modeling, domain
adaptation, and image-to-image translation. Our method builds
upon the Schrodinger bridge (SB) formulation, widely used ¨
in prior research. However, unlike existing SB models, which
involve computationally expensive simulations for training and
inference, our approach enables simulation-free training and
one-step sample generation. Through empirical evaluation, we
demonstrate the effectiveness of our method on several toy EOT
tasks, highlighting its potential for scalability.*

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1+ high-end NVIDIA GPU for sampling and 8+ GPUs for training. We have done all testing and development using V100 and A100 GPUs.
* 64-bit Python 3.8 and PyTorch 1.12.0 (or later). See https://pytorch.org for PyTorch install instructions.
* Python libraries: See [environment.yml](./environment.yml) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda env create -f environment.yml -n eot`
  - `conda activate eot`


## Training new models

You can train new models using `train.py`. For example:

```.bash
# Train DDPM++ model for Color-MNIST 2 -> 3 using 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py --name=mnist --outdir=outdir --data1train=datasets/2_train.zip --data2train=datasets/3_train.zip --data1test=datasets/2_test.zip --data2test=datasets/3_test.zip --data1stats=datasets/2_train.npz --data2stats=datasets/3_train.npz --batch=64 --batch-gpu=16 --G_iters=10 --D_iters=1 --f_iters=2 --samples_dir_G=samples_G_1 --samples_dir_SDE=samples_SDE_1 --gamma=1.0 --model_channels=32
```




The above example uses the batch size of 64 images (controlled by `--batch`) that is divided evenly among 4 GPUs (controlled by `--nproc_per_node`) to yield 16 images per GPU. Training large models may run out of GPU memory; the best way to avoid this is to limit the per-GPU batch size, e.g., `--batch-gpu=16`. This employs gradient accumulation to yield the same results as using full per-GPU batches. See [`python train.py --help`](./docs/train-help.txt) for the full list of options.

The results of each training run are saved to a newly created directory, for example `training-runs/00000-mnist-uncond-ddpmpp-edm-gpus4-batch64-fp32`. The training loop exports network snapshots (`network-snapshot-*.pkl`) and training states (`training-state-*.pt`) at regular intervals (controlled by `--snap` and `--dump`). 