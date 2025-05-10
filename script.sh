#!/bin/bash
#SBATCH --job-name=wild2cat_1
#SBATCH --output=outputs/wild2cat_1.log
#SBATCH --error=outputs/wild2cat_1.err
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint="type_a|type_b|type_c"

torchrun --standalone --nproc_per_node=1 train.py --name=mnist --outdir=outdir --data1train=datasets/2_train.zip --data2train=datasets/3_train.zip --data1test=datasets/2_test.zip --data2test=datasets/3_test.zip --data1stats=datasets/2_train.npz --data2stats=datasets/3_train.npz --batch=64 --batch-gpu=32 --G_iters=10 --D_iters=1 --f_iters=2 --samples_dir_G=samples_G_1 --samples_dir_SDE=samples_SDE_1 --gamma=1.0 --model_channels=32