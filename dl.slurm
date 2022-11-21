#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=cifar-10
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=cifar10.%j


ml GCCcore/10.2.0 Python/3.8.6
cd "$SCRATCH" || exit
source project/bin/activate
cd CIFAR-10/code || exit
python3 main.py train