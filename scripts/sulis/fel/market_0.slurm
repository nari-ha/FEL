#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=16:00:00
#SBATCH --account=su008-ec23709

module load GCCcore/12.2.0 Python/3.10.8
cd /home/e/ec23709/project/FEL
source ./../reid/bin/activate
python train.py --config_file configs/fel/market_0.yml