#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=fine-tune-contriever
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --output=output/fine_tune_contriever_%A.out

module purge
module load 2025
module load CUDA/12.6.0
module load Anaconda3/2025.06-1

source activate tevatron

cd /home/scur1731/tevatron

pip install -e .

cd /home/scur1731/ReasonIR/models/RetroMAE

python -m torch.distributed.launch --nproc_per_node=2 -m tevatron.retriever.driver.train \
  --output_dir models \
  --dataset_name cmpatino/reasonir-data \
  --dataset_config all \
  --model_name_or_path Shitao/RetroMAE_MSMARCO_finetune \
  --do_train \
  --save_steps 5000 \
  --fp16 False \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --attn_implementation eager \
  --run_name fine-tune-contriever \
  --report_to tensorboard \
  --logging_dir /home/scur1731/ReasonIR/models/RetroMAE/output/tensorboard/fine_tune_retromae \
  --logging_strategy steps \
  --logging_steps 50
