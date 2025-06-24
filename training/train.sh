#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --account fair_amaia_cw_explore
#SBATCH --qos explore
#SBATCH --mem 1000G
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 120:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --requeue
#SBATCH --chdir=/home/rulin/gritlm-dev/gritlm
#SBATCH --output=/checkpoint/amaia/explore/rulin/gritlm/slurm_cache/slurm-%A_%a.out
#SBATCH --array=0

######################
### Set enviroment ###
######################
cd /home/rulin/gritlm-dev/gritlm
source /home/rulin/miniconda3/bin/activate
conda activate grit
export HF_HOME=/checkpoint/amaia/explore/rulin/cache/.cache/huggingface
#NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_PROJECT="grit"
# Training setup
GPUS_PER_NODE=8
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################



LAUNCHER="accelerate launch \
    --config_file /home/rulin/gritlm-dev/scripts/configs/config_128gpusfsdp_llama.yml \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --num_processes $WORLD_SIZE \
    --machine_rank \$SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 \
    --tee 3 \
    "


TRAIN_DATA=data/ # replace with the directory of your training data

# please refer to https://github.com/ContextualAI/gritlm/blob/main/gritlm/training/run.py for training scripts (e.g., run.py)
export CMD=" \
    -m training.run \
    --output_dir checkpoints/$(date "+%Y-%m-%d-%H_%M_%S")/ \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --train_data $TRAIN_DATA \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.06 \
    --max_steps 1000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --dataloader_drop_last \
    --normalized \
    --temperature 0.02 \
    --train_group_size 2 \
    --negatives_cross_device \
    --query_max_len 2048 \
    --passage_max_len 2048 \
    --mode embedding \
    --logging_steps 1 \
    --bf16 \
    --pooling_method mean \
    --use_unique_indices \
    --loss_gen_factor 2 \
    --attn bbcc \
    --gradient_checkpointing \
    --attn_implementation sdpa \
    --split_emb \
    --save_steps 500 
    "

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD" 2>&1
