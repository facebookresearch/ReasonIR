#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --job-name=cb-qwen3-8b-evals
#SBATCH --output=./logs_eval/cb-qwen3-8b-evals.%j.out
#SBATCH --error=./logs_eval/cb-qwen3-8b-evals.%j.err
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00

set -euo pipefail

source .venv/bin/activate

## User-configurable variables
MODEL="Qwen/Qwen3-8B"
PORT=${PORT:-30000}

# Determine node IP (prefer first non-loopback IPv4)
NODE_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$NODE_IP" ]; then
  NODE_IP=$(getent hosts ${SLURMD_NODENAME:-$(hostname)} | awk '{print $1}' | head -n1)
fi
if [ -z "$NODE_IP" ]; then
  echo "Failed to detect node IP. Exiting." >&2
  exit 1
fi

echo "Starting vLLM OpenAI server for $MODEL on ${NODE_IP}:${PORT}..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$NODE_IP" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max_model-len 16392 \
  --gpu_memory_utilization 0.9 \
  > ./logs_eval/vllm_server_${SLURM_JOB_ID:-manual}.log 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "Stopping vLLM server (PID=$VLLM_PID)..."
  if ps -p $VLLM_PID > /dev/null 2>&1; then
    kill $VLLM_PID || true
  fi
}
trap cleanup EXIT

echo "Waiting for server to become ready..."
sleep 150

echo "Running GPQA (closed-book, reasoning)"
pushd evaluation/rag/gpqa > /dev/null
PYTHONPATH=. python src/main.py \
  --config-name qwen_3_think \
  model_path="$MODEL" \
  llm_endpoint="http://${NODE_IP}:${PORT}/v1" \
  dataset_name=gpqa \
  split=diamond \
  top_k=0 \
  closed_book=true \
  search_engine=offline_massiveds \
  use_query_rewriting=false \
  reasoning=true
popd > /dev/null

echo "Running GPQA (closed-book, non-reasoning)"
pushd evaluation/rag/gpqa > /dev/null
PYTHONPATH=. python src/main.py \
  --config-name qwen_3_no-think \
  model_path="$MODEL" \
  llm_endpoint="http://${NODE_IP}:${PORT}/v1" \
  dataset_name=gpqa \
  split=diamond \
  top_k=0 \
  closed_book=true \
  search_engine=offline_massiveds \
  use_query_rewriting=false \
  reasoning=false
popd > /dev/null

echo "Stopping vLLM server before MMLU to free GPU memory..."
if ps -p $VLLM_PID > /dev/null 2>&1; then
  kill $VLLM_PID || true
  sleep 5
fi

echo "Running MMLU (closed-book, reasoning)"
python evaluation/rag/mmlu_cot/evaluate_from_local_mmlu.py \
  --selected_subjects all \
  --save_dir eval_results/qwen3_0_6b_closedbook \
  --model "$MODEL" \
  --gpu_util 0.9 \
  --closed_book \
  --concat_k 0 \
  --reasoning

echo "Running MMLU (closed-book, non-reasoning)"
python evaluation/rag/mmlu_cot/evaluate_from_local_mmlu.py \
  --selected_subjects all \
  --save_dir eval_results/qwen3_0_6b_closedbook \
  --model "$MODEL" \
  --gpu_util 0.9 \
  --closed_book \
  --concat_k 0

echo "All evaluations completed."
