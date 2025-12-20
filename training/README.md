# ReasonIR: Training

This directory contains scripts for contrastive learning to fine-tune Large Language Models (LLMs) for information retrieval tasks.

We use [gritlm](https://github.com/ContextualAI/gritlm) for contrastive training. We provide our script for training in `training/train.sh`.

Note: to convert llama to embedding model, you need to replace the `modeling_llama.py` in your transformers package with `training/modeling_llama.py` to support non-causal attention mask.