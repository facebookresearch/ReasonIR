# First, launch
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --disable-cuda-graph --tp 1 --host 0.0.0.0



PYTHONPATH=. python src/main.py \
    --config-name naive_rag_default \
    model_path=Qwen/Qwen2.5-7B-Instructt \
    llm_endpoint=http://rulin@a100-st-p4de24xlarge-435:30000/v1 \
    top_k=5 \
    search_engine=offline_massiveds \
    use_query_rewriting=false \
    dataset_name=gpqa \
    split=diamond