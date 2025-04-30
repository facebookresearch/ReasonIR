#! /bin/bash

## BRIGHT tasks ##
tasks=(
    biology
    earth_science
    economics
    psychology
    robotics
    stackoverflow
    sustainable_living
    leetcode 
    pony 
    aops 
    theoremqa_theorems 
    theoremqa_questions
)


MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
queries_per_doc=1
num_docs=10
prompt_id=hq_gen
output_dir=synthetic_data/$MODEL/hq

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Loop through each task and run the script
for TASK in "${tasks[@]}"; do
    python -m doc_to_query  --model_id $MODEL  --queries_per_doc $queries_per_doc  --num_docs $num_docs  --subject $TASK  --output_dir $output_dir  --filter fineweb  --prompt_id $prompt_id
    python -m generate_reasoning  --model_id $MODEL  --num_docs $num_docs  --subject $TASK  --base_dir $output_dir  --prompt_id $prompt_id
done