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


# Loop through each task and run the script by adding
# for TASK in "${tasks[@]}"; do
# done

# This is a demo script that only runs for documents in the datastore of the biology task
TASK=biology
MODEL=gpt-4o
queries_per_doc=1
num_docs=10
prompt_id=hq_gen
output_dir=synthetic_data/$MODEL/hq

export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m doc_to_query_batch --model_id $MODEL --queries_per_doc $queries_per_doc --num_docs $num_docs --subject $TASK --output_dir $output_dir --filter fineweb --prompt_id $prompt_id
python -m doc_to_query_batch --model_id $MODEL --queries_per_doc $queries_per_doc --num_docs $num_docs --subject $TASK --output_dir $output_dir --filter fineweb --prompt_id $prompt_id --gather_results
python -m generate_reasoning_batch --model_id $MODEL --num_docs $num_docs --subject $TASK --base_dir $output_dir --prompt_id $prompt_id
python -m generate_reasoning_batch --model_id $MODEL --num_docs $num_docs --subject $TASK --base_dir $output_dir --prompt_id $prompt_id --gather_results