

save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
selected_subjects="all"
gpu_util=0.95


CUDA_VISIBLE_DEVICES=0 python evaluate_from_local_mmlu.py \
                 --selected_subjects $selected_subjects \
                 --save_dir $save_dir \
                 --model $model \
                 --global_record_file $global_record_file \
                 --gpu_util $gpu_util \
                 --retrieval_file $retrieval_file \
                 --raw_query_file $raw_query_file \
                 --concat_k 3
