#! /bin/bash

cd evaluation/bright

tasks=("biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow" "sustainable_living" "leetcode" "pony" "aops" "theoremqa_theorems" "theoremqa_questions")  

# use this block to run reasonir and store the reranker scores
MODEL=reasonir
REASONING=gpt4
for TASK in "${tasks[@]}"; do
    python run.py --task $TASK --model $MODEL --output_dir output/${MODEL} --cache_dir cache --reasoning $REASONING
done

# use this block to run bm25 and store the bm25 scores for combining with the reranker scores
# If you want to interpolate with retriever scores, you don't need to run this block
# Loop through tasks and run the command
# for task in "${tasks[@]}"; do
#     echo "Running task: $task"
#     python run.py --task "$task" --model bm25 --output_dir output/bm25/ --reasoning gpt4 --store_all_scores
#     # --reasoning gpt4
# done

# use this block to run the reranker
# If you want to combine with bm25 scores run the commented block above and
# add --bm25_score_file "output/bm25/${task}_bm25_long_False/${REASONING}_score.json" to the python command below
for task in "${tasks[@]}"; do
    echo "Running task: $task"
    python reranker.py --task "$task" --retriever_score_file "output/${MODEL}/${task}_${MODEL}_long_False/${REASONING}_score.json" --input_k 100 --k 100 --output_dir "output/reranker/${task}_${MODEL}_long_False/" --reasoning $REASONING
done