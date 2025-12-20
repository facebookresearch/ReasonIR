#! /bin/bash

cd evaluation/bright

MODEL=reasonir
REASONING=gpt4
BS=-1
for TASK in biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_theorems theoremqa_questions; do
    python run.py --task $TASK --model $MODEL --output_dir output/${MODEL}_${REASONING}_reasoning --cache_dir cache --reasoning $REASONING --encode_batch_size $BS
done