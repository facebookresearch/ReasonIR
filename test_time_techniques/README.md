# ReasonIR: Test-Time Techniques---Query Rewriting and LLM Reranking

This directory contains scripts for the two techniques we used to further enhance ReasonIR's performance---query rewriting and LLM reranking.

To run query rewriting with token limit, run
```bash
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

MODEL=gpt-4o-mini
for TOKEN_LIMIT in 2048; do
    for TASK in "${tasks[@]}"
    do  
        echo $TASK
        echo $TOKEN_LIMIT
        PYTHONPATH=. python /home/rulin/bright-dev/reason.py \
            --task $TASK \
            --llm $MODEL \
            --output_token_limit $TOKEN_LIMIT
    done
done
```
