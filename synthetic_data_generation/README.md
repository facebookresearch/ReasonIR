# ReasonIR: Synthetic Data Generation

This directory contains scripts for synthetic data generation.

## Setup
```bash
conda create -n reasonir python=3.10
conda activate reasonir
pip install -r requirements.txt
bash setup_java.sh
```

## Synthetic Data Generation

A simple example is provided in `script.sh`. In detail:
### Generate the queries based on the documents from the datastore of BRIGHT
```bash
python -m doc_to_query --model_id $MODEL   --queries_per_doc $queries_per_doc \
    --num_docs $num_docs  --subject $TASK  --output_dir $output_dir \
    --filter fineweb --prompt_id $prompt_id
```

### Generate the rewritten queries with reasoning given the queries
```bash
python -m generate_reasoning --model_id $MODEL  --num_docs $num_docs   --subject $TASK  \
    --base_dir $output_dir --prompt_id $prompt_id
```

### Batched version 
When generating data using the APIs, it is generally cheaper and faster to use the batch API. We provide a helper in `batch_api_helper.py` and scripts in `script_batch.sh`. In particular, the data synthesis steps can be performed via: 
```bash
python -m doc_to_query_batch --model_id $MODEL  --queries_per_doc $queries_per_doc \
    --num_docs $num_docs  --subject $TASK  --output_dir $output_dir \
    --filter fineweb --prompt_id $prompt_id

python -m doc_to_query_batch --model_id $MODEL  --queries_per_doc $queries_per_doc \
    --num_docs $num_docs  --subject $TASK  --output_dir $output_dir \
    --filter fineweb --prompt_id $prompt_id --gather_results

python -m generate_reasoning_batch --model_id $MODEL  --num_docs $num_docs   --subject $TASK  --base_dir $output_dir \
    --prompt_id $prompt_id

python -m generate_reasoning_batch --model_id $MODEL  --num_docs $num_docs   --subject $TASK  --base_dir $output_dir \
    --prompt_id $prompt_id --gather_results
```