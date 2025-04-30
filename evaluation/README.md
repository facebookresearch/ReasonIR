# ReasonIR: Evaluation

This directory contains evaluation scripts for Information Retrieval (IR) and Retrieval-Augmented Generation (RAG) tasks as described in our research paper.


## BRIGHT

Setup (same as for synthetic data generation):
```bash
conda create -n reasonir python=3.10
conda activate reasonir
pip install -r evaluation/bright/requirements.txt
bash synthetic_data_generation/setup_java.sh
```

To evaluate ReasonIR on BRIGHT, run
```bash
bash evaluation/bright/script.sh
```

To reproduce the results for some other baselines (such as Cohere and Voyage embeddings), please install other required packages via `pip install evaluation/bright/other_requirements.txt`.


## Downstream RAG evaluation

In order to reduce the cost of datastore construction, we first retrieve the top-1000 documents from the original [MassiveDS-1.4T](https://arxiv.org/abs/2407.12854) built with Contriever for each benchmark respectively. We then merge the retrieved documents as a new and smaller pool of datastore for experiments. To merge and deduplicate these documents, we use the script in `datastore/construct_datastore_corpus.py`.


To embed and index the filtered data with our retriever, run
```bash
git clone https://github.com/RulinShao/retrieval-scaling.git
bash evaluation/rag/datastore/build_datastore.sh
```
We then use the [MassiveDS codebase](https://github.com/RulinShao/retrieval-scaling) to search for the queries following the instructions in the [Datastore](#datastore).



### MMLU

To evaluate ReasonIR on MMLU, replace the data directories and run 
```bash
export retrieval_file=$YOUR_RETRIEVAL_FILE  # refer to REAMDE for more details
export raw_query_file=mmlu.jsonl  # refer to the original MMLU questions used for retrieval
bash evaluation/rag/mmlu_cot/scripts/eval_llama_3_8b_mmlu_rag.sh
```

### GPQA

First launch the LLM using vllm to obtain a local serving api:
```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --disable-cuda-graph --tp 1 --host 0.0.0.0
```

Then, run evaluation
```bash
cd evaluation/rag/gpqa
export RETRIEVED_FILE=YOUR_RETRIEVED_FILE
PYTHONPATH=. python src/main.py \
    --config-name naive_rag_default \
    model_path=Qwen/Qwen2.5-7B-Instructt \
    llm_endpoint=http://${VLLM_ENDPOINT}-${VLLM_PORT}:30000/v1 \
    top_k=5 \
    search_engine=offline_massiveds \
    use_query_rewriting=false \
    dataset_name=gpqa \
    split=diamond
```