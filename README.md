# ReasonIR

ReasonIR-8B is the first retriever specifically trained for general reasoning tasks, achieving the state-of-the-art retrieval performance on BRIGHT (reasoning-intensive retrieval). 
When employed for retrieval-augmented generation (RAG), ReasonIR-8B also brings substantial gains on MMLU and GPQA.

- Model: https://huggingface.co/reasonir/ReasonIR-8B 
- Paper: https://arxiv.org/abs/2504.20595

## General Usage
Make sure to install `transformers>=4.47.0` first!

### Transformers

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("reasonir/ReasonIR-8B", torch_dtype="auto", trust_remote_code=True)

query = "The quick brown fox jumps over the lazy dog."
document = "The quick brown fox jumps over the lazy dog."
query_instruction = ""
doc_instruction = ""
model = model.to("cuda")
model.eval()
query_emb = model.encode(query, instruction=query_instruction)
doc_emb = model.encode(document, instruction=doc_instruction)
sim = query_emb @ doc_emb.T
```

When using `AutoModel`, it is important to: 

1. Include `trust_remote_code=True` to make sure our custom bidirectional encoding architecture is used.
2. Use `torch_dtype="auto"` so that `bf16` is activated (by default torch will use `fp32`).

### Sentence Transformers

Ordinary retrieval models that use mean pooling can automatically be used with SentenceTransformer after being published on Huggingface. 

```python
from sentence_transformers import SentenceTransformer
model_kwargs = {"torch_dtype": "auto"}
model = SentenceTransformer("reasonir/ReasonIR-8B", trust_remote_code=True, model_kwargs=model_kwargs)
model.set_pooling_include_prompt(include_prompt=False) # exclude the prompt during pooling

query = "The quick brown fox jumps over the lazy dog."
document = "The quick brown fox jumps over the lazy dog."
query_instruction = ""
doc_instruction = ""
query_emb = model.encode(query, instruction=query_instruction)
doc_emb = model.encode(document, instruction=doc_instruction)
sim = query_emb @ doc_emb.T
```

It is important to also include `trust_remote_code=True` and `torch_dtype="auto"` as discussed earlier. 

NOTE: there seems to be some very slight floating point discrepancy when using the SentenceTransformer (because it does not support bf16 precision), though it should not affect the results in general.

## Evaluations
Please refer to the instructions in `evaluation`.

## Synthetic Data Generation
Please refer to the instructions in `synthetic_data_generation`.

## Test Time Scaling Techniques
Please refer to the instructions in `test_time_techniques`.

## Retriever Training
Please refer to the instructions in `training`.

## Citation
```
@article{shao2025reasonir,
      title={ReasonIR: Training Retrievers for Reasoning Tasks}, 
      author={Rulin Shao and Rui Qiao and Varsha Kishore and Niklas Muennighoff and Xi Victoria Lin and Daniela Rus and Bryan Kian Hsiang Low and Sewon Min and Wen-tau Yih and Pang Wei Koh and Luke Zettlemoyer},
      year={2025},
      journal={arXiv preprint arXiv:2504.20595},
      url={https://arxiv.org/abs/2504.20595}, 
}
```

## Acknowledgments
We thank the following great open-source repositories:
- [BRIGHT](https://github.com/xlang-ai/BRIGHT)
- [GritLM](https://github.com/ContextualAI/gritlm)
- [MassiveDS](https://github.com/RulinShao/retrieval-scaling)