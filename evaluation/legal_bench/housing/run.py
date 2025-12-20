# Embedding datastore creation below
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoModel
from sklearn.metrics import ndcg_score


def _mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Average non-masked token embeddings to get a sentence embedding."""

    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    )
    sum_embeddings = (last_hidden_state * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def encode_passages(
    model,
    texts: list[str],
    batch_size: int = 4,
) -> np.ndarray:
    embeddings = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding passages"):
        batch_texts = texts[start : start + batch_size]

        with torch.no_grad():
            outputs = model.encode(batch_texts, instruction="")

        embeddings.append(outputs)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def _ensure_cache_dir(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)


def _cache_path(cache_dir: str, model_name: str, suffix: str) -> str:
    safe_model = model_name.replace("/", "_")
    cache_subdir = os.path.join(cache_dir, suffix)
    os.makedirs(cache_subdir, exist_ok=True)
    return os.path.join(cache_subdir, f"{safe_model}.npy")


def load_or_encode(
    model,
    texts: list[str],
    batch_size: int,
    cache_dir: str,
    model_name: str,
    suffix: str,
) -> np.ndarray:
    cache_file = _cache_path(cache_dir, model_name, suffix)
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        return np.load(cache_file)

    embeddings = encode_passages(model, texts, batch_size=batch_size)
    np.save(cache_file, embeddings)
    print(f"Saved embeddings cache to {cache_file}")
    return embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create FAISS datastore for passages")
    parser.add_argument(
        "--model-name",
        type=str,
        default="reasonir/ReasonIR-8B",
        help="Hugging Face model id to use for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of passages to encode per batch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = os.path.join(".", "cached embeddings")
    _ensure_cache_dir(cache_dir)

    passages_local_path = hf_hub_download(
        repo_id="reglab/housing_qa",
        filename="data/statutes.tsv",
        repo_type="dataset",
        local_dir=".",
    )

    qa_local_path = hf_hub_download(
        repo_id="reglab/housing_qa",
        filename="data/questions.json.zip",
        repo_type="dataset",
        local_dir=".",
    )

    qa_df = pd.read_json(qa_local_path)[["question", "statutes"]]
    rows = []
    for i, row in qa_df.iterrows():
        rows.append(
            {"question": row.question, "gold_idx": row.statutes[0]["statute_idx"]}
        )
    qa_df = pd.DataFrame(rows)
    passages_df = pd.read_csv(passages_local_path, sep="\t")[["idx", "text"]]

    # Check all the gold passages are in the passages dataset
    merged_df = qa_df.merge(
        passages_df, left_on="gold_idx", right_on="idx", how="inner"
    )
    assert merged_df.shape[0] == qa_df.shape[0], (
        "Some gold passages are missing in the passages dataset"
    )

    del merged_df

    print(f"Number of QA pairs: {qa_df.shape[0]}")
    print(f"Number of Passages: {passages_df.shape[0]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        args.model_name, torch_dtype="auto", trust_remote_code=True
    ).to(device)
    model.eval()

    text_embeddings = load_or_encode(
        model,
        passages_df["text"].tolist(),
        batch_size=args.batch_size,
        cache_dir=cache_dir,
        model_name=args.model_name,
        suffix="passages",
    )
    query_embeddings = load_or_encode(
        model,
        qa_df["question"].tolist(),
        batch_size=args.batch_size,
        cache_dir=cache_dir,
        model_name=args.model_name,
        suffix="queries",
    )

    scores = np.matmul(query_embeddings, text_embeddings.T)

    gold_indices = [
        passages_df[passages_df["idx"] == gid].index[0] for gid in qa_df["gold_idx"]
    ]
    gold_relevance = np.zeros_like(scores)
    for i, gid in enumerate(gold_indices):
        gold_relevance[i, gid] = 1

    ndcg = ndcg_score(gold_relevance, scores, k=10)
    print(f"NDCG@10 of the encoded passages: {ndcg:.4f}")

    outputs_dir = os.path.join(".", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    safe_model = args.model_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"evaluation_results_{safe_model}_{timestamp}.json"
    results_path = os.path.join(outputs_dir, results_filename)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model_name, "ndcg@10": float(ndcg)}, f, indent=2)
    print(f"Saved evaluation results to {results_path}")


if __name__ == "__main__":
    main()
