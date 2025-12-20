# Embedding datastore creation below
import argparse
import json
import os
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import ndcg_score
from FlagEmbedding import BGEM3FlagModel


def _reasonir_encode(model, texts: list[str], tokenizer) -> np.ndarray:
    return model.encode(texts, instruction="")


def _contriever_encode(model, texts: list[str], tokenizer) -> np.ndarray:
    # Apply tokenizer
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute token embeddings
    outputs = model(**inputs)

    # Mean pooling
    embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
    return embeddings.cpu().numpy()


def _bge_encode(model, texts: list[str], tokenizer) -> np.ndarray:
    embeddings = model.encode(
        texts,
        max_length=8192,
    )["dense_vecs"]
    return embeddings.cpu().numpy()


def _distilbert_encode(model, texts: list[str], tokenizer) -> np.ndarray:
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = model(**inputs)[0][:, 0, :].squeeze(0)
    return embeddings.cpu().numpy()


def _retromae_encode(model, texts: list[str], tokenizer) -> np.ndarray:
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute token embeddings
    outputs = model(**inputs)

    # Mean pooling
    embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
    return embeddings.cpu().numpy()


ENCODER_MAP: dict[str, Callable] = {
    "reasonir/ReasonIR-8B": _reasonir_encode,
    "facebook/contriever-msmarco": _contriever_encode,
    "BAAI/bge-m3": _bge_encode,
    "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco": _distilbert_encode,
    "Shitao/RetroMAE_MSMARCO_finetune": _retromae_encode,
}


def _cosine_similarity(queries: np.ndarray, passages: np.ndarray) -> np.ndarray:
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
    p_norm = passages / (np.linalg.norm(passages, axis=1, keepdims=True) + 1e-12)
    return np.matmul(q_norm, p_norm.T)


def _dot_product(queries: np.ndarray, passages: np.ndarray) -> np.ndarray:
    return np.matmul(queries, passages.T)


SCORER_MAP: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "reasonir/ReasonIR-8B": _cosine_similarity,
    "facebook/contriever-msmarco": _dot_product,
    "BAAI/bge-m3": _dot_product,
    "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco": _dot_product,
    "Shitao/RetroMAE_MSMARCO_finetune": _cosine_similarity,
}


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def _select_encoder(model_name: str) -> Callable:
    if model_name in ENCODER_MAP:
        return ENCODER_MAP[model_name]
    raise ValueError(f"Unsupported model for encoding: {model_name}")


def _select_scorer(model_name: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if model_name in SCORER_MAP:
        return SCORER_MAP[model_name]
    raise ValueError(f"Unsupported model for scoring: {model_name}")


def _recall_at_k(scores: np.ndarray, gold_indices: list[int], k: int = 10) -> float:
    if scores.shape[1] < k:
        k = scores.shape[1]
    top_k_indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    gold_array = np.array(gold_indices)
    hits = (top_k_indices == gold_array[:, None]).any(axis=1)
    return float(hits.mean())


def _get_model(model_name: str):
    if model_name.startswith("BGE"):
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    else:
        model = AutoModel.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True
        )
    return model


def encode_passages(
    model,
    tokenizer,
    encoder: Callable,
    texts: list[str],
    batch_size: int = 4,
) -> np.ndarray:
    embeddings = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding passages"):
        batch_texts = texts[start : start + batch_size]

        with torch.no_grad():
            outputs = encoder(model, batch_texts, tokenizer)

        n_texts = len(batch_texts)
        embeddings.append(outputs.reshape(n_texts, -1))
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
    tokenizer,
    texts: list[str],
    batch_size: int,
    cache_dir: str,
    model_name: str,
    suffix: str,
    encoder: Callable,
) -> np.ndarray:
    cache_file = _cache_path(cache_dir, model_name, suffix)
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        return np.load(cache_file)

    embeddings = encode_passages(
        model, tokenizer, encoder, texts, batch_size=batch_size
    )
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
        repo_id="reglab/barexam_qa",
        filename="data/passages/test.tsv",
        repo_type="dataset",
        local_dir=".",
    )

    qa_local_path = hf_hub_download(
        repo_id="reglab/barexam_qa",
        filename="data/qa/test.csv",
        repo_type="dataset",
        local_dir=".",
    )

    passages_df = pd.read_csv(passages_local_path, sep="\t")[["idx", "text"]]
    qa_df = pd.read_csv(qa_local_path)[["prompt", "question", "gold_idx"]]
    qa_df["prompt"] = qa_df["prompt"].fillna("")

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
    model = _get_model(args.model_name)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True
    )
    encoder = _select_encoder(args.model_name)
    scorer = _select_scorer(args.model_name)

    texts = passages_df["text"].tolist()
    text_embeddings = load_or_encode(
        model,
        tokenizer,
        texts,
        batch_size=args.batch_size,
        cache_dir=cache_dir,
        model_name=args.model_name,
        suffix="passages",
        encoder=encoder,
    )
    query_embeddings = load_or_encode(
        model,
        tokenizer,
        (qa_df["prompt"] + " " + qa_df["question"]).tolist(),
        batch_size=args.batch_size,
        cache_dir=cache_dir,
        model_name=args.model_name,
        suffix="queries",
        encoder=encoder,
    )

    scores = scorer(query_embeddings, text_embeddings)

    gold_indices = [
        passages_df[passages_df["idx"] == gid].index[0] for gid in qa_df["gold_idx"]
    ]
    gold_relevance = np.zeros_like(scores)
    for i, gid in enumerate(gold_indices):
        gold_relevance[i, gid] = 1

    ndcg = ndcg_score(gold_relevance, scores, k=10)
    recall = _recall_at_k(scores, gold_indices, k=10)
    print(f"NDCG@10 of the encoded passages: {ndcg:.4f}")
    print(f"Recall@10 of the encoded passages: {recall:.4f}")

    outputs_dir = os.path.join(".", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    safe_model = args.model_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"{safe_model}_{timestamp}.json"
    results_path = os.path.join(outputs_dir, results_filename)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": args.model_name, "ndcg@10": float(ndcg), "recall@10": recall},
            f,
            indent=2,
        )
    print(f"Saved evaluation results to {results_path}")


if __name__ == "__main__":
    main()
