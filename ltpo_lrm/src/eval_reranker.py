import json
import pytrec_eval
from collections import defaultdict
import os
from utils import RerankerModel, RerankerGPT
from time import time
from tqdm import tqdm

def calculate_retrieval_metrics(results, qrels, k_values=[1, 10, 25, 50, 100]):
    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"mrr": 0}

    for k in k_values:
        ndcg[f"ndcg_at_{k}"] = 0.0
        _map[f"map_at_{k}"] = 0.0
        recall[f"recall_at_{k}"] = 0.0
        precision[f"precision_at_{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)
    print(f"scores length is {len(scores)}")
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"ndcg_at_{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"map_at_{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"recall_at_{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"precision_at_{k}"] += scores[query_id]["P_" + str(k)]
        mrr["mrr"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"ndcg_at_{k}"] = round(ndcg[f"ndcg_at_{k}"] / len(scores), 5)
        _map[f"map_at_{k}"] = round(_map[f"map_at_{k}"] / len(scores), 5)
        recall[f"recall_at_{k}"] = round(recall[f"recall_at_{k}"] / len(scores), 5)
        precision[f"precision_at_{k}"] = round(precision[f"precision_at_{k}"] / len(scores), 5)
    mrr["mrr"] = round(mrr["mrr"] / len(scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr}
    print(output)
    return output

def load_retrieval_data(hf_hub_name="", topk_doc_path=""):
    corpus = defaultdict(dict)
    with open(hf_hub_name + "/corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            pid = e['id']
            corpus[pid] = {"text": e['text']}
    queries = {}
    query_path = hf_hub_name + f"/query.jsonl"
    with open(query_path, "r", encoding="utf-8") as f:
        print(f"Current query file path is {query_path}")
        for line in f:
            e = json.loads(line)
            qid = e['id']
            queries[qid] = e['text']
    relevant_docs = defaultdict(dict)
    with open(hf_hub_name + "/qrels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            pid = e['p_id']
            qid = e['q_id']
            relevant_docs[qid][pid] = int(e["score"])
    qrels_num = 0
    for k, v in relevant_docs.items():
        qrels_num += len(v)
    print(f"共{len(queries)}个queries,{len(corpus)}个文章！,{qrels_num}个相关数据！")
    init_results = defaultdict(dict)
    with open(topk_doc_path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            id = e["id"]
            for pid, s in e["topk_pid"]:
                init_results[id][pid] = s
    return corpus, queries, relevant_docs, init_results

def reranker_by_dense(
        queries=None,
        corpus=None,
        relevant_docs=None,
        init_results=None,
        model=None,
        top_k=None,
):
    print(f">> Before Reranker score is: ")
    init_score = calculate_retrieval_metrics(init_results, relevant_docs)
    new_results = dict()
    if type(model).__name__ == "RerankerGPT":
        for qid in tqdm(list(init_results.keys())):
            topk_pids = list(init_results[qid].keys())
            q_text = [qid, queries[qid]]
            p_texts = [[pid, corpus[pid]["text"]] for pid in topk_pids[:top_k]]
            ranking_pids = model.predict(q_text, p_texts)
            ranking = {}
            for rank_id, r in enumerate(ranking_pids[:top_k]):
                ranking[r] = top_k - rank_id
            new_results[qid] = ranking
    else:
        for qid in tqdm(list(init_results.keys())):
            topk_pids = list(init_results[qid].keys())
            q_text = queries[qid]
            sentence_pairs = [[q_text, corpus[pid]["text"]] for pid in topk_pids[:top_k]]
            scores = model.predict(sentence_pairs)
            ranking = {pid: score for pid, score in zip(topk_pids[:top_k], scores)}
            ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:top_k])
            new_results[qid] = ranking
    print(f">> After Reranker score is: ")
    ranking_score = calculate_retrieval_metrics(new_results, relevant_docs)
    return ranking_score


def eval_reranker(retrieval_name="", reranker_name="", recall_k=10, task_names=None):
    model_path_dict = {
        "bge-reranker": "model_dir/BAAI/bge-reranker-v2-m3",
        "monobert": "model_dir/castorini/monobert-large-msmarco",
        "rankllama": ["model_dir/castorini/rankllama-v1-7b-lora-passage", "model_dir/meta-llama/Llama-2-7b-hf"],
    }
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    reranker_model_path = model_path_dict[reranker_name]
    reranking_model = RerankerModel(model_name_or_path=reranker_model_path,
                                    max_length=1024 if reranker_name == "llama" else 512,
                                    mode=reranker_name)
    print(f"Reranker model have been loaded from {reranker_model_path}")
    t00 = time()
    top_k = recall_k
    ndcg_values = []
    for task_name in task_names:
        t0 = time()
        save_path = f"../results/reranker/{reranker_name}/top_{top_k}/{task_name}.json"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        data_path = f'../dataset/{task_name}/'
        topk_doc_path = f"../output/topk_docs/{retrieval_name}/{task_name}.jsonl"
        corpus, queries, relevant_docs, init_results = load_retrieval_data(data_path, topk_doc_path)
        scores = reranker_by_dense(queries=queries, corpus=corpus, relevant_docs=relevant_docs, init_results=init_results, model=reranking_model, top_k=top_k)
        evaluation_time = round((time() - t0) / 60, 2)
        task_results = {
            "dataset_name": task_name,
            "model_name": reranker_name,
            "topk": top_k,
            "evaluation_time": str(evaluation_time) + " minutes",
            "test": scores,
        }
        with open(save_path, "w") as f_out:
            json.dump(task_results, f_out, indent=2, sort_keys=True)
        ndcg_values.append(scores["ndcg_at_10"])
        print(f"{task_name} evaluation cost {evaluation_time} minutes!")
    print(f"Model {reranker_name} evaluation all task cost {round((time() - t00) / 60, 2)} minutes!")

    ndcg_scaled = [round(value * 100, 2) for value in ndcg_values]
    print(f"Reranker {reranker_name} ranking based on retrieval {retrieval_name} in top-{top_k} settings score is:")
    print("\t".join(map(str, ndcg_scaled)))