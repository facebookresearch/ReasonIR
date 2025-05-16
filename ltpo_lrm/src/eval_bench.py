import os
from time import time
from collections import defaultdict
import json
import sys
from mteb.abstasks.AbsTaskRetrieval import DRESModel
DRES_METHODS = ["encode_queries", "encode_corpus"]
import re
from utils import FlagDRESModel, InstructorModel, BiEncoderModel, HighScaleModel, GritModel, NVEmbedModel, RetrievalOPENAI
from instrcution import *
from tqdm import tqdm
from utils import writejson_bench
from base import split_response_for_r1, r1_model_list


def is_dres_compatible(model):
    for method in DRES_METHODS:
        op = getattr(model, method, None)
        if not (callable(op)):
            return False
    return True


def load_retrieval_data_beir(hf_hub_name=""):
    corpus = defaultdict(dict)
    with open(hf_hub_name + "/corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            pid = e['_id']
            corpus[pid] = {"title": e["title"], "text": e['text']}
    test_qids = []
    relevant_docs = defaultdict(dict)
    with open(hf_hub_name + "/qrels/test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            pid = e['corpus-id']
            qid = e['query-id']
            relevant_docs[qid][pid] = int(e["score"])
            test_qids.append(qid)
    queries = {}
    query_path = hf_hub_name + "/queries.jsonl"
    with open(query_path, "r", encoding="utf-8") as f:
        print(f"Current query file path is {query_path}")
        for line in f:
            e = json.loads(line)
            qid = e['_id']
            if qid not in test_qids:
                continue
            queries[qid] = e['text']
    qrels_num = 0
    for k, v in relevant_docs.items():
        qrels_num += len(v)

    print(f"共{len(queries)}个queries,{len(corpus)}个文章！,{qrels_num}个相关数据！")
    return corpus, queries, relevant_docs

def load_retrieval_data(hf_hub_name="", gar_method="", gar_llm="", r1_mode=False):
    corpus = defaultdict(dict)
    with open(hf_hub_name + "/corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            pid = e['id']
            corpus[pid] = {"text": e['text']}

    queries = {}
    output_format_error = 0
    if gar_method != "" and gar_llm != "":
        query_path = hf_hub_name + f"/{gar_method}/{gar_llm}/query_with_hydoc.jsonl"
    else:
        query_path = hf_hub_name + f"/query.jsonl"
    with open(query_path, "r", encoding="utf-8") as f:
        print(f"Current query file path is {query_path}")
        for line in f:
            e = json.loads(line)
            qid = e['id']
            if gar_method != "" and gar_llm != "":
                if r1_mode:
                    queries[qid] = [e['text'], split_response_for_r1(gar_llm, e["hy_doc"])]
                else:
                    if gar_method == "query2doc":
                        queries[qid] = e['text'] + "[SEP]" + e["hy_doc"]
                    else:
                        queries[qid] = [e['text'], e["hy_doc"]]
            else:
                queries[qid] = e['text']

    print(f"一共{output_format_error}/{len(queries)}格式有误！")
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
    return corpus, queries, relevant_docs

def save_topk_docs(results, relevant_docs, save_path, k):
    # candidates = dict()
    # for q_id in tqdm(results.keys()):
    #     can = results[q_id]
    #     sorted_can = dict(sorted(can.items(), key=lambda item: item[1], reverse=True))
    #     p_ids = [[d,s] for d, s in sorted_can.items()][:k]
    #     candidates[q_id] = p_ids
    # new_data = []
    # for id in candidates.keys():
    #     data = candidates[id]
    #     new_data.append(
    #         {
    #             "id": id,
    #             "topk_pid": data,
    #         }
    #     )
    # writejson_bench(new_data, save_path)
    candidates = defaultdict(list)
    for q_id in tqdm(results.keys()):
        can = results[q_id]
        target_ids = list(relevant_docs[q_id].keys())
        sorted_can = dict(sorted(can.items(), key=lambda item: item[1], reverse=True))
        p_ids = [d for d in sorted_can.keys()]
        for target_id in target_ids:
            if target_id in p_ids:
                index = p_ids.index(target_id)
            else:
                index = 1000
            candidates[q_id].append([target_id, index])
    new_data = []
    for id in candidates.keys():
        data = candidates[id]
        item = {
            "id": id,
            "label_location": data,
        }
        new_data.append(item)
    writejson_bench(new_data, save_path)


def retrieval_by_dense(
        queries=None,
        corpus=None,
        relevant_docs=None,
        model=None,
        batch_size=512,
        corpus_chunk_size=None,
        score_function="cos_sim",
        **kwargs
):
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError:
        raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")

    corpus, queries = corpus, queries
    model = model if is_dres_compatible(model) else DRESModel(model)
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    model = DRES(
        model,
        batch_size=batch_size,
        corpus_chunk_size=corpus_chunk_size if corpus_chunk_size is not None else 100000,
        **kwargs,
    )
    retriever = EvaluateRetrieval(model, k_values=[1, 3, 5, 10, 100, 1000],
                                  score_function=score_function)  # or "cos_sim" or "dot"

    start_time = time()
    results = retriever.retrieve(corpus, queries)
    end_time = time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
    ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values,
                                                       ignore_identical_ids=kwargs.get("ignore_identical_ids", True))
    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }
    print(scores)
    return scores, results

model_path_dict = {
    "contriever": "/share/project/ll/Retrieval-models/facebook/contriever-msmarco",
    "medcpt": ["/share/project/ll/Retrieval-models/ncbi/MedCPT-Query-Encoder", "/share/project/ll/Retrieval-models/ncbi/MedCPT-Article-Encoder"],
    "inst-l": "/share/project/ll/Retrieval-models/hkunlp/instructor-large",
    "inst-xl": "/share/project/ll/Retrieval-models/hkunlp/instructor-xl",
    "bmr-410m": "/share/project/ll/Retrieval-models/BMRetriever/BMRetriever-410M",
    "bmr-2b": "/share/project/ll/Retrieval-models/BMRetriever/BMRetriever-2B",
    "bmr-7b": "/share/project/ll/Retrieval-models/BMRetriever/BMRetriever-7B",
    "bge": "/share/project/ll/Retrieval-models/BAAI/bge-large-en-v1.5",
    "e5": "/share/project/ll/Retrieval-models/intfloat/e5-mistral-7b-instruct",
    "grit": "/share/project/ll/Retrieval-models/GritLM/GritLM-7B",
    "sfr": "/share/project/ll/Retrieval-models/Salesforce/SFR-Embedding-Mistral",
    "nv": "/share/project/ll/Retrieval-models/nvidia/NV-Embed-v2",
    "openai": "text-embedding-3-large",
    "voyage": "voyage-3",
}

gar_method = "hyde"
gar_llm = "qwq-32b"
# gar_method = "search-o1-medcorpus"
# gar_llm = "qwen3-32b"
# gar_method = ""
# gar_llm = ""
r1_mode = False
if gar_llm in r1_model_list:
    r1_mode = True
retriever_names = ["nv"]
for retriever_name in retriever_names:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    retrieral_model_path = model_path_dict[retriever_name]
    if retriever_name in ["contriever"]:
        retrieval_model = FlagDRESModel(model_name_or_path=retrieral_model_path,
                                        pooling_method="mean",
                                        normalize_embeddings=False)
    elif retriever_name in ["inst-l", "inst-xl"]:
        retrieval_model = InstructorModel(model_name_or_path=retrieral_model_path,
                                      query_instruction_for_retrieval="",
                                      document_instruction_for_retrieval="",
                                      batch_size=64 if retriever_name == "inst-l" else 32)
    elif retriever_name in ["e5", "sfr"]:
        retrieval_model = HighScaleModel(model_name_or_path=retrieral_model_path,
                                        query_instruction_for_retrieval="",
                                        pooling_method='last',
                                        max_length=2048,
                                        batch_size=32)
    elif retriever_name in ["bge"]:
        retrieval_model = FlagDRESModel(model_name_or_path=retrieral_model_path,
                                  query_instruction_for_retrieval=query_instruction[retriever_name],
                                  pooling_method='cls',
                                  max_length=512)
    elif retriever_name in ["bmr-410m", "bmr-2b"]:
        MAX_LENGTH = {"bmr-410m":512, "bmr-2b":1024}
        retrieval_model = FlagDRESModel(model_name_or_path=retrieral_model_path,
                                        encode_mode="BMR",
                                        query_instruction_for_retrieval="",
                                        pooling_method="last-bmr",
                                        max_length=MAX_LENGTH[retriever_name],
                                        document_instruction_for_retrieval=doc_instruction[retriever_name],
                                        batch_size=32)
    elif retriever_name in ["bmr-7b"]:
        retrieval_model = HighScaleModel(model_name_or_path=retrieral_model_path,
                                        encode_mode="BMR",
                                        query_instruction_for_retrieval="",
                                        pooling_method="last-bmr",
                                        max_length=2048,
                                        document_instruction_for_retrieval=doc_instruction[retriever_name],
                                        batch_size=64)
    elif retriever_name in ["medcpt"]:
        retrieval_model = BiEncoderModel(query_encoder_name_or_path=retrieral_model_path[1],
                                         doc_encoder_name_or_path=retrieral_model_path[1],
                                        max_length=512,
                                        batch_size=512)
    elif retriever_name in ["grit"]:
        retrieval_model = GritModel(model_name_or_path=retrieral_model_path,
                                      query_instruction_for_retrieval="",
                                     document_instruction_for_retrieval=doc_instruction[retriever_name],
                                        batch_size=64)
    elif retriever_name in ["nv"]:
        retrieval_model = NVEmbedModel(model_name_or_path=retrieral_model_path,
                                    query_instruction_for_retrieval="",
                                    document_instruction_for_retrieval=doc_instruction[retriever_name],
                                    batch_size=16,
                                    max_length=4096)
    elif retriever_name in ["openai", "voyage"]:
        retrieval_model = RetrievalOPENAI(model_name_or_path=retrieral_model_path, batch_size=64)
    print(f"Embedding model have been loaded from {retrieral_model_path}")
    t00 = time()
    # task_names = ["Stack-Biology"] # Stack-Biology BEIR_NFCorpus
    task_names = ["Stack-Biology", "Stack-Bioinformatics", "Stack-Medical", "MedXpertQA-Exam",
                  'MedQA-Diag', "PMC-Treat", "PMCPatients", "IIYiPatients-EN"][:]
    ndcg_values = []
    save_top_k = True # default to False
    for task_name in task_names[:]:
        t0 = time()
        if "BEIR" in task_name:
            # data_path = f'/share/project/ll/Retrieval-datasets/Retrieval/benchmark/{task_name}/'
            data_path = f'../Benchmark/{task_name}/'
        else:
            data_path = f'../Benchmark/{task_name}/'
        if gar_method != "":
            save_path = f"./results/gar/{gar_method}/{gar_llm}/{retriever_name}/{task_name}.json"
        else:
            save_path = f"./results/retrieval/{retriever_name}/{task_name}.json"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        cache_path = f"./doc_embs/{retriever_name}/{task_name}.npy"
        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))
        if os.path.exists(save_path):
            print(f">>>WARNING: Model {retriever_name} in dataset {task_name} results already exists. Skipping.")
            print()
            # continue
        if "BEIR" in task_name:
            corpus, queries, relevant_docs = load_retrieval_data_beir(data_path)
        else:
            corpus, queries, relevant_docs = load_retrieval_data(data_path, gar_method, gar_llm, r1_mode)

        if retriever_name in ["e5", "bmr-410m", "bmr-2b", "bmr-7b","inst-l", "inst-xl", "grit", "sfr", "nv"]:
            retrieval_model.query_instruction_for_retrieval = query_instruction[retriever_name][task_name]
        if retriever_name in ["inst-l", "inst-xl"]:
            retrieval_model.document_instruction_for_retrieval = doc_instruction[retriever_name][task_name]
        if retriever_name in ["openai", "voyage"]:
            doc_cache_path = cache_path.replace(f"{task_name}", f"{task_name}-doc")
            retrieval_model.query_cache_path = cache_path
            retrieval_model.doc_cache_path = doc_cache_path
        else:
            retrieval_model.cache_path = cache_path
        scores, top_k_results = retrieval_by_dense(queries=queries, corpus=corpus, relevant_docs=relevant_docs, model=retrieval_model)
        evaluation_time = round((time() - t0) / 60, 2)
        task_results = {
            "dataset_name": task_name,
            "model_name": retriever_name,
            "evaluation_time": str(evaluation_time) + " minutes",
            "test": scores,
        }
        with open(save_path, "w") as f_out:
            json.dump(task_results, f_out, indent=2, sort_keys=True)
        ndcg_values.append(scores["ndcg_at_10"])
        print(f"{task_name} evaluation cost {evaluation_time} minutes!")
        if save_top_k:
            save_top_k_path = f"./topk_docs/{gar_method}-{gar_llm}/{retriever_name}/{task_name}_pos_label_location.jsonl"
            if not os.path.exists(os.path.dirname(save_top_k_path)):
                os.makedirs(os.path.dirname(save_top_k_path))
            save_topk_docs(top_k_results, relevant_docs, save_top_k_path, 100)

    print(f"Model {retriever_name} evaluation all task  cost {round((time() - t00) / 60, 2)} minutes!")

    ndcg_scaled = [round(value * 100, 2) for value in ndcg_values]
    print("\t".join(map(str, ndcg_scaled)))

