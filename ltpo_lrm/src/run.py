import os
import argparse
import json
from eval_retrieval import eval_retrieval
from eval_BM25 import eval_bm25
from eval_reranker import eval_reranker
from generate_hypothetical_doc import generate_hy_doc

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=["eval_retrieval", "eval_reranker", "generate_hydoc"])
    parser.add_argument('--task', type=str, required=True,
                        choices=["Biology", "Bioinformatics", "Medical-Sciences", "MedXpertQA-Exam",
                  'MedQA-Diag', "PMC-Treatment", "PMC-Clinical", "IIYi-Clinical", "All"])
    parser.add_argument('--retriever_name', type=str, required=True,
                        choices=['bm25', 'contriever', 'medcpt', 'inst-l', 'inst-xl', 'bmr-410m', 'bmr-2b',
                                 'bmr-7b', 'bge', 'e5', 'grit', 'sfr', 'voyage', 'openai', 'voyage'])
    parser.add_argument('--reranker_name', type=str, required=True,
                        choices=["bge-reranker", "monobert", "rankllama"])
    parser.add_argument('--recall_k', type=int, default=10, choices=[10, 100])
    parser.add_argument('--gar_method', type=str, default="",
                        choices=['hyde', 'query2doc', 'lamer', 'search-o1', 'search-r1'])
    parser.add_argument('--gar_llm', type=str, default="",
                        choices=['qwen-7b','qwen-32b','qwen-72b','llama-70b','r1-qwen-32b','r1-llama-70b',
                                 'huatuo-o1-70b','qwq-32b','qwen3-32b','gpt4','o3-mini'])
    args = parser.parse_args()
    if args.task == "All":
        args.task = ["Biology", "Bioinformatics", "Medical-Sciences", "MedXpertQA-Exam",
                  'MedQA-Diag', "PMC-Treatment", "PMC-Clinical", "IIYi-Clinical"]
    else:
        args.task = [args.task]
    if args.mode == "eval_retrieval":
        if args.retriever_name == "bm25":
            eval_bm25(gar_method=args.gar_method,
                gar_llm=args.gar_llm,
                task_names=args.task)
        else:
            eval_retrieval(retriever_name=args.retriever_name,
                gar_method=args.gar_method,
                gar_llm=args.gar_llm,
                task_names=args.task)
    elif args.mode == "eval_reranker":
        eval_reranker(retrieval_name=args.retriever_name,
                      reranker_name=args.reranker_name,
                      recall_k=args.recall_k,
                      task_names=args.task)
    elif args.mode == "generate_hydoc":
        generate_hy_doc(gar_method=args.gar_method,
                        gar_llm=args.gar_llm,
                        task_names=args.task)

    else:
        raise ValueError("Invalid mode. Choose from 'eval_retrieval', 'eval_reranker', or 'generate_hydoc'.")