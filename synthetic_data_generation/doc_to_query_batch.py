import os
import json
import random
import argparse
from tqdm import tqdm
import pdb
from collections import defaultdict
import numpy as np

from hard_negative_mining import BM25_Miner
from data_gen_prompts import *
from gen_utils import *
from lm_helper import OpenAILM, HFLM
from batch_api_helper import BatchAPIHelper
import re
import time


def base_data_to_dict(base_data):
    base_dict = {}
    for item in base_data:
        doc_id = item['doc_id']
        base_dict[doc_id] = item
    return base_dict


def format_example(bm25_miner, query, document, queries_per_doc=1):
    decoder = json.JSONDecoder()
    try:
        json_data, _ = decoder.raw_decode(query[query.find('{'):])
        if 'hard_query' in json_data:
            queries = json_data['hard_query']
        else:
            queries = json_data['questions']
        if len(queries) > queries_per_doc:
            queries = queries[:queries_per_doc]
    except Exception as e:
        print(e)
        print(f"Skipping query: {query}")
        return None
    items = []
    for _query in queries:
        pos = document
        try:
            _question = _query
            negs = bm25_miner.select_hard_negatives(_question, pos, 1)
        except Exception as e:
            print(e)
            print(f"Skipping query with type {type(_query)}: {_query}")
            continue
        item = {
            'query': _question,
            'pos': [pos],
            'neg': negs,
        }
        items.append(item)
    return items

def doc2query(bm25_miner, model_id="meta-llama/Meta-Llama-3.1-70B-Instruct", num_docs=100, queries_per_doc=1, filter_name=None, 
        output_dir='synthetic_data', cache_dir='cache', prompt_id='hq_gen', diverse_prompt=False, num_prompts=1, temperature=0, top_p=0, batch_helper=None):

    prompt = prompt_registry[prompt_id]
    dataset = bm25_miner.dataset
    subject = bm25_miner.task if bm25_miner.task else dataset
    
    if bm25_miner.dataset == 'BRIGHT' and bm25_miner.task:
        examples = load_dataset('xlangai/BRIGHT', 'examples', cache_dir='cache')[subject]
    else:
        print("No task specified, using all examples.")
        tasks = ['biology', 'earth_science', 'economics', 'psychology', 'robotics', 
            'stackoverflow', 'sustainable_living', 'leetcode', 'pony', 
            'aops', 'theoremqa_theorems', 'theoremqa_questions']
        all_examples = load_dataset('xlangai/BRIGHT', 'examples', cache_dir='cache')
        examples = []
        for task in tasks:
            examples.extend(all_examples[task])
    
    documents, doc_ids = bm25_miner.documents, bm25_miner.doc_ids
    doc_dicts = [{'doc_id': doc_id, 'doc': doc} for doc_id, doc in zip(doc_ids, documents)]
    
    total_num_docs = len(doc_dicts)
    num_docs_sample_pool = min(num_docs*1, total_num_docs)  # document pool to sample num_oversample_docs docs
    num_oversample_docs = min(num_docs*2, total_num_docs)  # obtain more documents than expected to make the final output matches the expectation
    filter_cache_dir = f'cache/{subject}'
    os.makedirs(filter_cache_dir, exist_ok=True)
    print(f"Filtering documents based on {filter_name}...")

    if dataset == 'msmarco':
        doc_dicts, doc_ids = document_filter(doc_dicts, doc_ids, filter_name=filter_name, num_docs=num_docs, cache_dir=filter_cache_dir)
    else:
        doc_dicts, doc_ids = document_filter(doc_dicts, doc_ids, filter_name=filter_name, num_docs=num_docs, cache_dir=filter_cache_dir)
            
    num_filtered_docs = len(doc_dicts)
    print(f"Total number of documents: {total_num_docs}, number of filtered documents with oversampling: {num_filtered_docs}")

    model_id_str = model_id.split('/')[-1]
    # path to save intermediate model generated results, will check if the same document has been used before to avoid repetitive generation.
    final_output_path = os.path.join(output_dir, f'all_docs_train_data/{prompt_id}/{model_id_str}/{subject}_{num_docs}_train_data.jsonl')
    final_output_path = os.path.expanduser(final_output_path)
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

    if 'gpt' in model_id:
        model = OpenAILM(model_id, temperature=temperature, top_p=top_p, seed=0)
    else:
        model = HFLM(model_id, temperature=temperature, top_p=top_p)
    
    system_prompt = fill_sys_prompt(prompt, queries_per_doc=queries_per_doc)
    system_prompts = [system_prompt]
    
    batch_request_messages = []
    batch_request_ids = []
    batch_base_data = []
    
    final_training_data = []
    print("Sampling from system prompts:", system_prompts)
    for doc_dict in tqdm(doc_dicts):
        document = doc_dict['doc']
        formatted_query = format_query_doc(document)

        if len(system_prompts) > num_prompts:
            sampled_prompts = random.choices(system_prompts, k=num_prompts)
        else:
            sampled_prompts = system_prompts

        for system_prompt in sampled_prompts: # generate examples for each prompt
            max_num_attempt_per_doc = 3
            num_attempt = 0
            has_generated = False
            while not has_generated and num_attempt < max_num_attempt_per_doc:
                if batch_helper is not None:
                    doc_id = doc_dict['doc_id']
                    messages = model.apply_chat_template(formatted_query, system_prompt=system_prompt)
                    batch_request_messages.append(messages)
                    batch_request_ids.append(doc_id)
                    batch_base_data.append({'doc_id': doc_id, 'document': document, 'prompt': system_prompt})
                    items = None
                    has_generated = True
                else:
                    query = model.generate(formatted_query, system_prompt=system_prompt)
                    print(f"Generated query: {query}")
                    items = {'query': query, 'document': document, 'prompt': system_prompt}
                    items = format_example(bm25_miner, query, document, queries_per_doc)
                    num_attempt += 1
                    if items is not None:
                        has_generated = True
            
            if not items:
                print(f"Skipping document: {document}")
                continue
            if isinstance(items, list):
                if len(items) < queries_per_doc:
                    continue
                final_training_data.extend(items[:queries_per_doc])
        if len(final_training_data) >= num_docs * queries_per_doc:
            break
    
    if batch_helper is not None:
        batch_helper.batch_save_base(batch_base_data)
        batch_helper.batch_request(batch_request_messages, batch_request_ids)
    else:
        write_jsonl(final_training_data, final_output_path)


def main():
  pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None, help='mode')
    parser.add_argument('--model_id', type=str, default='gpt-4o', help='model id')
    parser.add_argument('--dataset', type=str, default='bright', help='dataset')
    parser.add_argument('--subject', type=str, default=None, help='subject')
    parser.add_argument('--queries_per_doc', type=int, default=3, help='number of generated samples per document')
    parser.add_argument('--num_docs', type=int, default=None, help='number of documents to sample for each subject')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--filter', type=str, default=None, help='the default filter is the length filter', choices=['length', 'fineweb', 'dclm'])
    parser.add_argument('--data_path', type=str, default='~/data/chunks/mathpile_wiki_chunks.jsonl', help='data path')
    parser.add_argument('--output_dir', type=str, default='data/synthetic_questions', help='base directory to save the generated data')
    parser.add_argument('--cache_dir', type=str, default='cache/', help='cache directory to save cached data during document filtering.')
    parser.add_argument('--prompt_id', type=str, default='hq_gen', help='prompt to use')
    parser.add_argument("--num_prompts", type=int, default=1, help='number of prompts to use for diversity')
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--no_batch", action='store_true', help='disable batch mode')
    parser.add_argument('--gather_results', action='store_true', help='check batch job and process')
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    args.data_path = os.path.expanduser(args.data_path)
    os.makedirs(args.cache_dir, exist_ok=True)
    print(args)


    print(f"Dataset: {args.dataset}, Task: {args.subject}")
    bm25_miner = BM25_Miner(dataset=args.dataset, task=args.subject, data_path=args.data_path)

    model_id = args.model_id
    prompt_id = args.prompt_id
    num_docs = args.num_docs
    queries_per_doc = args.queries_per_doc
    dataset = bm25_miner.dataset
    subject = bm25_miner.task if bm25_miner.task else dataset
    output_dir = args.output_dir

    model_id_str = model_id.replace('/', '_')
    task_filename = f'{model_id_str}_{bm25_miner.dataset}_{bm25_miner.task}.jsonl'
    task_filename = os.path.join(args.output_dir, task_filename)
    batch_helper = None
    if not args.no_batch or args.gather_results:
        batch_helper = BatchAPIHelper(model_id, task_filename)

    if args.gather_results:
        base_data = batch_helper.batch_load_base()
        base_data_dict = base_data_to_dict(base_data)
        outputs = None
        while outputs is None:
            outputs = batch_helper.gather_results()
            if outputs is None:
                print("Waiting for the batch job to finish...")
                time.sleep(60)

        # process the outputs and cache the results
        model_id_str = model_id.split('/')[-1]
        # path to save intermediate model generated results, will check if the same document has been used before to avoid repetitive generation.
        final_output_path = os.path.join(output_dir, f'all_docs_train_data/{prompt_id}/{model_id_str}/{subject}_{num_docs}_train_data.jsonl')
        final_output_path = os.path.expanduser(final_output_path)
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

        final_training_data = []
        for output in outputs:
            query_id = output['id']
            response = output['response']
            document = base_data_dict[query_id]['document']
            items = format_example(bm25_miner, response, document, queries_per_doc)

            if not items:
                print(f"Skipping document: {document}")
                continue
            if isinstance(items, list):
                if len(items) < queries_per_doc:
                    continue
                final_training_data.extend(items[:queries_per_doc])
        write_jsonl(final_training_data, final_output_path)
        if len(final_training_data) > 0:
            print(f"Generated {len(final_training_data)} samples.")
            batch_helper.clean_up()
        exit()

    doc2query(bm25_miner, model_id=model_id, num_docs=args.num_docs, 
        filter_name=args.filter, queries_per_doc=args.queries_per_doc, output_dir=args.output_dir, cache_dir=args.cache_dir,
        prompt_id=args.prompt_id, temperature=args.temperature, top_p=args.top_p, batch_helper=batch_helper)
    
