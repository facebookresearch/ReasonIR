import os
import json
import random
import argparse
from tqdm import tqdm

from hard_negative_mining import BM25_Miner
from data_gen_prompts import *
from gen_utils import *
from data_gen_prompts import *
from lm_helper import MultiTurnOpenAILM, OpenAILM, HFLM
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='gpt-4o', help='model id')
    parser.add_argument('--prompt_id', type=str, default=None, help='prompt id')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--base_dir', type=str, default='synthetic_data/top100', help='base directory to save the generated data')
    parser.add_argument('--num_docs', type=int, default=None, help='number of samples to collect')
    parser.add_argument('--dataset', type=str, default='bright', help='dataset to collect the data for')
    parser.add_argument('--subject', type=str, default='biology', help='subject to collect the data for')
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0)
    args = parser.parse_args()
    print(args)

    # load the queries
    model_id = args.model_id
    model_id_str = model_id.split('/')[-1]
    num_docs = args.num_docs
    if args.prompt_id is not None:
        train_data_path = os.path.join(args.base_dir, 'all_docs_train_data', args.prompt_id, model_id_str)
        output_path = f'{args.base_dir}/reasoning_data/{args.prompt_id}/{model_id_str}'
    else:
        train_data_path = os.path.join(args.base_dir, 'all_docs_train_data', model_id_str)
        output_path = f'{args.base_dir}/reasoning_data/{model_id_str}'
    train_data_path = os.path.expanduser(train_data_path)
    from gen_utils import load_training_data
    subject = args.dataset if args.dataset in ['msmarco'] else args.subject
    data = load_training_data(train_data_path, num_docs=num_docs, subject=subject)

    # Initialize the model
    if 'gpt' in model_id:
        model = OpenAILM(model_id, temperature=args.temperature, top_p=args.top_p, seed=0)
    else:
        model = HFLM(model_id, temperature=args.temperature, top_p=args.top_p)
    
    system_prompt = PROMPT_COT_BRIGHT

    print(f'System prompt:\n{system_prompt}')
    os.makedirs(output_path, exist_ok=True)
    for subject in data.keys():
        subject_filename = f'{output_path}/{subject}_{num_docs}.jsonl'
        subject_data = data[subject]
        for datum in tqdm(subject_data):
            query = datum['query']
            reasoning = model.generate(query, system_prompt=system_prompt)
            datum['reasoning'] = reasoning
            if args.debug:
                break
        write_jsonl(subject_data, subject_filename)
