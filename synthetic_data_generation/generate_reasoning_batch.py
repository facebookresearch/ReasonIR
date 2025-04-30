import os
import json
import random
import argparse
from tqdm import tqdm
import time

from hard_negative_mining import BM25_Miner
from data_gen_prompts import *
from gen_utils import *
from data_gen_prompts import *
from lm_helper import MultiTurnOpenAILM, OpenAILM, HFLM
from batch_api_helper import BatchAPIHelper
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
    parser.add_argument("--no_batch", action='store_true', help='disable batch generation')
    parser.add_argument("--gather_results", action='store_true', help='gather batch results')
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


    if not args.no_batch or args.gather_results:
        for subject in data.keys():
            subject_filename = f'{output_path}/{subject}_{num_docs}.jsonl'
            subject_data = data[subject]
            final_training_data = []

            batch_helper = BatchAPIHelper(model_id, subject_filename)
            if args.gather_results:
                outputs = None
                while outputs is None:
                    outputs = batch_helper.gather_results()
                    if outputs is None:
                        print("Waiting for the batch job to finish...")
                        time.sleep(60)

                if len(outputs) == len(subject_data):
                    for output, datum in zip(outputs, subject_data):
                        response = output['response']
                        datum['reasoning'] = response
                    write_jsonl(subject_data, subject_filename)
                else:
                    print("Warning: Some outputs are missing.")
                    print("Collecting results using the ids")
                    outputs_dict = {output['id']: output for output in outputs}
                    base_data = batch_helper.batch_load_base()
                    for base_datum in base_data:
                        query_id = base_datum['id']
                        query = base_datum['query']
                        if query_id in outputs_dict:
                            response = outputs_dict[query_id]['response']
                            base_datum['reasoning'] = response
                        else:
                            print(f"Warning: Missing response for query id: {query_id}")
                            try:
                                print("Regenerating reasoning...")
                                reasoning = model.generate(query, system_prompt=system_prompt)
                                base_datum['reasoning'] = reasoning
                            except Exception as e:
                                print(f"Error: {e}")
                                base_datum['reasoning'] = None
                    
                    base_data_dict = {datum['query']: datum for datum in base_data if datum['reasoning'] is not None}
                    for datum in subject_data:
                        query = datum['query']
                        if query in base_data_dict:
                            datum['reasoning'] = base_data_dict[query]['reasoning']
                            final_training_data.append(datum)
                    write_jsonl(final_training_data, subject_filename)
                batch_helper.clean_up()
                continue

            batch_request_messages = []
            batch_request_ids = []
            batch_base_data = []
            
            for i, datum in enumerate(tqdm(subject_data)):
                query = datum['query']
                query_id = str(i)
                messages = model.apply_chat_template(query, system_prompt=system_prompt)
                batch_request_messages.append(messages)
                batch_request_ids.append(query_id)
                batch_base_data.append({'id': query_id, 'query': query, 'prompt': system_prompt})
            batch_helper.batch_save_base(batch_base_data)
            batch_helper.batch_request(batch_request_messages, batch_request_ids)
    else:
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
