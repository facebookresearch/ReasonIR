import os
import json
import pdb


def load_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def write_jsonl(data, path):
    with open(path, 'w') as fout:
        for ex in data:
            fout.write(json.dumps(ex)+'\n')


def format_query(input_dir, output_file):
    all_filenames = os.listdir(input_dir)
    all_queries = []
    for filename in all_filenames:
        if not filename.endswith('.json'):
            continue
        data = load_json(os.path.join(input_dir, filename))
        for ex in data:
            query = ex['model_outputs'].split('\n\nThe answer is')[0]
            all_queries.append({
                'query': query,
                'question': ex['question'],
                })
    write_jsonl(all_queries, output_file)
    