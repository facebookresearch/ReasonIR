import os
import random
import re
import json
import argparse
from tqdm import tqdm
import torch.distributed as dist
from lm_helper import HFLM


sample_generation_prompt = """
You have been assigned a passage generation task: 

You will be provided an incomplete data with the below information
- "input": a string, a random input specified by one task.
- "positive_document": a string, a relevant document for the "input" according to the task.

Your task is to generate a "hard_negative_document" in a JSON format:
- The "hard_negative_document" contains some relevant information with superficial lexical overlapping, but it should be not helpful to address the question in the input and is less relevant to the input compared with the "positive_document".

Please adhere to the following guidelines:
- The values of "hard_negative_document" should be in {language}.
- The "hard_negative_document" should be long documents (at least 300 words), avoid substantial word overlaps, otherwise the task would be too easy.
- The "input", "positive_document", and "hard_negative_document" should be independent of each other.

Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!

Now process the below data following the above instruction: 
{input_str}

Your response:
"""


language_options = ['English']


def sample_negative_passage_prompt(input_str):
    sample_prompt = sample_generation_prompt.format(
        language=random.choice(language_options),
        input_str=input_str,
    )
    return sample_prompt


class NegativePassageGenerator:
    def __init__(self, input_file, output_file, num_workers=None, worker_id=None) -> None:
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.model = HFLM()
        self.data = self.load_file(input_file)
        self.output_file = output_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.hashed_queries = set()
        if os.path.exists(output_file):
            with open(output_file, 'r') as fin:
                for line in fin:
                    query = json.loads(line)['query'][-1]
                    if query not in self.hashed_queries:
                        self.hashed_queries.add(query)
    
    def load_file(self, input_file):
        data = []
        with open(input_file, 'r') as fin:
            for line in fin:
                data.append(json.loads(line))
        
        if self.num_workers is not None and self.worker_id is not None:
            shard_size = len(data) // self.num_workers
            remainder = len(data) % self.num_workers
            start_idx = self.worker_id * shard_size + min(self.worker_id, remainder)
            end_idx = start_idx + shard_size + (1 if self.worker_id < remainder else 0)
            return data[start_idx:end_idx]
        return data
    
    def prepare_input_string(self, orig_ex):
        query_str = orig_ex['query'][0]+orig_ex['query'][1]
        pos_str = orig_ex['pos'][0][0]+orig_ex['pos'][0][1]
        input_str = ""
        input_str += f"'input': {query_str}"
        input_str += f"'positive_document': {pos_str}"
        return input_str
        
    def generate_sample(self, orig_ex):
        input_str = self.prepare_input_string(orig_ex)
        sample_instruction = sample_negative_passage_prompt(input_str)
        outputs = self.model.generate(sample_instruction, '')
        return outputs
    
    def parse_generated_sample(self, outputs):
        outputs = outputs.replace('```json', '').replace('```', '').strip().replace('\n', '')
        outputs = re.sub(r'(\w)"s', r'\1\'s', outputs)
        outputs = re.sub(r'"\s+"hard_negative_document"', '",    "hard_negative_document"', outputs)
        try:
            sample = json.loads(outputs)
        except:
            outputs = outputs.replace("'", '"').replace('\\"', '"').replace("\\'", "'")
            sample = json.loads(outputs)
        print(sample)
        return sample
    
    def generate(self,):
        new_samples = []
        for orig_ex in tqdm(self.data):
            if orig_ex['query'][-1] in self.hashed_queries:
                print("Skipping one example that has been processed.")
                continue
            max_attempts = 3
            num_attempts = 0
            while num_attempts < max_attempts:
                num_attempts += 1
                try:
                    sample = self.generate_sample(orig_ex)
                    sample = self.parse_generated_sample(sample)
                    assert sample['hard_negative_document'] and isinstance(sample['hard_negative_document'], str)
                    orig_ex['neg'] = [['', sample['hard_negative_document']]]
                    new_samples.append(orig_ex)
                    break
                except Exception as e:
                    print(f"An error occured: {e}")
                    print(sample)
                    if num_attempts > max_attempts:
                        break

            if len(new_samples) % 10 == 0:
                with open(self.output_file, 'a+') as fout:
                    for ex in new_samples:
                        fout.write(json.dumps(ex) + '\n')
                new_samples = []
        
        if len(new_samples) > 0:
            with open(self.output_file, 'a+') as fout:
                for ex in outputs:
                    fout.write(json.dumps(ex) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=None, help="Total number of workers that will be used to generate negative documents.")
    parser.add_argument('--worker_id', type=int, default=None, help="Used to separate the output file to avoid I/O error when writing to the same file at the same time.")
    args = parser.parse_args()
    
    if args.worker_id is None or args.num_workers is None:
        output_file = args.input_file.replace('.jsonl', '_generated_negative.jsonl')
    else:
        output_file = args.input_file.replace('.jsonl', f'_generated_negative_{args.worker_id}_of_{args.num_workers}.jsonl')
    
    generator = NegativePassageGenerator(args.input_file, output_file, args.num_workers, args.worker_id)
    outputs = generator.generate()
    
