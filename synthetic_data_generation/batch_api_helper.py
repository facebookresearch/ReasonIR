import os, sys
import json
from openai import OpenAI


def write_jsonl(data, path):
    with open(path, 'w') as fout:
        for ex in data:
            fout.write(json.dumps(ex)+'\n')


def format_request_json(messages, model="gpt-4o", custom_id="request-1"):
    return {"custom_id": custom_id, 
        "method": "POST", 
        "url": "/v1/chat/completions", 
        "body": {
            "model": model, 
            "messages": messages}}


def process_batch_response(response):
    outputs = []
    json_data = response.content.decode('utf-8')
    for line in json_data.splitlines():
        # Parse the JSON record (line) to validate it
        json_record = json.loads(line)

        custom_id = json_record.get("custom_id")

        # Navigate to the 'choices' key within the 'response' -> 'body'
        choices = json_record.get("response", {}).get("body", {}).get("choices", [])

        # Loop through the choices to find messages with the 'assistant' role
        for choice in choices:
            message = choice.get("message", {})
            if message.get("role") == "assistant":
                assistant_content = message.get("content")
                outputs.append({"id": custom_id, "response": assistant_content})
            break
    return outputs


class BatchAPIHelper(object):
    '''Helper class for calling OpenAI API for batch completions
    Usage:
        batch_helper = BatchAPIHelper(model_id, task_filename)
        batch_request_data = []
        batch_request_ids = []
        batch_helper.batch_request(batch_request_data, batch_request_ids)
        # gather the results after the batch job is completed. It will return None if the job is not completed yet. 
        outputs = batch_helper.gather_results()
        batch_helper.clean_up()
    '''
    def __init__(self, model_id, task_filename, batch_id=None):
        self.model_id = model_id
        self.client = OpenAI()

        self.task_filename = task_filename
        self.batch_request_filename = task_filename.replace('.jsonl', '_batch.jsonl')
        self.batch_id_filename = self.batch_request_filename.replace('.jsonl', '_batch_id.txt')
        self.batch_base_filename = self.batch_request_filename.replace('.jsonl', '_base.jsonl')

        self.batch_id = batch_id

    def gather_results(self):
        '''Gather the results from the batch job
        Return the outputs if the batch job is completed, otherwise return None
        '''
        if self.batch_id is not None:
            batch_id = self.batch_id
        else:
            if not os.path.exists(self.batch_id_filename):
                print('Batch id file does not exist')
                raise FileNotFoundError
            if os.path.exists(self.task_filename):
                print('Warning: Results are already collected as the task file already exists')
            with open(self.batch_id_filename, 'r') as f:
                batch_id = f.read().strip()
        response = self.client.batches.retrieve(batch_id)
        if response.status == 'completed':
            response = self.client.files.content(response.output_file_id)
            outputs = process_batch_response(response)
            return outputs
        else:
            print('Batch job is not completed yet')
            print('Response status:', response.status)
            return None

    def batch_request(self, batch_request_data, batch_request_ids=None):
        '''the batch request data should be a list of messages
        
        Example:
        batch_request_data = [
            [{"role": "system", "content": "Hello, how are you?"},
            {"role": "user", "content": "I am doing well, thank you."}]
        ]

        Provide the batch_request_ids if you want to keep track of the request ids
        '''
        if batch_request_ids is not None:
            assert len(batch_request_data) == len(batch_request_ids)
            batch_request_data = [format_request_json(messages, model=self.model_id, custom_id=custom_id) for messages, custom_id in zip(batch_request_data, batch_request_ids)]
        else:
            batch_request_data = [format_request_json(messages, model=self.model_id) for messages in batch_request_data]
        batch_request_filename = self.batch_request_filename
        write_jsonl(batch_request_data, batch_request_filename)
        response = self.client.files.create(
            file=open(batch_request_filename, "rb"),
            purpose="batch"
        )
        file_id = response.id
        response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        batch_id = response.id
        with open(self.batch_id_filename, 'w') as f:
            f.write(batch_id)

    def batch_save_base(self, base_data):
        '''Save the base data for the batch job
        The base data contains information that will be combined with the batch results
        '''
        write_jsonl(base_data, self.batch_base_filename)

    def batch_load_base(self):
        '''Load the base data for the batch job
        The base data contains information that will be combined with the batch results
        '''
        with open(self.batch_base_filename, 'r') as f:
            base_data = [json.loads(line) for line in f]
        return base_data

    def clean_up(self):
        if os.path.exists(self.batch_request_filename):
            os.remove(self.batch_request_filename)
        if os.path.exists(self.batch_id_filename):
            os.remove(self.batch_id_filename)
        if os.path.exists(self.batch_base_filename):
            os.remove(self.batch_base_filename)