from datasets import load_dataset
import os
import json
import re
import random


def format_query_doc(doc, example_query=None):
    user_prompt = '''The document is given below:
<document>
<DOC_CONTENT_FILL_ME>
</document>\n'''.replace('<DOC_CONTENT_FILL_ME>', doc)
    if example_query is not None:
        user_prompt += f'''
\nThe example question is given below:
<question>
<QUESTION_FILL_ME>
</question>\n'''.replace('<QUESTION_FILL_ME>', example_query)
    user_prompt += '\nPlease start generating the questions.'
    return user_prompt


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


def format_request_json(messages, model="gpt-4o", custom_id="request-1"):
    return {"custom_id": custom_id, 
        "method": "POST", 
        "url": "/v1/chat/completions", 
        "body": {
            "model": model, 
            "messages": messages}}


def extract_json_from_text(text):
    decoder = json.JSONDecoder()
    try:
        json_data, _ = decoder.raw_decode(text[text.find('{'):])
    except json.decoder.JSONDecodeError as e:
        # text = text.replace('\\', '\\\\')
        text = re.sub(r'(?<!\\)\\(?!\\)', r'\\\\', text)
        try:
            json_data, _ = decoder.raw_decode(text[text.find('{'):])
        except Exception as e:
            print(e)
            # breakpoint()
            return None
    return json_data


def load_jsonl(path):
    data = []
    with open(path, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    return data


def write_jsonl(data, path):
    with open(path, 'w') as fout:
        for ex in data:
            fout.write(json.dumps(ex)+'\n')


def load_json(path):
    with open(path, 'r') as fin:
        data = json.load(fin)
    return data


def write_json(data, path):
    with open(path, 'w') as fout:
        json.dump(data, fout)
            

def load_training_data(folder_path, num_docs=None, subject=None):
    all_data = {}
    print(f"Loading data from {folder_path}...")
    for f in os.listdir(folder_path):
        if f.endswith('.jsonl'):
            # if subject is not None and file_subject != subject:
            #     continue
            if subject is not None and subject not in f:
                continue
            if subject is None:
                file_subject = f.split('_')[0]
                if not f.split('_')[1][0].isdigit():
                    file_subject += '_'+f.split('_')[1]
            else:
                file_subject = subject
            if 'exgold' in f:
                continue
            # dataset_n_samples = int(f.split('_')[1])
            dataset_n_samples = int(f.split('.')[0].split(file_subject+'_')[1].split('_')[0])
            if num_docs is not None:
                if dataset_n_samples != num_docs:
                    continue
            data = load_jsonl(os.path.join(folder_path, f))
            if file_subject not in all_data:
                all_data[file_subject] = data
            else:
                # load the data with the most examples
                if len(all_data[file_subject]) < len(data):
                    all_data[file_subject] = data
    # summarize the data
    for file_subject, data in all_data.items():
        print(f"Subject: {file_subject}, number of examples: {len(data)}")
    return all_data


# given a subject, get the examples containing query, reasoning, and document ids
def get_examples(task, cache_dir='cache'):
    return load_dataset('xlangai/BRIGHT', 'examples', cache_dir=cache_dir)[task]


# get the documents for a given subject/task
def get_docs(task, long_context=False, cache_dir='cache'):
    if long_context:
        docs = load_dataset('xlangai/BRIGHT', 'long_documents', cache_dir=cache_dir)[task]
    else:
        docs = load_dataset('xlangai/BRIGHT', 'documents', cache_dir=cache_dir)[task]
    return docs


# filter the documents based on the filter_name
def document_filter(documents, doc_ids, filter_name=None, threshold=1, return_scores=False, num_docs=None, num_sample_pool=None, cache_dir=None, debug=False):
    print(f"Deduplicating {len(documents)} documents...")
    documents, indices = deduplicate_docs(documents)
    print(f"Number of unique documents: {len(documents)}")
    doc_ids = [doc_ids[i] for i in indices]
    if len(documents) > 0 and isinstance(documents[0], dict):
        # check if the documents are in the form of dictionaries
        passages = [documents[i]['doc'] for i in range(len(documents))]
    else:
        passages = [i for i in documents]

    if filter_name is None:
        print("No filter specified. Using the default length filter.")
        filter_name = 'length'
    if filter_name == 'length' or filter_name == 'chunk':
        if filter_name == 'length':
            ids = length_filter(passages)
        elif filter_name == 'chunk':
            ids = chunk_filter(passages)
        filtered_docs = [documents[i] for i in ids]
        filtered_doc_ids = [doc_ids[i] for i in ids]
        if num_docs is not None and num_sample_pool is not None:
            # Sample num_docs documents from a larger pool for diversity 
            # and avoid conjection chunks with similar quality scores being chosen too much (might lead to false negative)
            num_sample_pool = max(num_docs, num_sample_pool)
            filtered_docs = filtered_docs[:num_sample_pool]
            filtered_doc_ids = filtered_doc_ids[:num_sample_pool]
            indices = random.sample(range(num_sample_pool), num_docs)
            filtered_docs = [filtered_docs[i] for i in indices]
            filtered_doc_ids = [filtered_doc_ids[i] for i in indices]
        elif num_docs is not None:
            filtered_docs = filtered_docs[:num_docs]
            filtered_doc_ids = filtered_doc_ids[:num_docs]
        if return_scores:
            return filtered_docs, filtered_doc_ids, None
        return filtered_docs, filtered_doc_ids

    if filter_name == 'fineweb':
        # NOTE: the threshold is set to 1 because the fineweb filter returns a score between 0 and 5
        print(f"Using fineweb_edu filter: {filter_name}")
        from document_filters.fineweb_edu_filter import fineweb_quality_filter
        quality_filter = fineweb_quality_filter
        cache_file = 'fineweb_scores.json'
    else:
        raise ValueError(f"Unknown filter name: {filter_name}")

    
    if cache_dir is not None:
        cache_dir = os.path.expanduser(cache_dir)
        # load scores from cache if available
        cache_path = os.path.join(cache_dir, cache_file)
        use_cache = True
    else:
        use_cache = False

    if use_cache and os.path.exists(cache_path):
        # NOTE: disabled caching since the filename is shared by different tasks
        with open(cache_path, 'r') as fin:
            score_dict = json.load(fin)
        scores = [score_dict[doc_id] for doc_id in doc_ids]
    else:
        scores = quality_filter(passages)
        score_dict = {}
        for i, score in enumerate(scores):
            score_dict[doc_ids[i]] = score
        if use_cache:
            with open(cache_path, 'w') as fout:
                json.dump(score_dict, fout)
                
    if debug:
        from copy import deepcopy
        all_docs = deepcopy(documents)
        for item in all_docs:
            if isinstance(item, dict):
                item['score'] = score_dict[item['doc_id']]
        # save the list of dict items into excel
        import pandas as pd
        df = pd.DataFrame(all_docs)
        output_filename = cache_path.replace('.json', '.pkl')
        # df.to_csv(excel_filename, index=False)
        df.to_pickle(output_filename)

    filtered_docs = []
    filtered_doc_ids = []
    if num_docs is not None:
        num_docs = min(num_docs, len(passages))
    sorted_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    # top10_scores = [scores[i] for i in sorted_ids[:10]]
    # print(f"Top 10 scores: {top10_scores}")
    
    for i in sorted_ids:
        if scores[i] >= threshold:
            filtered_docs.append(documents[i])
            filtered_doc_ids.append(doc_ids[i])
        if len(filtered_docs) >= num_docs:
            break
    if return_scores:
        return filtered_docs, filtered_doc_ids, scores
    return filtered_docs, filtered_doc_ids


def get_long_doc_id(doc_id):
    # The doc id is in the format of 'long_doc_id_%d_%d' or 'long_doc_id_%d'
    # We need to extract the "long_doc_id" to get the actual document id
    doc_id = doc_id.split('_')[:-1]
    doc_id = '_'.join(doc_id)
    doc_id = re.sub(r'_\d+$', '', doc_id)
    return doc_id + '.txt'

        
def random_sample_documents(documents, doc_ids, num_docs, seed=42):
    assert len(documents) == len(doc_ids)
    random.seed(seed)
    indices = list(range(len(documents)))
    sampled_indices = random.sample(indices, num_docs)
    sampled_documents = [documents[i] for i in sampled_indices]
    sampled_doc_ids = [doc_ids[i] for i in sampled_indices]
    return sampled_documents, sampled_doc_ids

# filter the documents with less than 1024 tokens
def chunk_filter(documents, chunk_size=1024):
    filtered_doc_ids = []
    for doc_id, doc in enumerate(documents):
        if len(doc.split(' ')) < 1024:
            continue
        filtered_doc_ids.append(doc_id)
    return filtered_doc_ids


# filter the documents with less than 20 tokens
def length_filter(documents):
    filtered_doc_ids = []
    for doc_id, doc in enumerate(documents):
        if len(doc.split(' ')) < 20:
            continue
        filtered_doc_ids.append(doc_id)
    return filtered_doc_ids


def deduplicate_docs(documents):
    unique_docs = set()
    unique_doc_ids = []
    for doc_id, doc in enumerate(documents):
        if isinstance(doc, dict):
            doc = doc['doc']
        if doc not in unique_docs:
            unique_docs.add(doc)
            unique_doc_ids.append(doc_id)
    unique_docs = [documents[i] for i in unique_doc_ids]
    return unique_docs, unique_doc_ids


def retrieve_gold_docs(subject, cache_dir='cache'):
    examples = get_examples(subject, cache_dir)

    docs = get_docs(subject, cache_dir=cache_dir)
    doc_dict = {}
    for d in docs:
        doc_dict[d['id']] = d['content']

    results = []
    for ex in examples:
        gold_ids = ex['gold_ids']
        for gold_id in gold_ids:
            result_dict = {
                'query': ex['query'], 
                'reasoning': ex['reasoning'],
                'doc_id': gold_id, 
                'doc': doc_dict[gold_id]
            }
            results.append(result_dict)
    return results


import tiktoken
def count_tokens(messages, model_name="gpt-4o"):
    enc = tiktoken.encoding_for_model(model_name)
    n_tokens = 0
    for message in messages:
        text = message['content']
        n_tokens += len(enc.encode(text))
    return n_tokens
        

if __name__ == "__main__":
    folder_name = 'top100'
    model_id = 'gpt-4o'
    path = f'synthetic_data/{folder_name}/all_docs_train_data/{model_id}'
    path = os.path.expanduser(path)
    from gen_utils import load_training_data
    data = load_training_data(path)
    # the data is structured as follows:
    # {subject: [{'query': ..., 'pos': ..., 'neg': ...}, ...]}