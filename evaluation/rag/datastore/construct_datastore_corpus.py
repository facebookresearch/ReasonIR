import json
import os
import pdb



def load_jsonl(path):
    data = []
    with open(path, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    return data

def write_to_jsonl(data, path):
    with open(path, 'w') as fout:
        for ex in data:
            fout.write(json.dumps(ex) + '\n')


def construct_datastore_pool(input_dir, output_dir, top_k=float('inf')):
    """
    (Do not use this function which will save duplicated documents.)
    Construct a new corpus of the retrieved documents by Contriever for MMLU.
    Use this data pool to test out complex retrieval pipeline such as CoT, IRCoT.
    """

    os.makedirs(output_dir, exist_ok=True)

    for i, filename in enumerate(os.listdir(input_dir)):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.replace('_retrieved_results.jsonl', '.jsonl'))
        print(f"{i}: Processing {output_file}")

        if os.path.exists(output_file):
            continue
        
        data = load_jsonl(input_file)

        all_retrieved_documents = []
        for idx, ex in enumerate(data):
            K = min(top_k, len(ex['ctxs']))
            retrieved_documents = [{'id': idx, 'source': ctx['source'], 'text': ctx['retrieval text']} for ctx in ex['ctxs'][:K]]
            all_retrieved_documents.extend(retrieved_documents)
        
        write_to_jsonl(all_retrieved_documents, output_file)


def construct_deduplicated_datastore_pool(input_dir, output_dir, top_k=float('inf')):
    """
    Construct a new corpus of the retrieved documents by Contriever for MMLU.
    Use this data pool to test out complex retrieval pipeline such as CoT, IRCoT.
    Conduct deduplication on the raw data.
    """

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'mmlu_datastore_pool_dedupped.jsonl')
    if os.path.exists(output_file):
        print(f"Found prebuilt {output_file}")
        return
    
    unique_documents = set()
    all_retrieved_documents = []
    idx = 0

    for i, filename in enumerate(os.listdir(input_dir)):
        input_file = os.path.join(input_dir, filename)
        
        data = load_jsonl(input_file)

        for ex in data:
            K = min(top_k, len(ex['ctxs']))
            for ctx in ex['ctxs'][:K]:
                retrieval_text = ctx['retrieval text']
                if retrieval_text not in unique_documents:
                    all_retrieved_documents.append({'id': idx, 'source': ctx['source'], 'text': retrieval_text})
                    unique_documents.add(retrieval_text)
                    idx += 1
                else:
                    print(f"The document is in the pool. Skipped.")
        
        print(f"{i}: Proccessed {input_file}, {idx} passages added.")
        
    write_to_jsonl(all_retrieved_documents, output_file)
     
    
def construct_deduplicated_datastore_pool_from_api_searched_results(input_dir, output_dir):
    """
    Construct a new corpus of the retrieved documents by Contriever API for GPQA.
    Use this data pool to test out complex retrieval pipeline such as CoT, IRCoT.
    Conduct deduplication on the raw data.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'gpqa_datastore_pool_dedupped.jsonl')
    if os.path.exists(output_file):
        print(f"Found prebuilt {output_file}")
        return
    
    unique_documents = set()
    all_retrieved_documents = []
    idx = 0

    for i, filename in enumerate(os.listdir(input_dir)):
        input_file = os.path.join(input_dir, filename)
        
        data = load_jsonl(input_file)

        for ex in data:
            results = ex['results']['results']
            for retrieval_text in results["passages"][0]:
                if retrieval_text not in unique_documents:
                    assert isinstance(retrieval_text, str)
                    all_retrieved_documents.append({'id': idx, 'source': 'MassiveDS API', 'text': retrieval_text})
                    unique_documents.add(retrieval_text)
                    idx += 1
                else:
                    print(f"The document is in the pool. Skipped.")
        
        print(f"{i}: Proccessed {input_file}, {idx} passages added.")
        
    write_to_jsonl(all_retrieved_documents, output_file)
