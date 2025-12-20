import os
import json


original_query_retrieved_file = os.getenv("RETRIEVED_FILE")
retrieved_file = os.getenv("RETRIEVED_FILE")  # when evaluating query rewriting, replace it with the path to the retrieved file with query rewriting

def load_original_queries():
    queries = []
    with open(original_query_retrieved_file, 'r') as fin:
        for line in fin:
            result = json.loads(line)
            query = result['query']
            queries.append(query)
    return queries

def load_offline_searched_results(top_k=5):
    if "reasoning" in retrieved_file:
        queries = load_original_queries()
        query2docs = {}
        with open(retrieved_file, 'r') as fin:
            for idx, line in enumerate(fin):
                result = json.loads(line)
                if 'searched_results' in retrieved_file:
                    # search results from API calls
                    passages = result['results']['results']['passages'][0][:top_k]
                    passages = [{"retrieval text": psg} for psg in passages]
                    query = queries[idx]
                    query2docs[query] = passages
                else:
                    query = queries[idx]
                    passages = result['ctxs']
                    query2docs[query] = passages
    else:
        query2docs = {}
        with open(retrieved_file, 'r') as fin:
            for line in fin:
                result = json.loads(line)
                if 'searched_results' in retrieved_file:
                    # search results from API calls
                    passages = result['results']['results']['passages'][0][:top_k]
                    passages = [{"retrieval text": psg} for psg in passages]
                    query = result['results']['query']
                    query2docs[query] = passages
                else:
                    query = result['query']
                    passages = result['ctxs']
                    query2docs[query] = passages
    return query2docs

def search_offline_cached_massiveds(data, top_k=5):
    questions = [item['Question'] for item in data]
    query2docs = load_offline_searched_results(top_k)
    results = []
    for question in questions:
        if question in query2docs:
            results.append(query2docs[question])
        else:
            results.append(None)
            print(f"Question {question} not found in offline cached massiveds.")
    return results


def format_offline_document_string(results, top_k=5, max_doc_len=None):
    passage_str = ""
    for i in range(min(top_k, len(results))):
        passage_str += f"Passage {i+1}:\n{results[i]['retrieval text']}\n\n"
    print(len(passage_str.split(" ")))
    return passage_str.strip()

if __name__ == '__main__':
    load_offline_searched_results(5)