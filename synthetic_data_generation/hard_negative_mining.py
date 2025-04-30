from datasets import load_dataset
from pyserini import analysis
from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity
import pdb
import json


class BM25_Miner():
    def __init__(self, documents=None, doc_ids=None, task=None, long_context=False, cache_dir='cache', dataset='bright', data_path=None):
        assert (documents is not None and doc_ids is not None) or (dataset in ['bright'] and task is not None) or dataset in ['msmarco']
        if documents is None:
            if dataset == 'msmarco':
                documents, doc_ids = self.get_ms_marco_documents(cache_dir)
            elif dataset == 'bright':
                documents, doc_ids = self.get_bright_documents(task, long_context, cache_dir)
            else:
                raise ValueError("Invalid dataset")
        self.dataset = dataset
        self.task = task if dataset in ['bright'] else None
        self.documents = documents
        self.doc_ids = doc_ids
        self.hashed_documents = self.get_hashed_documents(documents, doc_ids)
        self.analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
        corpus = [self.analyzer.analyze(x) for x in documents]
        self.dictionary = Dictionary(corpus)
        self.model = LuceneBM25Model(dictionary=self.dictionary, k1=0.9, b=0.4)
        bm25_corpus = self.model[list(map(self.dictionary.doc2bow, corpus))]
        self.bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(self.dictionary),
                                                normalize_queries=False, normalize_documents=False)
        
    def search(self, query):
        query = self.analyzer.analyze(query)
        bm25_query = self.model[self.dictionary.doc2bow(query)]
        similarities = self.bm25_index[bm25_query].tolist()
        all_scores = {}
        for did, s in zip(self.doc_ids, similarities):
            all_scores[did] = s
        cur_scores = sorted(all_scores.items(),key=lambda x:x[1],reverse=True)[:1000]
        all_scores = {}
        for pair in cur_scores:
            all_scores[pair[0]] = pair[1]
        return all_scores

    def get_ms_marco_documents(self, cache_dir):
        dataset = load_dataset("microsoft/ms_marco", "v1.1")
        # dataset = load_dataset("microsoft/ms_marco", "v2.1")
        doc_ids = []
        documents = []
        max_length = 0
        for dp in dataset['train']:
            passages = dp['passages']['passage_text']
            doc = ' '.join(passages)
            # chunk the document into 2000 words
            doc_split = doc.split()
            max_length = max(max_length, len(doc_split))
            doc = ' '.join(doc_split[:2000])
            documents.append(doc)
            doc_ids.append(str(dp['query_id']))
        print(f"Max length of all documents: {max_length}")
        return documents, doc_ids
    
    def get_bright_documents(self, task, long_context, cache_dir):
        if long_context:
            doc_pairs = load_dataset('xlangai/BRIGHT', 'long_documents', cache_dir=cache_dir)[task]
        else:
            doc_pairs = load_dataset('xlangai/BRIGHT', 'documents', cache_dir=cache_dir)[task]
        
        doc_ids = []
        documents = []
        for dp in doc_pairs:
            doc_ids.append(str(dp['id']))
            documents.append(dp['content'])
        return documents, doc_ids

    def get_hashed_documents(self, documents, doc_ids):
        hashed_documents = {}
        for docid, doc in zip(doc_ids, documents):
            hashed_documents[docid] = doc
        return hashed_documents
        
    
    def get_documents_text(self, docids):
        return [self.hashed_documents[docid] for docid in docids]

    def select_hard_negatives(self, query, gold_doc, num_neg=1, hard_neg_start_index=20):
        scores = self.search(query)
        
        num_added = 0
        hard_negatives_ids = []
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for i, (doc_id, score) in enumerate(sorted_scores):
            if i >= hard_neg_start_index:
                # avoid selecting false negative
                if self.hashed_documents[doc_id] != gold_doc:
                    hard_negatives_ids.append(doc_id)
                    num_added += 1
            if num_added == num_neg:
                break
        
        hard_negative_documents = self.get_documents_text(hard_negatives_ids)
        return hard_negative_documents
