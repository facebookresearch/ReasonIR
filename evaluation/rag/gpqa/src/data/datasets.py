"""
Adapted from https://github.com/sunnynexus/Search-o1/blob/main/scripts/run_naive_rag.py
"""
import json


def load_datasets(cfg):
    # Paths to datasets
    if cfg.dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{cfg.split}.json'
    elif cfg.dataset_name in ['math500', 'gpqa', 'aime', 'amc']:
        data_path = f'./data/{cfg.dataset_name.upper()}/{cfg.split}.json'
    else:
        data_path = f'./data/QA_Datasets/{cfg.dataset_name}.json'
        
        
    # ---------------------- Data Loading ----------------------
    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        if cfg.subset_num is not None:
            data = data[:cfg.subset_num]
            
    return data