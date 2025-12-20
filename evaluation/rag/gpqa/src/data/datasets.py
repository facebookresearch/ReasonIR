"""
Adapted from https://github.com/sunnynexus/Search-o1/blob/main/scripts/run_naive_rag.py
"""
import getpass
import json
import os

def get_slurm_user():
    # SLURM exposes this for most schedulers
    user = os.environ.get("SLURM_JOB_USER")
    if user:
        return user
    
    # Fallbacks just in case (e.g., interactive srun)
    return os.environ.get("USER") or getpass.getuser()


def load_datasets(cfg):
    # Paths to datasets
    slurm_user = get_slurm_user()

    if cfg.dataset_name == 'livecode':
        data_path = f'/home/{slurm_user}/ReasonIR/data/LiveCodeBench/{cfg.split}.json'
    elif cfg.dataset_name in ['math500', 'gpqa', 'aime', 'amc']:
        data_path = f'/home/{slurm_user}/ReasonIR/data/{cfg.dataset_name.upper()}/{cfg.split}.json'
    else:
        data_path = f'/home/{slurm_user}/ReasonIR/data/QA_Datasets/{cfg.dataset_name}.json'
        
        
    # ---------------------- Data Loading ----------------------
    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        if cfg.subset_num is not None:
            data = data[:cfg.subset_num]
            
    return data