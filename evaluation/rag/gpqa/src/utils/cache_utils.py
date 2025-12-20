import os
import json


class Cache:
    def __init__(self, cfg):
        self.cfg = cfg
        # ---------------------- Caching Mechanism ----------------------
        # Define cache directories and file paths
        search_cache_dir = self.get_search_cache_dirname()
        self.search_cache_path = os.path.join(search_cache_dir, 'search_cache.json')
        self.url_cache_path = os.path.join(search_cache_dir, 'url_cache.json')
        os.makedirs(search_cache_dir, exist_ok=True)
        
        if self.cfg.use_query_rewriting:
            query_cache_dir = self.get_cot_query_cache_dirname()
            self.cot_query_path = os.path.join(query_cache_dir, f'query_cache.json')
            os.makedirs(query_cache_dir, exist_ok=True)
            
        # Load existing caches or initialize empty dictionaries
        if os.path.exists(self.search_cache_path):
            with open(self.search_cache_path, 'r', encoding='utf-8') as f:
                self.search_cache = json.load(f)
        else:
            self.search_cache = {}

        if os.path.exists(self.url_cache_path):
            with open(self.url_cache_path, 'r', encoding='utf-8') as f:
                self.url_cache = json.load(f)
        else:
            self.url_cache = {}
            
        if self.cfg.use_query_rewriting and os.path.exists(self.cot_query_path):
            with open(self.cot_query_path, 'r', encoding='utf-8') as f:
                self.cot_query_cache = json.load(f)
        elif self.cfg.use_query_rewriting:
            self.cot_query_cache = {}

    def get_search_cache_dirname(self,):
        cache_dir = f'./cache/{self.cfg.search_engine}'
        if self.cfg.use_query_rewriting:
            query_writer_name = self.cfg.model_path.split('/')[-1].replace('-', '_').replace('.', '_')
            cache_dir += f'/{query_writer_name}_cot_query'
        return cache_dir

    def get_cot_query_cache_dirname(self,):
        query_writer_name = self.cfg.model_path.split('/')[-1].replace('-', '_').replace('.', '_')
        cache_dir = f'./cache/cot_query/{self.cfg.search_engine}_with_query_rewrite_{query_writer_name}'
        return cache_dir
    
    # Function to save caches
    def save_caches(self,):
        with open(self.search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.search_cache, f, ensure_ascii=False, indent=2)
        with open(self.url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.url_cache, f, ensure_ascii=False, indent=2)
        if self.cfg.use_query_rewriting:
            with open(self.cot_query_path, 'w', encoding='utf-8') as f:
                json.dump(self.cot_query_cache, f, ensure_ascii=False, indent=2)