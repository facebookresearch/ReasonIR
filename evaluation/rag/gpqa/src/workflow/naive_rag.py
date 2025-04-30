import time
from tqdm import tqdm
import re
from dataclasses import dataclass, field

from src.utils.cache_utils import Cache
from apis.base import (
    search_api,
    format_document_string,
)
from apis.offline_cached_massiveds_searched_results import search_offline_cached_massiveds
from src.prompts.task_instructions import (
    get_task_user_prompt
)


@dataclass
class NaiveRAGOutput:
    all_relevant_info: None
    input_prompts: None
    output_list: None
    total_time: None


class NaiveRAG:
    def __init__(self, llm, cfg, search_llm=None):
        self.cfg = cfg
        self.llm = llm
        self.search_llm = search_llm
        self.cache = Cache(cfg)
        
    def run_search(self, data):
        print("Performing Bing Web Searches for all questions...")

        # Initialize a list to hold relevant information for each question
        all_relevant_info = []

        if self.cfg.search_engine == 'offline_massiveds':
            all_relevant_info = search_offline_cached_massiveds(data, self.cfg.top_k)        
        else:
            for item in tqdm(data, desc="Searching"):
                question = item['Question']
                # Check if the question has already been searched and cached
                if self.cache and question in self.cache.search_cache:
                    results = self.cache.search_cache[question]
                else:
                    search_question = question
                    results, new_cache = search_api(
                        self.cfg.search_engine, 
                        search_question, 
                        self.llm if self.search_llm is None else self.search_llm, 
                        self.cfg.model_path,
                        self.cfg.use_query_rewriting,
                        self.cache,
                        )
                    self.cache = new_cache
                    # self.cache.search_cache[question] = results['searched_results']
                all_relevant_info.append(results)

        # Save search cache after retrieval
        self.cache.save_caches()
        print("Search cache saved.")
        return all_relevant_info
    
    def get_naive_rag_instruction(self, question, documents):
        return (
            "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
            "Question:\n"
            f"{question}\n"
            "Documents:\n"
            f"{documents}\n"
        )
    
    def prepare_full_context(self, data, all_relevant_info):
        print("Constructing prompts for generation...")
        input_prompts = []

        # Set max input tokens to leave reasonable room for output while maximizing context usage
        # Model context length is 32,768 tokens
        # Reserve ~8k tokens for output, leaving ~24k for input
        max_input_tokens = 24000

        for idx, item in enumerate(tqdm(data, desc="Constructing Prompts")):
            question = item['Question']
            formatted_documents = format_document_string(self.cfg.search_engine, all_relevant_info[idx], self.cfg.top_k, max_doc_len=max_input_tokens)  # Increased document length
            
            instruction = self.get_naive_rag_instruction(question, formatted_documents)
            user_prompt = get_task_user_prompt(self.cfg.dataset_name, self.cfg.model_path, question)
            full_prompt = instruction + "\n\n" + user_prompt

            input_prompts.append(full_prompt)
        
        return input_prompts
    
    def llm_generate(self, input_prompts):
        # Generate model outputs
        print("Generating answers with LLM...")
        output_list = []
        for input_prompt in tqdm(input_prompts):
            try:
                chat_completion = self.llm.chat.completions.create(
                    model=self.cfg.model_path,
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": input_prompt}],
                    max_tokens=8000,  # Increased to allow for longer outputs while staying within context limit
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    frequency_penalty=self.cfg.repetition_penalty,
                )
                output_list.append(chat_completion.choices[0].message.content)
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                output_list.append("Error: Failed to generate response due to context length.")
                continue

        return output_list
    
    def run(self, data):
        
        all_relevant_info = self.run_search(data)
        input_prompts = self.prepare_full_context(data, all_relevant_info)
        
        start_time = time.time()
        output_list = self.llm_generate(input_prompts)
        total_time = time.time() - start_time
        
        return NaiveRAGOutput(
            all_relevant_info=all_relevant_info, 
            input_prompts=input_prompts, 
            output_list=output_list,
            total_time=total_time,
            )