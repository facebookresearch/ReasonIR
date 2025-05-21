from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
import json
from tqdm import tqdm
from collections import defaultdict
from instrcution import *
from openai import OpenAI
from base import writejson_bench

global_num = 0

class generate_func_by_gpt():
    def __init__(self, base_prompt, data_dir, save_file_path, model_name, method, pre_retrieval_path=None, temperature=0.8, top_p=0.8):
        self.base_prompt = base_prompt
        self.data_dir = data_dir
        self.save_file_path = save_file_path
        self.model_name = model_name
        self.temperature = temperature
        self.gar_method = method
        self.pre_retrieval_path = pre_retrieval_path
        self.top_p = top_p
        self.model = OpenAI(
            api_key="*****",
            base_url="*****"
        )

    def batch_list(self, data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def cut_text(self, text, threshold=512):
        text = text.split()
        if len(text) > threshold:
            text = text[:threshold]
        return " ".join(text)

    def read_data(self, data_dir, pre_retrieval_path):
        qrels = defaultdict(dict)
        with open(data_dir + "qrels.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                qrels[row["q_id"]][row["p_id"]] = 1
        id2doc = {}
        with open(data_dir + "corpus.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                id2doc[row["id"]] = row["text"]
        if self.gar_method == "lamer":
            pre_retrieval = dict()
            with open(pre_retrieval_path, "r", encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line)
                    id = e["id"]
                    passages = ""
                    for num, [pid, _] in enumerate(e["topk_pid"][:10]):
                        text = id2doc[pid].replace("\n", " ")
                        text = self.cut_text(text)
                        passages = passages + f"\n[{num+1}]. {text}"
                    pre_retrieval[id] = passages
        all_texts = []
        with open(data_dir + "query.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                qid = row["id"]
                row["doc"] = [id2doc[pid] for pid in qrels[qid].keys()]
                if self.gar_method == "lamer":
                    row["topk_passage"] = pre_retrieval[qid]
                all_texts.append(row)
        return all_texts

    def process(self):
        all_texts = self.read_data(self.data_dir, self.pre_retrieval_path)
        dir_path = os.path.dirname(self.save_file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        num = 0
        print(f"一共{len(all_texts)}个query！")
        global global_num
        global_num += 1
        results = []
        print_flag1, print_flag2 = 0, 0
        for entry in tqdm(all_texts):
            if self.gar_method == "lamer":
                input_prompt = self.base_prompt.format(TEXT=entry['text'], PASSAGE=entry["topk_passage"])
            else:
                input_prompt = self.base_prompt.format(TEXT=entry['text'])
            if print_flag1 == 0 and global_num == 1:
                print(input_prompt)
            print_flag1 += 1
            chat_completion = self.model.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_prompt}
                ],
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            generated_text = chat_completion.choices[0].message.content
            if print_flag2 == 0 and global_num == 1:
                print()
                print("generated_text")
                print(generated_text)
            print_flag2 += 1
            results.append({
                "id": entry["id"],
                "text": entry["text"],
                "hy_doc": generated_text,
                "doc_id": entry["doc_id"],
            })
            num += 1
        print()
        writejson_bench(results, self.save_file_path)

class generate_func():
    def __init__(self, base_prompt, data_dir, save_file_path, tokenizer, sampling_params, llm, method, pre_retrieval_path=None):
        self.base_prompt = base_prompt
        self.data_dir = data_dir
        self.save_file_path = save_file_path
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.llm = llm
        self.gar_method = method
        self.pre_retrieval_path = pre_retrieval_path

    def batch_list(self, data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def cut_text(self, text, threshold=512):
        text = text.split()
        if len(text) > threshold:
            text = text[:threshold]
        return " ".join(text)

    def read_data(self, data_dir, pre_retrieval_path):
        qrels = defaultdict(dict)
        with open(data_dir + "qrels.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                qrels[row["q_id"]][row["p_id"]] = 1
        id2doc = {}
        with open(data_dir + "corpus.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                id2doc[row["id"]] = row["text"]
        if self.gar_method == "lamer":
            pre_retrieval = dict()
            with open(pre_retrieval_path, "r", encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line)
                    id = e["id"]
                    passages = ""
                    for num, [pid, _] in enumerate(e["topk_pid"][:10]):
                        text = id2doc[pid].replace("\n", " ")
                        text = self.cut_text(text)
                        passages = passages + f"\n[{num+1}]. {text}"
                    pre_retrieval[id] = passages
        all_texts = []
        with open(data_dir + "query.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                qid = row["id"]
                row["doc"] = [id2doc[pid] for pid in qrels[qid].keys()]
                if self.gar_method == "lamer":
                    row["topk_passage"] = pre_retrieval[qid]
                all_texts.append(row)
        return all_texts

    def process(self, model=""):
        all_texts = self.read_data(self.data_dir, self.pre_retrieval_path)
        dir_path = os.path.dirname(self.save_file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        num = 0
        batch_size = 32
        print(f"一共{len(all_texts)}个query！")
        global global_num
        global_num += 1
        batchs = self.batch_list(all_texts, batch_size)
        results = []
        print_flag1, print_flag2 = 0, 0
        for batch in tqdm(batchs):
            batch_text = []
            for entry in batch:
                if self.gar_method == "lamer":
                    input_prompt = self.base_prompt.format(TEXT=entry['text'], PASSAGE=entry["topk_passage"])
                else:
                    input_prompt = self.base_prompt.format(TEXT=entry['text'])
                if print_flag1 == 0 and global_num == 1:
                    print(input_prompt)
                print_flag1 += 1
                sys_prompt = "You are a helpful assistant."
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": input_prompt}
                ]
                if model in ["qwen3-8b", "qwen3-32b"]:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True
                    )
                else:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                batch_text.append(text)
            outputs = self.llm.generate(batch_text, self.sampling_params, use_tqdm=False)
            for entry, output in zip(batch, outputs):
                generated_text = output.outputs[0].text
                if print_flag2 == 0 and global_num == 1:
                    print()
                    print("generated_text")
                    print(generated_text)
                print_flag2 += 1
                results.append({
                    "id": entry["id"],
                    "text": entry["text"],
                    "hy_doc": generated_text,
                    "doc_id": entry["doc_id"],
                })
                num += 1
        print()
        writejson_bench(results, self.save_file_path)

def generate_hy_doc(gar_method="", gar_llm="", task_names=[""]):
    seed = 42
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    model_dict = {
        "qwen-7b": "model_dir/Qwen/Qwen2.5-7B-Instruct",
        "qwen-32b": "model_dir/Qwen/Qwen2.5-32B-Instruct",
        "qwen-72b": "model_dir/Qwen/Qwen2.5-72B-Instruct",
        "llama-70b": "model_dir/LLM-Research/Meta-Llama-3.1-70B-Instruct",
        "gpt4": "gpt-4o-2024-11-20",
        "o3-mini": "o3-mini",
        "r1-qwen-32b": "model_dir/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "r1-llama-70b": "model_dir/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "huatuo-o1-70b": "model_dir/FreedomIntelligence/HuatuoGPT-o1-70B",
        "qwq-32b": "model_dir/Qwen/QwQ-32B",
        "qwen3-32b": "model_dir/Qwen/Qwen3-32B",
    }
    model_path = model_dict[gar_llm]
    if gar_llm not in ["gpt4", "o3-mini"]:
        sampling_params = SamplingParams(temperature=0.7, seed=seed, max_tokens=10240)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tensor_parallel_size = 4
        llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, disable_custom_all_reduce=True, trust_remote_code=True, gpu_memory_utilization=0.85)
        print(f"llm have downloaded from {model_path}!")
    for task_name in task_names:
        base_prompt = generate_hy_doc[gar_method][task_name]
        data_dir = f'../dataset/{task_name}/'
        save_path = f'../dataset/{task_name}/{gar_method}/{gar_llm}/query_with_hydoc.jsonl'
        pre_retrieval_path = None
        if gar_method == "lamer":
            pre_retrieval_path = f"../output/topk_docs/bm25/{task_name}.jsonl"
        if gar_llm in ["gpt4", "o3-mini"]:
            qua = generate_func_by_gpt(base_prompt, data_dir, save_path, model_path, gar_method, pre_retrieval_path=pre_retrieval_path)
            qua.process()
        else:
            qua = generate_func(base_prompt, data_dir, save_path, tokenizer, sampling_params, llm, gar_method, pre_retrieval_path)
            qua.process(gar_llm)