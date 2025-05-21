import re
import os
import json

r1_model_list = ["r1-llama-8b", "r1-qwen-32b", "r1-llama-70b", "qwq-32b", "huatuo-o1-8b", "huatuo-o1-70b",
                  "sky-32b", "qwen3-32b", "qwen3-8b", "qwen-7b-ins", "qwen-3b-ins"]
def split_response_for_r1(model, text):
    if model in ["r1-llama-8b", "r1-qwen-32b", "r1-llama-70b", "qwq-32b", "qwen3-32b", "qwen3-8b"]:
        try:
            think, response = text.split("</think>", 1)
            response = response.strip()
        except Exception as e:
            response = text
    elif model in ["huatuo-o1-8b", "huatuo-o1-70b"]:
        thinking_match = re.search(r"Thinking\s*(.*?)\s* Final Response", text, re.DOTALL)
        response_match = re.search(r"Final Response\s*(.*)", text, re.DOTALL)
        think = thinking_match.group(1).strip() if thinking_match else None
        response = response_match.group(1).strip() if response_match else None
        if response is None:
            response = text
    elif model in ["sky-32b"]:
        thinking_match = re.search(r"<\|begin_of_thought\|>\s*([\s\S]*?)\s*<\|end_of_thought\|>", text)
        response_match = re.search(r"<\|begin_of_solution\|>\s*([\s\S]*?)\s*<\|end_of_solution\|>", text)
        think = thinking_match.group(1).strip() if thinking_match else None
        response = response_match.group(1).strip() if response_match else None
        if response is None:
            response = text
    elif model in ["qwen-7b-ins", "qwen-3b-ins"]:
        response = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if len(response) == 0:
            response = text
        else:
            response = response[0].strip()
    else:
        response = text
    return response

def writejson_bench(data, json_file_path):
    dir_path = os.path.dirname(json_file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    num = 0
    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        for entry in data:
            jsonfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
            num += 1
    print(f"{json_file_path}共写入{num}条数据!")
