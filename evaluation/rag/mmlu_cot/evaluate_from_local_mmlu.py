import csv
import json
import argparse
import os
import torch
import random
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset
import pdb
from extract_mmlu_group import get_mmlu_group_subjects

choices = ["A", "B", "C", "D"]
ids_to_letters = {0: "A", 1: "B", 2: "C", 3: "D"}
max_model_length = 4096
max_new_tokens = 2048


def load_mmlu():
    dataset = load_dataset("cais/mmlu", 'all')
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def load_model():
    # TODO: support data parallel
    # refer to https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/vllm_causallms.py#L240
    llm = LLM(model=args.model, gpu_memory_utilization=float(args.gpu_util),
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=max_model_length,
                trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                        stop=["Question:"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    return (llm, sampling_params), tokenizer


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["choices"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["choices"] = options
        res_df.append(each)
    return res_df


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    concat_k = f'Concat_{args.concat_k}'
    return [model_name, scoring_method, subjects, concat_k]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["subject"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["choices"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["subject"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def generate_rag_cot_prompt(val_df, curr, k, hashed_retrieval_results):
    question = curr["question"]
    if question in hashed_retrieval_results.keys():
        formatted_question = question
    else:
        choices = curr["choices"]
        subject = curr["subject"]
        instruction = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"
        formatted_question = f"{instruction}{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
        
    if formatted_question not in hashed_retrieval_results.keys():
        if curr['subject'] == 'business_ethics':
            pass
        elif curr['subject'] == 'college_medicine':
            pass
        elif curr['subject'] == 'formal_logic':
            pass
        else:
            pdb.set_trace()
        rag_context = ""
    else:
        rag_context = "Related background:\n"
        for ctx in hashed_retrieval_results[formatted_question]:
            rag_context += ctx['retrieval text'] + '\n\n'

    prompt = ""
    with open(f"cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["subject"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return rag_context + prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-D])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-D])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-D]\b(?!.*\b[A-D]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(llm, sampling_params, inference_batch):
    start = time.time()
    outputs = llm.generate(inference_batch, sampling_params)
    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, len(each["choices"]) - 1)
            if x == each["answer"]:
                corr += 1
                # print("random hit.")
            else:
                wrong += 1
        elif each["pred"] == ids_to_letters[each["answer"]]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


def get_hashed_retrieval_results(retrieval_path, k=3, raw_query_file=None):
    if retrieval_path is None or k == 0:
        return {}
    
    data = []
    with open(retrieval_path, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    
    if raw_query_file is not None:
        print(f"Using raw queries provided in {raw_query_file} to create hashes")
        raw_queries = []
        with open(raw_query_file, 'r') as fin:
            for line in fin:
                raw_queries.append(json.loads(line))
    
        assert len(raw_queries) == len(data), f"Raw queries size mismatched: expecting {len(data)}, got {len(raw_queries)}."
        
        hashed_retrieval_results = {}
        for ex, raw_query in zip(data, raw_queries):
            query = raw_query['query']
            topk_ctxs = ex['ctxs'][:min(k, len(ex['ctxs']))]
            key = query
            print(key)
            hashed_retrieval_results[key] = topk_ctxs
    
    else:
        hashed_retrieval_results = {}
        for ex in data:
            query = ex['question'] if 'question' in ex else ex['raw_query']
            topk_ctxs = ex['ctxs'][:min(k, len(ex['ctxs']))]

            key = query
            print(key)
            hashed_retrieval_results[key] = topk_ctxs

    return hashed_retrieval_results


@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path):
    llm, sampling_params = model
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
        assert k == 0, "Error: MMLU does not have CoT fewshot examples."
        curr = test_df[i]
        prompt = generate_cot_prompt(val_df, curr, k)
        message = [{"role": "system", "content": "You are a helpful assistent."}, {"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        inference_batches.append(prompt)

    pred_batch, response_batch = batch_inference(llm, sampling_params, inference_batches)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong


@torch.no_grad()
def eval_rag_cot(subject, model, tokenizer, val_df, test_df, output_path, hashed_retrieval_results):
    llm, sampling_params = model
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
        assert k == 0, "Error: MMLU does not have CoT fewshot examples."
        curr = test_df[i]
        prompt = generate_rag_cot_prompt(val_df, curr, k, hashed_retrieval_results)
        message = [{"role": "system", "content": "You are a helpful assistent."}, {"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        inference_batches.append(prompt)

    pred_batch, response_batch = batch_inference(llm, sampling_params, inference_batches)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong


def main():
    model, tokenizer = load_model()
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    full_test_df, full_val_df = load_mmlu()
    all_subjects = []
    for each in full_test_df:
        if each["subject"] not in all_subjects:
            all_subjects.append(each["subject"])
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)
    
    # Check every subject is assigned a group properly
    groups, subject_to_group = get_mmlu_group_subjects()
    for subject in selected_subjects:
        assert subject in subject_to_group.keys(), subject
        print(subject_to_group[subject])
    
    # Load retrieval file
    use_rag = args.retrieval_file is not None and args.concat_k > 0
    if use_rag:
        hashed_retrieval_results = get_hashed_retrieval_results(args.retrieval_file, args.concat_k, args.raw_query_file)
    
    # Start evaluation
    logging.info("selected subjects:\n" + "\n".join(selected_subjects))
    print("selected subjects:\n" + "\n".join(selected_subjects))
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))
        if use_rag:
            acc, corr_count, wrong_count = eval_rag_cot(subject, model, tokenizer, val_df, test_df, output_path, hashed_retrieval_results)
        else:
            acc, corr_count, wrong_count = eval_cot(subject, model, tokenizer, val_df, test_df, output_path)
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    
    # Compute group-wise accuracy
    group_weighted_acc, group_names = [], []
    for group_name, group in groups.items():
        total_corr, total_wrong = 0.0, 0.0
        for k, v in sta_dict.items():
            if k in group:
                total_corr += v["corr"]
                total_wrong += v["wrong"]
        total_accu = total_corr / (total_corr + total_wrong + 0.000001)
        # sta_dict[group_name] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}
        group_names.append(group_name)
        group_weighted_acc.append(total_accu)

    # Compute total accuracy
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    # Write out results
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------group-wise average acc sta------\n")
        for group_name, weighted_acc in zip(group_names, group_weighted_acc):
            f.write("Average accuracy ({}): {:.4f}\n".format(group_name, weighted_acc))
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc] + group_weighted_acc
        writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--retrieval_file", type=str, default=None)
    parser.add_argument("--raw_query_file", type=str, default=None, 
                        help="Pass the original query to this argument, where the order of the query matches the reasoning query in the retrieval_file.")
    parser.add_argument("--concat_k", type=int, default=0)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    main()


