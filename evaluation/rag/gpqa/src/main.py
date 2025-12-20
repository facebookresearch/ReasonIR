# run_naive_rag.py
import os
import logging
from omegaconf.omegaconf import OmegaConf, open_dict



from openai import OpenAI

from src.eval.evaluate import run_evaluation, extract_answer
from src.data.datasets import load_datasets
from src.workflow.naive_rag import NaiveRAG
from src.utils.hydra_runner import hydra_runner



@hydra_runner(config_path="conf", config_name="default")
def main(cfg):
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # ---------------------- Model Loading ----------------------
    llm = OpenAI(base_url=cfg.llm_endpoint, api_key="")
    if cfg.search_llm_endpoint is not None:
        search_llm = OpenAI(base_url=cfg.search_llm_endpoint, api_key="")
    else:
        search_llm = None
    # ---------------------- Dataset Loading ----------------------
    data = load_datasets(cfg)
    
    # ---------------------- Run Workflow ----------------------
    workflow = NaiveRAG(llm, cfg, search_llm=search_llm)
    outputs = workflow.run(data)
    
    # ---------------------- Evaluation ----------------------
    print("Evaluating generated answers...")
    # Define output directory based on model and dataset
    model_short_name = cfg.model_path.split('/')[-1].lower()
    output_dir = f'./outputs/runs.naive_rag/{cfg.dataset_name}/{model_short_name}.{cfg.search_engine}'
    os.makedirs(output_dir, exist_ok=True)
    
    run_evaluation(
        filtered_data=data,
        input_list=outputs.input_prompts,
        output_list=outputs.output_list,
        dataset_name=cfg.dataset_name,
        output_dir=output_dir,
        total_time=outputs.total_time,
        split=cfg.split,
    )

    print("Process completed.")

if __name__ == "__main__":
    main()