import os
import random

class OpenAILM():
    def __init__(self, model_id, temperature=1, top_p=1, seed=None, base_url=None, api_key=None) -> None:
        from openai import OpenAI
        if api_key is None:
            OPENAI_KEY = os.environ["OPENAI_API_KEY"]
        else:
            OPENAI_KEY = api_key
        self.client = OpenAI(api_key=OPENAI_KEY, base_url=base_url)
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

    def generate(self, prompt, system_prompt=None):
        messages = self.apply_chat_template(prompt, system_prompt)
        output = self.client.chat.completions.create(
                model=self.model_id,  # e.g., "gpt-4o-mini" with 128k context length
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed,
            )
        self.log_output(output)
        response = output.choices[0].message.content
        return response

    def generate_batch(self, prompts, system_prompt=None):
        responses = []
        for prompt in prompts:
            resp = self.generate(prompt, system_prompt)
            responses.append(resp)
        return responses
    
    def apply_chat_template(self, user_prompt, system_prompt):
        if system_prompt is not None:
            messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
        else:
            messages = [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": user_prompt},
                    ]
        return messages

    def log_output(self, output):
        print(output.choices[0].message.content)
        print(output.usage)


class MultiTurnOpenAILM(OpenAILM):
    def __init__(self, model_id, temperature=1, top_p=1, seed=None) -> None:
        super().__init__(model_id, temperature=temperature, top_p=top_p, seed=seed)

    def generate(self, messages):
        output = self.client.chat.completions.create(
                model=self.model_id,  # e.g., "gpt-4o-mini" with 128k context length
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed,
            )
        self.log_output(output)
        response = output.choices[0].message.content
        return response

    def apply_chat_template(self, user_prompt, system_prompt=None):
        if system_prompt is None:
            system_prompt = ""
        messages=[
                        self.format_prompt(system_prompt, role="system"),
                        self.format_prompt(user_prompt, role="user"),
                    ]
        return messages

    def append_history(self, messages, prompt, role="user"):
        messages.append(self.format_prompt(prompt, role))
        return messages

    def format_prompt(self, prompt, role="user"):
        if role == "user":
            return {"role": "user", "content": prompt}
        elif role == "system":
            return {"role": "system", "content": prompt}
        elif role == "assistant":
            return {"role": "assistant", "content": prompt}
        else:
            raise ValueError(f"Invalid role: {role}")


class HFLM():
    def __init__(self, model_id, temperature=0, top_p=1e-9, deterministic=False) -> None:
        import torch
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        
        self.model = LLM(
                model_id, 
                tensor_parallel_size=torch.cuda.device_count(),
                enable_chunked_prefill=False,
                gpu_memory_utilization=0.96,
                max_model_len=4096,
            )
            
        if deterministic:
            self.sampling_params = SamplingParams(
                max_tokens=1536,
                temperature=0,
                top_p=1e-9,
                do_sample=False,
                num_beams=1,
            )
        else:
            self.sampling_params = SamplingParams(
                    max_tokens=1536,
                    temperature=temperature,
                    top_p=max(top_p, 1e-9),
                )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    def generate(self, prompt, system_prompt=None):
        _prompt = prompt
        if system_prompt is not None:
            _prompt = self.apply_chat_template(system_prompt, prompt)
        output = self.model.generate(_prompt, self.sampling_params)

        self.log_output(output)
        return output[0].outputs[0].text

    def generate_batch(self, prompts, system_prompt=None):
        responses = []
        for prompt in prompts: 
            response = self.generate(prompt, system_prompt)
            responses.append(response)
        return responses

    def apply_chat_template(self, system_prompt, user_prompt):
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        return prompt

    def log_output(self, output):
        print(output[0].outputs[0].text)
        print(f"Number of input tokens: {len(output[0].prompt_token_ids)}")
        print(f"Number of output tokens: {len(output[0].outputs[0].token_ids)}")
            
