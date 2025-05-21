from transformers import AutoModel, AutoTokenizer, is_torch_npu_available
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer, BertModel
from transformers.file_utils import PaddingStrategy
from sentence_transformers import SentenceTransformer
import torch
from typing import cast, List, Dict, Union
import numpy as np
from tqdm import tqdm, trange
import json
import os
from gritlm import GritLM
from peft import PeftModel, PeftConfig
from torch import Tensor
from openai import OpenAI
import re
import tiktoken
import voyageai


class FlagDRESModel:
    def __init__(
            self,
            load_mode: str = "Automodel",
            model_name_or_path: str = None,
            encode_mode: str = "Base",
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            document_instruction_for_retrieval: str = None,
            batch_size: int = 512,
            max_length: int = 512,
            cache_path: str = "",
    ) -> None:
        if load_mode == "Automodel":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModel.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
            self.model = BertModel.from_pretrained(model_name_or_path)
        if encode_mode == "BMR":
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.document_instruction_for_retrieval = document_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.batch_size = batch_size
        self.max_length = max_length
        self.encode_mode = encode_mode
        self.cache_path = cache_path
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.batch_size = self.batch_size * num_gpus

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        print(f"# Instruction for query is : {self.query_instruction_for_retrieval}")
        if isinstance(queries[0], str):
            if self.query_instruction_for_retrieval is not None:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
            else:
                input_texts = queries
            print(f">>>> Encoding Queries <<<<")
            return self.encode(input_texts)
        else:
            queries_base = [q[0] for q in queries]
            if self.query_instruction_for_retrieval is not None:
                input_texts_base = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries_base]
            else:
                input_texts_base = queries_base
            print(f">>>> Encoding Queries <<<<")
            emb_base = self.encode(input_texts_base)
            print(f">>>> Encoding Rewrite Queries <<<<")
            queries_rewrite = [q[1] for q in queries]
            emb_rewrite = self.encode(queries_rewrite)
            emb_final = (emb_base + emb_rewrite) / 2
            return emb_final

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        print(f"# Instruction for doc is : {self.document_instruction_for_retrieval}")
        print(f">>>> Encoding Documents <<<<")
        if os.path.isfile(self.cache_path):
            print(f">>Document embedding already exists so we can just load it!")
            doc_embeddings = np.load(self.cache_path, allow_pickle=True)
            return doc_embeddings
        if isinstance(corpus[0], dict):
            input_texts = [doc['text'] for doc in corpus]
        else:
            input_texts = corpus
        if self.document_instruction_for_retrieval is not None:
            input_texts = ['{}{}'.format(self.document_instruction_for_retrieval, t) for t in input_texts]
        doc_embeddings = self.encode(input_texts)
        np.save(self.cache_path, doc_embeddings)
        return doc_embeddings

    @torch.no_grad()
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        if self.encode_mode == "Base":
            self.model.eval()
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences)<self.batch_size):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                inputs = self.tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=self.max_length,
                ).to(self.device)
                last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
                embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                embeddings = cast(torch.Tensor, embeddings)
                all_embeddings.append(embeddings.cpu().numpy())
            return np.concatenate(all_embeddings, axis=0)
        elif self.encode_mode == "BMR":
            self.model.eval()
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences)<self.batch_size):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                batch_dict = self.tokenizer(
                    sentences_batch,
                    max_length=self.max_length - 1,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    padding=PaddingStrategy.DO_NOT_PAD,
                    truncation=True
                )
                with torch.cuda.amp.autocast():
                    batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in
                                               batch_dict['input_ids']]
                    batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True,
                                                    return_tensors='pt').to(self.device)
                    outputs = self.model(**batch_dict)
                    embeddings = self.pooling(outputs.last_hidden_state, batch_dict['attention_mask'])
                    embeddings = cast(torch.Tensor, embeddings)
                    all_embeddings.append(embeddings.cpu().numpy())
            return np.concatenate(all_embeddings, axis=0)

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor=None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif self.pooling_method == 'last':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        elif self.pooling_method == 'last-bmr':
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden.shape[0]
                return last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]

class InstructorModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            document_instruction_for_retrieval: str = None,
            batch_size: int = 512,
            max_length: int = 512,
            cache_path: str = "",
    ) -> None:
        self.model = SentenceTransformer(model_name_or_path)
        self.model.set_pooling_include_prompt(False)
        print(f"模型默认的最大长度是{self.model.max_seq_length}")
        # self.model.max_seq_length = max_length
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.document_instruction_for_retrieval = document_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.batch_size = batch_size
        self.cache_path = cache_path

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.batch_size = self.batch_size * num_gpus

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        print(f"# Instruction for query is : {self.query_instruction_for_retrieval}")
        if isinstance(queries[0], str):
            print(f">>>> Encoding Queries <<<<")
            return self.encode(self.query_instruction_for_retrieval, queries)
        else:
            queries_base = [q[0] for q in queries]
            print(f">>>> Encoding Queries <<<<")
            emb_base = self.encode(self.query_instruction_for_retrieval, queries_base)
            print(f">>>> Encoding Rewrite Queries <<<<")
            queries_rewrite = [q[1] for q in queries]
            emb_rewrite = self.encode(self.query_instruction_for_retrieval, queries_rewrite)
            emb_final = (emb_base + emb_rewrite) / 2
            return emb_final

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        print(f"# Instruction for doc is : {self.document_instruction_for_retrieval}")
        print(f">>>> Encoding Documents <<<<")
        if os.path.isfile(self.cache_path):
            print(f">>Document embedding already exists so we can just load it!")
            doc_embeddings = np.load(self.cache_path, allow_pickle=True)
            return doc_embeddings
        if isinstance(corpus[0], dict):
            input_texts = [doc['text'] for doc in corpus]
        else:
            input_texts = corpus
        doc_embeddings = self.encode(self.document_instruction_for_retrieval, input_texts)
        np.save(self.cache_path, doc_embeddings)
        return doc_embeddings

    @torch.no_grad()
    def encode(self, prompt: str, sentences: List[str], **kwargs) -> np.ndarray:
        self.model.eval()
        embeddings = self.model.module.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=True,
            prompt=prompt,
            normalize_embeddings=True
        )
        return embeddings

class BiEncoderModel:
    def __init__(
            self,
            query_encoder_name_or_path: str = None,
            doc_encoder_name_or_path: str = None,
            query_instruction_for_retrieval: str = None,
            document_instruction_for_retrieval: str = None,
            batch_size: int = 512,
            max_length: int = 512,
            cache_path: str = "",
    ) -> None:
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_encoder_name_or_path)
        self.query_model = AutoModel.from_pretrained(query_encoder_name_or_path)
        self.doc_tokenizer = AutoTokenizer.from_pretrained(doc_encoder_name_or_path)
        self.doc_model = AutoModel.from_pretrained(doc_encoder_name_or_path)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.document_instruction_for_retrieval = document_instruction_for_retrieval
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_path = cache_path
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
        self.query_model = self.query_model.to(self.device)
        self.doc_model = self.doc_model.to(self.device)
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.query_model = torch.nn.DataParallel(self.query_model)
            self.doc_model = torch.nn.DataParallel(self.doc_model)
            self.batch_size = self.batch_size * num_gpus

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        print(f"# Instruction for query is : {self.query_instruction_for_retrieval}")
        if isinstance(queries[0], str):
            if self.query_instruction_for_retrieval is not None:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
            else:
                input_texts = queries
            print(f">>>> Encoding Queries <<<<")
            return self.encode(input_texts, "query")
        else:
            queries_base = [q[0] for q in queries]
            if self.query_instruction_for_retrieval is not None:
                input_texts_base = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries_base]
            else:
                input_texts_base = queries_base
            print(f">>>> Encoding Queries <<<<")
            emb_base = self.encode(input_texts_base, "query")
            print(f">>>> Encoding Rewrite Queries <<<<")
            queries_rewrite = [q[1] for q in queries]
            emb_rewrite = self.encode(queries_rewrite, "query")
            emb_final = (emb_base + emb_rewrite) / 2
            return emb_final

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        print(f"# Instruction for doc is : {self.document_instruction_for_retrieval}")
        print(f">>>> Encoding Documents <<<<")
        if os.path.isfile(self.cache_path):
            print(f">>Document embedding already exists so we can just load it!")
            doc_embeddings = np.load(self.cache_path, allow_pickle=True)
            return doc_embeddings
        if isinstance(corpus[0], dict):
            input_texts = [doc['text'] for doc in corpus]
        else:
            input_texts = corpus
        doc_embeddings = self.encode(input_texts, "doc")
        np.save(self.cache_path, doc_embeddings)
        return doc_embeddings

    @torch.no_grad()
    def encode(self, sentences: List[str], encode_mode: str, **kwargs) -> np.ndarray:
        if encode_mode == "query":
            self.query_model.eval()
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences)<self.batch_size):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                inputs = self.query_tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=self.max_length,
                ).to(self.device)
                last_hidden_state = self.query_model(**inputs).last_hidden_state
                embeddings = last_hidden_state[:, 0, :]
                embeddings = cast(torch.Tensor, embeddings)
                all_embeddings.append(embeddings.cpu().numpy())
            return np.concatenate(all_embeddings, axis=0)
        else:
            self.doc_model.eval()
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches",
                                    disable=len(sentences) < self.batch_size):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                title_batch = ["" for s in sentences_batch]
                inputs = self.doc_tokenizer(
                    title_batch,
                    sentences_batch,
                    padding=True,
                    truncation='longest_first',
                    return_tensors='pt',
                    max_length=self.max_length,
                ).to(self.device)
                last_hidden_state = self.doc_model(**inputs).last_hidden_state
                embeddings = last_hidden_state[:, 0, :]
                embeddings = cast(torch.Tensor, embeddings)
                all_embeddings.append(embeddings.cpu().numpy())
            return np.concatenate(all_embeddings, axis=0)

class HighScaleModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            encode_mode: str = "Base",
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            document_instruction_for_retrieval: str = None,
            batch_size: int = 512,
            max_length: int = 512,
            cache_path: str = "",
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path,device_map="auto")
        if encode_mode == "BMR":
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.document_instruction_for_retrieval = document_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.batch_size = batch_size
        self.max_length = max_length
        self.encode_mode = encode_mode
        self.cache_path = cache_path

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        print(f"# Instruction for query is : {self.query_instruction_for_retrieval}")
        if isinstance(queries[0], str):
            if self.query_instruction_for_retrieval is not None:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
            else:
                input_texts = queries
            print(f">>>> Encoding Queries <<<<")
            return self.encode(input_texts)
        else:
            queries_base = [q[0] for q in queries]
            if self.query_instruction_for_retrieval is not None:
                input_texts_base = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries_base]
            else:
                input_texts_base = queries_base
            print(f">>>> Encoding Queries <<<<")
            emb_base = self.encode(input_texts_base)
            print(f">>>> Encoding Rewrite Queries <<<<")
            queries_rewrite = [q[1] for q in queries]
            emb_rewrite = self.encode(queries_rewrite)
            emb_final = (emb_base + emb_rewrite) / 2
            return emb_final

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        print(f"# Instruction for doc is : {self.document_instruction_for_retrieval}")
        print(f">>>> Encoding Documents <<<<")
        if os.path.isfile(self.cache_path):
            print(f">>Document embedding already exists so we can just load it!")
            doc_embeddings = np.load(self.cache_path, allow_pickle=True)
            return doc_embeddings
        if isinstance(corpus[0], dict):
            input_texts = [doc['text'] for doc in corpus]
        else:
            input_texts = corpus
        if self.document_instruction_for_retrieval is not None:
            input_texts = ['{}{}'.format(self.document_instruction_for_retrieval, t) for t in input_texts]
        doc_embeddings =  self.encode(input_texts)
        np.save(self.cache_path, doc_embeddings)
        return doc_embeddings

    @torch.no_grad()
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        if self.encode_mode == "Base":
            self.model.eval()
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences)<self.batch_size):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                inputs = self.tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=self.max_length,
                ).to(self.device)
                last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
                embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                embeddings = cast(torch.Tensor, embeddings)
                all_embeddings.append(embeddings.cpu().numpy())
            return np.concatenate(all_embeddings, axis=0)
        elif self.encode_mode == "BMR":
            self.model.eval()
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences)<self.batch_size):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                batch_dict = self.tokenizer(
                    sentences_batch,
                    max_length=self.max_length - 1,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    padding=PaddingStrategy.DO_NOT_PAD,
                    truncation=True
                )
                with torch.cuda.amp.autocast():
                    batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in
                                               batch_dict['input_ids']]
                    batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True,
                                                    return_tensors='pt').to(self.device)
                    outputs = self.model(**batch_dict)
                    embeddings = self.pooling(outputs.last_hidden_state, batch_dict['attention_mask'])
                    embeddings = cast(torch.Tensor, embeddings)
                    all_embeddings.append(embeddings.cpu().numpy())
            return np.concatenate(all_embeddings, axis=0)

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor=None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif self.pooling_method == 'last':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        elif self.pooling_method == 'last-bmr':
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden.shape[0]
                return last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]

class GritModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            query_instruction_for_retrieval: str = None,
            document_instruction_for_retrieval: str = None,
            batch_size: int = 512,
            query_max_length: int = 512,
            doc_max_length: int = 2048,
            cache_path: str = "",
    ) -> None:
        self.model = GritLM(model_name_or_path,torch_dtype="auto", mode="embedding",device_map="auto")
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.document_instruction_for_retrieval = document_instruction_for_retrieval
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.cache_path = cache_path
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        print(f"# Instruction for query is : {self.query_instruction_for_retrieval}")
        if isinstance(queries[0], str):
            input_texts = queries
            print(f">>>> Encoding Queries <<<<")
            return self.encode(self.query_instruction_for_retrieval,self.query_max_length, input_texts)
        else:
            queries_base = [q[0] for q in queries]
            input_texts_base = queries_base
            print(f">>>> Encoding Queries <<<<")
            emb_base = self.encode(self.query_instruction_for_retrieval,self.query_max_length, input_texts_base)
            print(f">>>> Encoding Rewrite Queries <<<<")
            queries_rewrite = [q[1] for q in queries]
            emb_rewrite = self.encode(self.query_instruction_for_retrieval,self.query_max_length, queries_rewrite)
            emb_final = (emb_base + emb_rewrite) / 2
            return emb_final

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        print(f"# Instruction for doc is : {self.document_instruction_for_retrieval}")
        print(f">>>> Encoding Documents <<<<")
        if os.path.isfile(self.cache_path):
            print(f">>Document embedding already exists so we can just load it!")
            doc_embeddings = np.load(self.cache_path, allow_pickle=True)
            return doc_embeddings
        if isinstance(corpus[0], dict):
            input_texts = [doc['text'] for doc in corpus]
        else:
            input_texts = corpus
        doc_embeddings =  self.encode(self.document_instruction_for_retrieval,self.doc_max_length, input_texts)
        np.save(self.cache_path, doc_embeddings)
        return doc_embeddings


    @torch.no_grad()
    def encode(self, instruction: str, max_length: int, sentences: List[str], **kwargs) -> np.ndarray:
        self.model.eval()
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences)<self.batch_size):
            sentences_batch = sentences[start_index:start_index + self.batch_size]
            embeddings = self.model.encode(sentences_batch, instruction=instruction, batch_size=self.batch_size, max_length=max_length)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)

class NVEmbedModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            document_instruction_for_retrieval: str = None,
            batch_size: int = 512,
            max_length: int = 512,
            cache_path: str = "",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.document_instruction_for_retrieval = document_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_path = cache_path
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        print(f"# Instruction for query is : {self.query_instruction_for_retrieval}")
        if isinstance(queries[0], str):
            print(f">>>> Encoding Queries <<<<")
            return self.encode(self.query_instruction_for_retrieval, queries)
        else:
            queries_base = [q[0] for q in queries]
            print(f">>>> Encoding Queries <<<<")
            emb_base = self.encode(self.query_instruction_for_retrieval, queries_base)
            print(f">>>> Encoding Rewrite Queries <<<<")
            queries_rewrite = [q[1] for q in queries]
            emb_rewrite = self.encode(self.query_instruction_for_retrieval, queries_rewrite)
            emb_final = (emb_base + emb_rewrite) / 2
            return emb_final

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        print(f"# Instruction for doc is : {self.document_instruction_for_retrieval}")
        print(f">>>> Encoding Documents <<<<")
        if os.path.isfile(self.cache_path):
            print(f">>Document embedding already exists so we can just load it!")
            doc_embeddings = np.load(self.cache_path, allow_pickle=True)
            return doc_embeddings
        if isinstance(corpus[0], dict):
            input_texts = [doc['text'] for doc in corpus]
        else:
            input_texts = corpus
        doc_embeddings = self.encode(self.document_instruction_for_retrieval, input_texts)
        np.save(self.cache_path, doc_embeddings)
        return doc_embeddings

    @torch.no_grad()
    def encode(self, prompt: str, sentences: List[str], **kwargs) -> np.ndarray:
        self.model.eval()
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches",
                                disable=len(sentences) < self.batch_size):
            sentences_batch = sentences[start_index:start_index + self.batch_size]
            embeddings = self.model.encode(sentences_batch, instruction=prompt, max_length=self.max_length)
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

class RetrievalOPENAI:
    def __init__(
            self,
            model_name_or_path: str = None,
            query_cache_path: str = None,
            doc_cache_path: str = None,
            query_instruction_for_retrieval: str = None,
            document_instruction_for_retrieval: str = None,
            batch_size: int = 128,
    ) -> None:
        if model_name_or_path == "text-embedding-3-large":
            self.model = OpenAI(
                api_key="*****",
                base_url="*****"
            )
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif "voyage" in model_name_or_path:
            self.model = voyageai.Client(
                api_key="*****",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(f'../voyageai/{model_name_or_path}')
        self.model_name = model_name_or_path
        self.query_cache_path = query_cache_path,
        self.doc_cache_path = doc_cache_path,
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.document_instruction_for_retrieval = document_instruction_for_retrieval
        self.batch_size = batch_size


    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        print(f"# Instruction for query is : {self.query_instruction_for_retrieval}")
        if isinstance(queries[0], str):
            if os.path.isfile(self.query_cache_path):
                print(f">>Query embedding already exists so we can just load it!")
                query_embeddings = np.load(self.query_cache_path, allow_pickle=True)
                return query_embeddings
            if self.query_instruction_for_retrieval is not None:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
            else:
                input_texts = queries
            print(f">>>> Encoding Queries <<<<")
            query_embeddings = self.encode(input_texts, "query")
            np.save(self.query_cache_path, query_embeddings)
            return query_embeddings
        else:
            queries_base = [q[0] for q in queries]
            if self.query_instruction_for_retrieval is not None:
                input_texts_base = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries_base]
            else:
                input_texts_base = queries_base
            print(f">>>> Encoding Queries <<<<")
            emb_base = self.encode(input_texts_base, "query")
            print(f">>>> Encoding Rewrite Queries <<<<")
            queries_rewrite = [q[1] for q in queries]
            emb_rewrite = self.encode(queries_rewrite, "query")
            emb_final = (emb_base + emb_rewrite) / 2
            return emb_final

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        print(f"# Instruction for doc is : {self.document_instruction_for_retrieval}")
        print(f">>>> Encoding Documents <<<<")
        if os.path.isfile(self.doc_cache_path):
            print(f">>Document embedding already exists so we can just load it!")
            doc_embeddings = np.load(self.doc_cache_path, allow_pickle=True)
            return doc_embeddings
        if isinstance(corpus[0], dict):
            input_texts = [doc['text'] for doc in corpus]
        else:
            input_texts = corpus
        if self.document_instruction_for_retrieval is not None:
            input_texts = ['{}{}'.format(self.document_instruction_for_retrieval, t) for t in input_texts]
        doc_embeddings = self.encode(input_texts, "document")
        np.save(self.doc_cache_path, doc_embeddings)
        return doc_embeddings

    def cut_text(self, text, threshold):
        text_ids = self.tokenizer(text)['input_ids']
        if len(text_ids) > threshold:
            text = self.tokenizer.decode(text_ids[:threshold])
        return text
    def cut_text_openai(self, text, threshold=6000):
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) > threshold:
            text = self.tokenizer.decode(token_ids[:threshold])
        return text

    @torch.no_grad()
    def encode(self, sentences: List[str],mode:str, **kwargs) -> np.ndarray:
        if self.model_name == "text-embedding-3-large":
            sentences = [self.cut_text_openai(s) for s in sentences]
        else:
            sentences = [self.cut_text(s, 16000) for s in sentences]
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences)<self.batch_size):
            sentences_batch = sentences[start_index:start_index + self.batch_size]
            texts = [json.dumps(text.replace("\n", " ")) for text in sentences_batch]
            success = False
            threshold = 6000
            count = 0
            cur_emb = None
            exec_count = 0
            while not success:
                exec_count += 1
                if exec_count > 5:
                    print('execute too many times')
                    exit(0)
                try:
                    if self.model_name == "text-embedding-3-large":
                        emb_obj = self.model.embeddings.create(input=texts, model=self.model_name).data
                        cur_emb = [e.embedding for e in emb_obj]
                    else:
                        cur_emb = self.model.embed(texts, model=self.model_name, input_type=mode).embeddings
                    success = True
                except Exception as e:
                    print(e)
                    count += 1
                    threshold -= 500
                    if count > 4:
                        print('openai cut', count)
                        exit(0)
                    new_texts = []
                    for t in texts:
                        new_texts.append(self.cut_text_openai(text=t, threshold=threshold))
                    texts = new_texts
            if cur_emb is None:
                raise ValueError("Fail to embed, openai")
            if cur_emb is not None:
                all_embeddings.append(cur_emb)
        return np.concatenate(all_embeddings, axis=0)

class RerankerModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            batch_size: int = 8,
            max_length: int = 512,
            mode: str="bge",
    ) -> None:
        self.mode = mode
        if self.mode == "llama":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path[1])
            base_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path[1], num_labels=1,
                                                                            device_map="auto", trust_remote_code=True,
                                                                            torch_dtype=torch.float16)
            self.model = PeftModel.from_pretrained(base_model, model_name_or_path[0])
            self.model = self.model.merge_and_unload()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.batch_size = batch_size
        self.max_length = max_length
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
        if self.mode != "llama":
            self.model = self.model.to(self.device)
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                self.model = torch.nn.DataParallel(self.model)

    @torch.no_grad()
    def predict(self, sentence_pairs: List[List], **kwargs) -> np.ndarray:
        self.model.eval()
        if self.mode == "bge":
            inputs = self.tokenizer(
                sentence_pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.max_length,
            ).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = scores.cpu().tolist()
        elif self.mode == "mono":
            queries, passages = zip(*sentence_pairs)  # 分别拿出query和passage
            inputs = self.tokenizer(
                list(queries),
                list(passages),
                return_tensors="pt",
                truncation=True,
                padding="longest",
                max_length=self.max_length,
            ).to(self.device)
            scores = self.model(**inputs).logits[:, 1].squeeze(-1)  # shape (batch_size,)
            scores = scores.cpu().tolist()
        elif self.mode == "llama":
            scores = []
            for sentence_pair in sentence_pairs:
                q_text = f"query: {sentence_pair[0]}"
                p_text = f"document: {sentence_pair[1]}"
                inputs = self.tokenizer(
                    q_text,
                    p_text,
                    return_tensors='pt',
                    max_length=self.max_length,
                ).to(self.device)
                score = self.model(**inputs).logits[0][0]
                scores.append(score.cpu().float().item())
        return scores
