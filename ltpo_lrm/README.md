<h1 align="center">R2MED: A Benchmark for Reasoning-Driven Medical Retrieval</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2505.14558">
        <img src="https://img.shields.io/badge/arXiv-2410.07166-B31B1B.svg?style=plastic&logo=arxiv" alt="arXiv">
    </a>
    <a href="https://r2med.github.io/">
        <img src="https://img.shields.io/badge/Website-R2MED-purple?style=plastic&logo=Google%20chrome" alt="Website">
    </a>
    <a href="https://huggingface.co/R2MED" target="_blank">
        <img src="https://img.shields.io/badge/Dataset-Download-yellow?style=plastic&logo=huggingface" alt="Download the EmbodiedAgentInterface Dataset from Hugging Face">
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=plastic" alt="License: MIT">
    </a>
<!--     <a href="https://github.com/embodied-agent-interface/embodied-agent-interface/tree/main/dataset">
        <img src="https://img.shields.io/badge/Dataset-Download-yellow?style=plastic&logo=Data" alt="Dataset">
    </a> -->
</p>

<p align="center">
    Lei Li, 
    Xiao Zhou, 
    Zheng Liu, 
</p>
<p align="center">Gaoling School of Artificial Intelligence, Renmin University of China</p>
<p align="center">Beijing Academy of Artificial
Intelligence</p>

<p align="center">
    <a href="https://r2med.github.io/" target="_blank">
        <img src="./images/R2MED.png" alt="R2MED" width="80%" height="80%" border="10" />
    </a>
</p>

## 🔭 Overview

### R2MED: First Reasoning-Driven Medical Retrieval Benchmark

 **R2MED** is a high-quality, high-resolution synthetic information retrieval (IR) dataset designed for medical scenarios. It contains 876 queries with three retrieval tasks, five medical scenarios, and twelve body systems.

| Dataset | #Q | #D | Avg. Pos | Q-Len | D-Len |   
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|
| [Biology](https://huggingface.co/datasets/R2MED/Biology) | 103 | 57359   | 3.6     | 115.2   | 83.6    |
| [Bioinformatics](https://huggingface.co/datasets/R2MED/Bioinformatics) | 77  | 47473   | 2.9     | 273.8   | 150.5   |
| [Medical Sciences](https://huggingface.co/datasets/R2MED/Medical-Sciences) | 88  | 34810   | 2.8     | 107.1   | 122.7   |
| [MedXpertQA-Exam](https://huggingface.co/datasets/R2MED/MedXpertQA-Exam) | 97  | 61379   | 3.0     | 233.2   | 154.9   |
| [MedQA-Diag](https://huggingface.co/datasets/R2MED/MedQA-Diag) | 118 | 56250   | 4.4     | 167.8   | 179.7   |
| [PMC-Treatment](https://huggingface.co/datasets/R2MED/PMC-Treatment) | 150 | 28954   | 2.1     | 449.3   | 149.3   |
| [PMC-Clinical](https://huggingface.co/datasets/R2MED/PMC-Clinical) | 114 | 60406   | 2.2     | 182.8   | 480.4   |
| [IIYi-Clinical](https://huggingface.co/datasets/R2MED/IIYi-Clinical) | 129 | 10449   | 3.5     | 602.3   | 1273.0  |

## ⚙️  Installation
Note that the code in this repo runs under **Linux** system. We have not tested whether it works under other OS.

1. **Clone this repository:**

    ```bash
    git clone https://github.com/R2MDE/R2MED.git
    cd R2MED
    ```

2. **Create and activate the conda environment:**

    ```bash
    conda create -n r2med python=3.10
    conda activate r2med
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip install mteb==1.1.1
    pip install transformers==4.44.2
    pip install vllm==0.5.4
    ```

## 💾 Datasets Preparation

### Download the R2MED dataset:

- **R2MED:** [![HF Datasets](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-yellow?style=flat-square)](https://huggingface.co/R2MED)

    Place all zip files under `./R2MED/dataset` and extract them.

### Data Structure:

For each dataset, the data is expected in the following structure:

```
${DATASET_ROOT} # Dataset root directory, e.g., /home/username/project/R2MED/dataset/Biology
├── query.jsonl        # Query file
├── corpus.jsonl        # Document file
└── qrels.txt         # Relevant label file
```

## 💽 Evaluate
You can find pretrained RS3DAda checkpoints here. At the moment we provide the following model versions:


## 📜Reference

If this code or dataset contributes to your research, please kindly consider citing our paper and give this repo ⭐️ :)
```
@article{xxx,
  title={R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  author={ Lei Li, Xiao Zhou, and Zheng Liu},
  journal={arXiv preprint arXiv:xxxx},
  year={2025}
}
```
