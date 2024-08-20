# LoRA Guide on LLaMA 3.1

This repository provides a detailed guide on fine-tuning the LLaMA 3.1 model using Low-Rank Adaptation (LoRA). The focus is on applying LoRA to optimize the fine-tuning process, particularly when working with limited computational resources.

## What is LoRA?

Low-Rank Adaptation (LoRA) is a technique used to reduce the number of trainable parameters in large language models by decomposing the weight matrices into lower-rank matrices. This allows for more efficient training without sacrificing much performance.


## Why Use LoRA?

- **Efficiency**: LoRA reduces the number of trainable parameters, making it possible to fine-tune large models on smaller hardware.
- **Scalability**: It enables the fine-tuning of models with billions of parameters without the need for massive computational resources.
- **Flexibility**: LoRA can be easily integrated into existing training pipelines with minimal changes.


### Loading the Dataset

In this guide, we use the `takala/financial_phrasebank` dataset from Hugging Face. The dataset is loaded, shuffled, and then converted into a Pandas DataFrame:

```python
import pandas as pd
from datasets import load_dataset

# Load and shuffle the dataset
dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree", split='train')
shuffled_dataset = dataset.shuffle(seed=32)
df = shuffled_dataset.to_pandas()

# Map sentiment labels to text (optional)
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
df['label'] = df['label'].map(label_mapping)
