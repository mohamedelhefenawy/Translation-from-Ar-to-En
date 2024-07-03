# Translation from Arabic to English

This project contains a Jupyter notebook for translating Historical texts from Arabic to English using a pre-trained transformer model from the Hugging Face `transformers` library. The notebook includes steps for loading the model, preparing the data, generating translations, and evaluating the translation quality using BLEU score.

## Overview

The notebook demonstrates the following:

1. Loading a pre-trained translation model and tokenizer.
2. Preparing the input data for translation.
3. Generating translations in batches.
4. Evaluating the translations using the BLEU score.

## Prerequisites

Before running the notebook, ensure you have the following libraries installed:

- `transformers`
- `datasets`
- `torch`
- `sacrebleu`
- `tqdm`

## Custom Usage

Import Libraries

```bash
import pandas as pd
from transformers import AutoModelForSeq2SeqLM ,TrainingArguments, AutoTokenizer , DataCollatorForSeq2Seq,Seq2SeqTrainer , Trainer , MarianMTModel, MarianTokenizer
from datasets import Dataset , load_dataset,load_metric
import tensorflow as tf
import json
import torch
from tqdm import tqdm
```


## Data Processing Function
The following function is used to process the input data for translation. It tokenizes both the input texts in Arabic and the target texts in English, ensuring they are within the maximum length constraints. This function is essential for preparing the data before feeding it into the translation model.

```bash
def process(example):
  
# Extract input and target texts
    inputs = [ss for ss in example['عربي']]
    target = [ss for ss in example['English']]
    
    # Tokenize the input texts
    model_inputs = tokenizer(inputs, max_length=max_input, truncation=True)
    
    # Tokenize the target texts
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target, max_length=max_output, truncation=True)
    
    # Assign the tokenized target texts to the 'labels' key in the model inputs
    model_inputs['labels'] = labels['input_ids']
```


Follow the instructions in the notebook to:

Load the pre-trained translation model and tokenizer.
Prepare your input data (source texts in Arabic).
Generate translations for the input data.
Evaluate the generated translations using BLEU score.
Example Code
Below is a snippet of the function used to generate translations in batches:

```bash
def generate_translations_in_batches(model, tokenizer, texts, batch_size=32, max_length=256):
    translations = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        batch_translations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        translations.extend(batch_translations)
    return translations
```
