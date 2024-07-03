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

## Evaluation of Translations
The evaluation of the generated translations is an essential part of the process to ensure the quality and accuracy of the translations. In this project, we use the BLEU (Bilingual Evaluation Understudy) score to evaluate the translations.

## BLEU Score
The BLEU score is a metric for evaluating a generated sentence to a reference sentence. It is commonly used in machine translation to compare machine-generated translations against human translations.

## Evaluation Process
To evaluate the translations, follow these steps:

Generate Translations: Use the translation model to generate translations for the input texts.
Compute BLEU Score: Compare the generated translations to the reference translations using the BLEU score.
Example Evaluation Code
Here is an example of how to compute the BLEU score for the generated translations:

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



## Interpretation of BLEU Score
The BLEU score ranges from 0 to 100, with 100 being a perfect match between the generated translation and the reference translation. Generally, a higher BLEU score indicates better translation quality.

- `0-10: Poor translation quality`
- `10-30: Fair translation quality`
- `30-50: Good translation quality`
- `50-100: Excellent translation quality`

## Example Output
The output of the BLEU score computation will look something like this:
```bash
BLEU score: 45.67
```
This score indicates that the generated translations have a good quality compared to the reference translations.
