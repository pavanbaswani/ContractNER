from huggingface_hub import notebook_login, login
import os
from tqdm import tqdm
import pandas as pd
import json

from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate
import torch
import numpy as np

from train_utils import *
from seqeval.metrics import classification_report

### ======================
### Parameters and config
### ======================
access_token = "<<huggingface_token_if_using_pushed_model>>"
login(token=access_token)

seqeval = evaluate.load("seqeval")
os.environ["WANDB_DISABLED"] = "true"

config = load_json('config.json')
model_checkpoint = config['model_checkpoint']
peft_model_id = "<<huggingface_model_id or checkpoint_folder_path>>"
base_url = config['dataset']


### =============================
### Load label2id and id2label
### =============================
data_dict = load_json('label2id.json')
label2id = {}
for key, value in data_dict.items():
    label2id[key] = int(value)
    
data_dict = load_json('id2label.json')
id2label = {}
for key, value in data_dict.items():
    id2label[int(key)] = value


### ===============================
### Loading the model and tokenizer
### ===============================
config = PeftConfig.from_pretrained(peft_model_id, token = access_token)
inference_model = AutoModelForTokenClassification.from_pretrained(
    config.base_model_name_or_path, num_labels=len(id2label), id2label=id2label, label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(inference_model, peft_model_id)


### ======================
### Inference Mode
### ======================
### Loading test dataset
test_sents = reformat_data(base_url + "test.txt")
print("# paras: ", len(test_sents))

### Getting the predictions for test set
dataset = []
for i in tqdm(range(len(test_sents))):

    sentence = ' '.join(test_sents[i])
    token_mapping = get_token_mapping(sentence)

    cause = []
    effect = []
    for entry in token_mapping:
        if entry['label'].lower() == "cause":
            cause.append(entry['token'])
        elif entry['label'].lower() == 'effect':
            effect.append(entry['token'])

    cause = ' '.join(cause)
    effect = ' '.join(effect)
    
    dataset.append({'Text': sentence, 'Cause': cause, 'Effect': effect})
print("Sample Prediction: ", dataset[0])

### Saving the predictions in csv file.
output_df = pd.DataFrame(data=dataset, columns = ['Text', 'Cause', 'Effect'])
print(output_df.shape)
print(output_df.head())
output_df.to_csv('predictions.csv', sep=';', index=False)