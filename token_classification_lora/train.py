from huggingface_hub import notebook_login, login
import os
from tqdm import tqdm
import pandas as pd
import json
from train_utils import *

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

from seqeval.metrics import classification_report
os.environ["WANDB_DISABLED"] = "true"

### ==============
###   Parameters
### ==============
seqeval = evaluate.load("seqeval")

config = load_json('config.json')
lr = config['lr']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
model_checkpoint = config['model_checkpoint']
base_url = config['dataset']
dataset_output_dir = config['outptu_dir']


### ======================
### Load Label2ID ID2Label
### ======================
data_dict = load_json('label2id.json')
label2id = {}
for key, value in data_dict.items():
    label2id[key] = int(value)
    
data_dict = load_json('id2label.json')
id2label = {}
for key, value in data_dict.items():
    id2label[int(key)] = value

label_list = list(label2id.keys())
print("Labels List: ", label_list)


### ================================================
### Loading train, test, dev dataset and reformating
### ================================================
train_data = reformat_data(base_url + "train.txt")
print("Length: ", len(train_data))
print("Train sample: ", train_data[0])

val_data = reformat_data(base_url + "dev.txt")
print("Length: ", len(val_data))
print("Val sample: ", val_data[0])

test_data = reformat_data(base_url + "test.txt")
print("Length: ", len(test_data))
print("Test sample: ", test_data[0])

if not os.path.exists(dataset_output_dir):
    os.makedirs(dataset_output_dir)
write_json(dataset_output_dir + "train.json", train_data)
write_json(dataset_output_dir + "valid.json", val_data)
write_json(dataset_output_dir + "test.json", test_data)


### Dataset loader
bionlp = load_dataset('json', data_files={'train': dataset_output_dir + 'train.json', 'valid': dataset_output_dir + 'valid.json', 'test': dataset_output_dir + 'test.json'})
print("Train Sample: ", bionlp["train"][0])
print("Dev Sample: ", bionlp["valid"][0])
print("Test Sample: ", bionlp["test"][0])


### Loading tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
tokenized_bionlp = bionlp.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list), id2label=id2label, label2id=label2id
)

### Adding peft and lora support
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


### ========================
###     Training
### ========================

training_args = TrainingArguments(
    output_dir="bert-large-token-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_bionlp["train"],
    eval_dataset=tokenized_bionlp["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()