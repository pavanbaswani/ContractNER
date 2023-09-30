### Importing necessary packages
import datetime
import json
import yaml
import pandas as pd
import numpy as np
import warnings
import argparse
from pathlib import Path
import random
import accelerate
from tqdm import tqdm

import torch
from typing import Any, Dict, List, Optional

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from instruction_ner.formatters.PredictionSpan import PredictionSpanFormatter
from instruction_ner.metrics import calculate_metrics

from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training

from instruction_ner.collator import Collator
from instruction_ner.dataset import T5NERDataset
from instruction_ner.utils.utils import (
    load_config,
    load_json,
    loads_json,
    set_global_seed,
)

from utils import get_train_args, get_data_args, get_evaluate_args
from eval_utils import *
prediction_span_formatter = PredictionSpanFormatter()
warnings.filterwarnings("ignore")

### Parameters and initial setup
access_token = "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"

set_global_seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
login(token=access_token)


### ==================
###  Helper Functions
### ==================
def get_predictions(model, tokenizer, test_dataloader):
    model.eval()
    dataset = test_dataloader.dataset

    eval_data = []
    for _id in tqdm(range(len(dataset))):
        dataset_item = dataset[_id]

        input_ids = tokenizer(
            [dataset_item.context], [dataset_item.question], return_tensors="pt"
        ).input_ids

        input_ids = input_ids.to(device)
        answer = model.generate(input_ids = input_ids)
        answer = tokenizer.decode(
            answer[0], skip_special_tokens=True
        )  # TODO change to false

        answer_spans = prediction_span_formatter.format_answer_spans(
            context=dataset_item.context, prediction=answer, options=options
        )
        
        eval_data.append({
            'input': dataset_item.context,
            'answer': answer,
            'span': [{'start': span.start, 'end': span.end, 'label': span.label} for span in answer_spans]
        })

    return eval_data


### ===================
###     Evaluation
### ===================

### Load Label2id and ID2Label
data_dict = load_json('label2id.json')
label2id = {}
for key, value in data_dict.items():
    label2id[key] = int(value)
    
data_dict = load_json('id2label.json')
id2label = {}
for key, value in data_dict.items():
    id2label[int(key)] = value


### load all Config, Options and Instructions
config = load_json('config.json')
options = load_json("options.json")
options = options[config["data"]["dataset"]]
instructions = load_json("instructions.json")


### Data loader for test dataset/inference dataset
tokenizer = T5Tokenizer.from_pretrained(config['model']['name'])
tokenizer_kwargs = dict(config["tokenizer"])
generation_kwargs = dict(config["generation"])

data_test = loads_json(config["data"]["test"])
test_dataset = T5NERDataset(
    data=data_test, instructions=instructions["test"], options=options
)

collator = Collator(
    tokenizer=tokenizer,
    tokenizer_kwargs=tokenizer_kwargs,
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=int(config["evaluation"]["batch_size"]),
    shuffle=True,
    collate_fn=collator,
)


### Load peft model with lora
lora_config = PeftConfig.from_pretrained(model_id, token = access_token)
inference_model = AutoModelForSeq2SeqLM.from_pretrained(
    't5-small', load_in_8bit = True, device_map = 'auto'
)
model = PeftModel.from_pretrained(inference_model, model_id)


eval_data = get_predictions(model, tokenizer, test_dataloader)
with open('inference.json', 'w', encoding='utf-8') as fp:
    json.dump(eval_data, fp)