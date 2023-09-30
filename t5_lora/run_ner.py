### Importing necessary packages
import datetime
import json
import yaml
import pandas as pd
import numpy as np
import warnings
import argparse
from pathlib import Path
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

warnings.filterwarnings("ignore")


### Parameters
base_url = "./instruction-ner-data/v2/"


def train(
    n_epochs: int,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    writer: Optional[SummaryWriter],
    device: torch.device,
    eval_every_n_batches: int,
    pred_every_n_batches: int,
    generation_kwargs: Dict[str, Any],
    options: List[str],
    path_to_save_model: Optional[str],
    metric_name_to_choose_best: str = "f1-score",
    metric_avg_to_choose_best: str = "weighted",
) -> None:

    metrics_best: Dict[str, Dict[str, float]] = {}

    for epoch in range(n_epochs):
        print(f"Epoch [{epoch + 1} / {n_epochs}]\n")

        metrics_best = train_epoch(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            writer=writer,
            device=device,
            epoch=epoch,
            eval_every_n_batches=eval_every_n_batches,
            pred_every_n_batches=pred_every_n_batches,
            generation_kwargs=generation_kwargs,
            options=options,
            path_to_save_model=path_to_save_model,
            metrics_best=metrics_best,
            metric_name_to_choose_best=metric_name_to_choose_best,
            metric_avg_to_choose_best=metric_avg_to_choose_best,
        )

        evaluate_metrics = evaluate(
            model=model,
            tokenizer=tokenizer,
            dataloader=test_dataloader,
            writer=writer,
            device=device,
            epoch=epoch,
            generation_kwargs=generation_kwargs,
            options=options,
        )

        if path_to_save_model is None:
            continue

        metrics_best = update_best_checkpoint(
            metrics_best=metrics_best,
            metrics_new=evaluate_metrics,
            metric_name=metric_name_to_choose_best,
            metric_avg=metric_avg_to_choose_best,
            model=model,
            tokenizer=tokenizer,
            path_to_save_model=path_to_save_model,
        )


def train_epoch(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    writer: Optional[SummaryWriter],
    device: torch.device,
    epoch: int,
    eval_every_n_batches: int,
    pred_every_n_batches: int,
    generation_kwargs: Dict[str, Any],
    options: List[str],
    path_to_save_model: Optional[str],
    metrics_best: Dict[str, Dict[str, float]],
    metric_name_to_choose_best: str = "f1-score",
    metric_avg_to_choose_best: str = "weighted",
    test_dataloader: torch.utils.data.DataLoader = None,
) -> Dict[str, Dict[str, float]]:
    """
    One training cycle (loop).
    Args:
        :param metric_avg_to_choose_best:
        :param metric_name_to_choose_best:
        :param metrics_best:
        :param path_to_save_model:
        :param options: list of labels in dataset
        :param generation_kwargs: arguments for generation (ex., beam_size)
        :param test_dataloader:
        :param train_dataloader:
        :param pred_every_n_batches: do sample prediction every n batches
        :param eval_every_n_batches: do evaluation every n batches
        :param model:
        :param optimizer:
        :param writer: tensorboard writer (optional)
        :param epoch: current epoch
        :param device: cpu or cuda
        :param tokenizer:
    """

    epoch_loss = []

    for i, inputs in tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc="Training",
    ):
        model.train()
        optimizer.zero_grad()

        inputs.pop("instances")
        answers = inputs.pop("answers")

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        answers = torch.tensor(answers.input_ids)
        answers[answers == tokenizer.pad_token_id] = -100

        inputs.to(device)
        answers = answers.to(device)
        outputs = model(**inputs, labels=answers)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        if writer:
            writer.add_scalar(
                "batch loss / train", loss.item(), epoch * len(train_dataloader) + i
            )

        if i % eval_every_n_batches == 0 and i >= eval_every_n_batches:
            if test_dataloader is not None:
                evaluate_metrics = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=test_dataloader,
                    writer=writer,
                    device=device,
                    epoch=epoch,
                    generation_kwargs=generation_kwargs,
                    options=options,
                )

                metrics_best = update_best_checkpoint(
                    metrics_best=metrics_best,
                    metrics_new=evaluate_metrics,
                    metric_name=metric_name_to_choose_best,
                    metric_avg=metric_avg_to_choose_best,
                    model=model,
                    tokenizer=tokenizer,
                    path_to_save_model=path_to_save_model,
                )

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")
    if writer:
        writer.add_scalar("loss / train", avg_loss, epoch)

    return metrics_best


### Eval utils
prediction_span_formatter = PredictionSpanFormatter()

def evaluate(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    dataloader: torch.utils.data.DataLoader,
    writer: Optional[SummaryWriter],
    device: torch.device,
    generation_kwargs: Dict[str, Any],
    epoch: int,
    options: List[str],
):
    model.eval()

    epoch_loss = []

    spans_true = []
    spans_pred = []

    with torch.no_grad():
        for i, inputs in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Evaluating",
        ):

            instances = inputs.pop("instances")
            contexts = [instance.context for instance in instances]
            spans_true_batch = [instance.entity_spans for instance in instances]
            spans_true.extend(spans_true_batch)

            answers = inputs.pop("answers")

            # replace padding token id's of the labels by -100 so it's ignored by the loss
            answers = torch.tensor(answers.input_ids)
            answers[answers == tokenizer.pad_token_id] = -100

            inputs.to(device)
            answers = answers.to(device)
            outputs = model(**inputs, labels=answers)
            loss = outputs.loss

            prediction_texts = model.generate(**inputs, **generation_kwargs)
            prediction_texts = tokenizer.batch_decode(
                prediction_texts, skip_special_tokens=True
            )
            if writer:
                writer.add_text("sample_prediction", prediction_texts[0])

            spans_pred_batch = [
                prediction_span_formatter.format_answer_spans(
                    context, prediction, options
                )
                for context, prediction in zip(contexts, prediction_texts)
            ]
            spans_pred.extend(spans_pred_batch)

            batch_metrics = calculate_metrics(
                spans_pred_batch, spans_true_batch, options=options
            )

            if writer:
                for metric_class, metric_dict in batch_metrics.items():
                    writer.add_scalars(
                        metric_class, metric_dict, epoch * len(dataloader) + i
                    )

            epoch_loss.append(loss.item())

            if writer:
                writer.add_scalar(
                    "batch loss / evaluation", loss.item(), epoch * len(dataloader) + i
                )

        epoch_metrics = calculate_metrics(spans_pred, spans_true, options=options)

        show_classification_report(epoch_metrics)

        return epoch_metrics


def get_sample_text_prediction(
    model: AutoModelForSeq2SeqLM,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    generation_kwargs,
    device: str,
    options: List[str],
    n: int = 3,
):
    """
    Generate sample N predictions
    :param model:
    :param dataloader:
    :param tokenizer:
    :param generation_kwargs: arguments for generation process
    :param device: cuda or cpu
    :param options: list of labels
    :param n: number of prediction to generate
    :return:
    """
    model.eval()

    dataset = dataloader.dataset

    ids_to_pick = random.sample(list(range(0, len(dataset))), n)

    for _id in ids_to_pick:
        dataset_item = dataset[_id]

        print(f"Input: {dataset_item.context}")
        print(f"{dataset_item.question}")

        input_ids = tokenizer(
            [dataset_item.context], [dataset_item.question], return_tensors="pt"
        ).input_ids

        input_ids = input_ids.to(device)
        answer = model.generate(input_ids, **generation_kwargs)
        answer = tokenizer.decode(
            answer[0], skip_special_tokens=True
        )  # TODO change to false

        answer_spans = prediction_span_formatter.format_answer_spans(
            context=dataset_item.context, prediction=answer, options=options
        )

        print(f"Prediction: {answer}")
        print(f"Found {len(answer_spans)} spans. {answer_spans}\n")


def update_best_checkpoint(
    metrics_new: Dict[str, Dict[str, float]],
    metrics_best: Dict[str, Dict[str, float]],
    metric_name: str,
    metric_avg: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    path_to_save_model: Optional[str],
):
    """
    Compares specific metric in two metric dictionaries: current and best.
    If new metric value is better -> new best model checkpoint saved
    :param metrics_new:
    :param metrics_best:
    :param metric_name:
    :param metric_avg:
    :param model:
    :param tokenizer:
    :param path_to_save_model:
    :return:
    """

    metric_current_value = metrics_new[metric_avg][metric_name]

    metric_best_value = 0.0
    if len(metrics_best) > 0:
        metric_best_value = metrics_best[metric_avg][metric_name]

    if metric_current_value > metric_best_value:
        print(
            f"Got Better results for {metric_name}. \n"
            f"{metric_current_value} > {metric_best_value}. Updating the best checkpoint"
        )
        metrics_best = metrics_new

        model.save_pretrained(path_to_save_model)
        tokenizer.save_pretrained(path_to_save_model)

        if path_to_save_model is not None:
            save_metrics_path = path_to_save_model + "/metrics.json"
        else:
            save_metrics_path = "metrics.json"

        with open(save_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_best, ensure_ascii=False, indent=4, fp=f)

    return metrics_best


def show_classification_report(metrics: Dict[str, Dict[str, float]]):
    """
    Based on dictionary of metrics show classification report aka sklearn
    :param metrics:
    :return:
    """
    df = pd.DataFrame.from_dict(metrics)
    print(df.transpose())



##### ====================================== xxx =========================================
#####                                   Run the model
##### ====================================== xxx =========================================
log_dir = "runs"
eval_n_batches = 3000
model_save_dir = "runs/model/"


args = get_train_args()
print("Args: ", args)

config = load_config(args.path_to_model_config)
print("Config: ", config)
set_global_seed(config["seed"])

writer = None
if log_dir is not None:
    log_dir = log_dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

# load all helper files
options = {}
with open("options.json", "r", encoding="utf-8") as fp:
    options = json.load(fp)
# options = load_json("options.json")
print("Options: ", options)

options = options[config["data"]["dataset"]]

instructions = {}
with open("instructions.json", "r", encoding="utf-8") as fp:
    instructions = json.load(fp)
# instructions = load_json("instructions.json")
print("Instructions: ", instructions)

# load data files
data_train = loads_json(config["data"]["train"])

valid_path = config["data"]["valid"]
if valid_path is None:
    data_train, data_valid = train_test_split(
        data_train, test_size=0.15, random_state=config["seed"]
    )
else:
    data_valid = loads_json(config["data"]["valid"])

# Create Datasets
train_dataset = T5NERDataset(
    data=data_train, instructions=instructions["train"], options=options
)

valid_dataset = T5NERDataset(
    data=data_valid, instructions=instructions["test"], options=options
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load model
# tokenizer = T5Tokenizer.from_pretrained(config["model"]["name"])
# model = T5ForConditionalGeneration.from_pretrained(config["model"]["name"], load_in_8bit=True, device_map = 'auto')

tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["name"], load_in_8bit = True, device_map = "auto")
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    target_modules = ["q", "v"],
    lora_dropout=0.1,
    bias="none"
)

### Prepare int-8 model for traiing
model = prepare_model_for_int8_training(model)

##add Lora adaptor
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.to(device)

tokenizer_kwargs = dict(config["tokenizer"])
generation_kwargs = dict(config["generation"])

if config["replace_labels_with_special_tokens"]:
    # TODO add special tokens to tokenizer and model
    pass

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=float(config["training"]["learning_rate"]),
)

collator = Collator(
    tokenizer=tokenizer,
    tokenizer_kwargs=tokenizer_kwargs,
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=int(config["training"]["batch_size"]),
    shuffle=True,
    collate_fn=collator,
)

valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=int(config["training"]["batch_size"]),
    shuffle=True,
    collate_fn=collator,
)

eval_every_n_batches = eval_n_batches
pred_every_n_batches = eval_n_batches

path_to_save_trained_model = Path(model_save_dir)
path_to_save_trained_model.mkdir(parents=True, exist_ok=True)

do_save_best_checkpoint = bool(config["training"]["do_save_best_checkpoint"])
path_to_save_best_checkpoint = None
if do_save_best_checkpoint:
    path_to_save_best_checkpoint = path_to_save_trained_model / "best"
    path_to_save_best_checkpoint.mkdir(exist_ok=True)

train(
    n_epochs=int(config["training"]["n_epoch"]),
    model=model,
    tokenizer=tokenizer,
    train_dataloader=train_dataloader,
    test_dataloader=valid_dataloader,
    optimizer=optimizer,
    writer=writer,
    device=device,
    eval_every_n_batches=eval_every_n_batches,
    pred_every_n_batches=pred_every_n_batches,
    generation_kwargs=generation_kwargs,
    options=options,
    path_to_save_model=path_to_save_best_checkpoint.as_posix(),
    metric_name_to_choose_best=config["training"]["metric_name"],
    metric_avg_to_choose_best=config["training"]["metric_avg"],
)

path_to_save_model_last = path_to_save_trained_model / "last"
path_to_save_model_last.mkdir(exist_ok=True)

model.save_pretrained(path_to_save_trained_model)
tokenizer.save_pretrained(path_to_save_trained_model)