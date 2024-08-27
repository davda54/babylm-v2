from __future__ import annotations

import torch
import torch.nn as nn
import argparse
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils import seed_everything, cosine_schedule_with_warmup
from functools import partial
import json

from evaluator import train, evaluate
from dataset import Dataset, collate_function


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Required Parameters
    parser.add_argument("--results_dir", default="glue_results", type=str, help="The output directory where the results will be written.")
    parser.add_argument("--train_data", default="data/mnli.subs.jsonl", type=str, help="Path to file containing the training dataset, we expect it to be in a JSONL format.")
    parser.add_argument("--model_path_or_name", default="../lambada/baseline/baseline.bin", type=str, help="The local path to the model binary.")
    parser.add_argument("--tokenizer_path", default="../../tokenizer_100M.json", type=str, help="The vocabulary the model was trained on.")
    parser.add_argument("--config_file", default="../../configs/base.json", type=str)
    parser.add_argument("--architecture", default="base", type=str, help="The architecture of the model, available: base, attglu, attgate, densemod, densesubmod, densecont, elc, qkln")
    parser.add_argument("--metrics", default=["accuracy"], nargs='+', help="List of metrics to evaluate for the model (accuracy, f1, and mcc).")
    parser.add_argument("--num_labels", default=3, type=int, help="The number of labels in the dataset.")
    parser.add_argument("--seed", default=42, type=int, help="The seed for the Random Number Generator.")
    parser.add_argument("--task", default="mnli", type=str, help="The task to fine-tune for.")

    # Optinal Parameters
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Whether to output the metrics in terminal during the run.")
    parser.add_argument("--valid_data", type=str, help="Path to file containing the validation dataset to test on, we expect it to be in a JSONL format.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the fine-tuned model.")
    parser.add_argument("--save_dir", type=str, help="The directory in which to save the fine-tuned model.")
    parser.add_argument("--keep_best_model", action=argparse.BooleanOptionalAction, default=False, help="Whether to only keep the model with the best score based on the metric_for_valid.")
    parser.add_argument("--metric_for_valid", type=str, help="The metric used to compare the model when finding the best model.")
    parser.add_argument("--higher_is_better", action=argparse.BooleanOptionalAction, default=False, help="Wheter a higher value for the metric for valid is better or not.")

    # Hyperparameters
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--sequence_length", default=512, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)

    args = parser.parse_args()

    return args


class ModelForSequenceClassification(nn.Module):

    def __init__(self, args: argparse.Namespace) -> None:
        pass

    def forward(self, input_data: torch.LongTensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        pass


def load_config(args: argparse.Namespace) -> argparse.Namespace:
    with open(args.config_file, "r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()

    print(args)

    seed_everything(args.seed)

    tokenizer: Tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer.enable_padding(pad_id=3, pad_token="␢")
    tokenizer.enable_truncation(args.sequence_length)

    train_dataset: Dataset = Dataset(args.train_data)
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=partial(collate_function, tokenizer), shuffle=True, drop_last=True)

    valid_dataloader: DataLoader | None = None

    if args.valid_data is not None:
        valid_dataset: Dataset = Dataset(args.valid_data)
        valid_dataloader: DataLoader | None = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=partial(collate_function, tokenizer))

    args = load_config(args)

    model: nn.Modulde = ModelForSequenceClassification(args)

    model.transformer.load_state_dict(torch.load(args.model_path_or_name, map_location="cpu"))

    optimizer: torch.optim.Optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    scheduler: torch.optim.lr_scheduler.LRScheduler = cosine_schedule_with_warmup(optimizer, 20, args.epochs * len(train_dataloader))

    train(model, train_dataloader, args, optimizer, scheduler, valid_dataloader, args.verbose)

    if valid_dataloader is not None:
        metrics = evaluate(model, valid_dataloader, args.metrics, args.verbose)

        with open(f"{args.results_dir}/results_{args.model_path_or_name.split('/')[-1][:-4]}_{args.task}", "r") as file:
            file.write([f"{key}: {value}" for key, value in metrics].join("\n"))
