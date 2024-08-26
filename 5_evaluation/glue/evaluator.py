import torch
from torch.nn import functional as F
from tqdm import tqdm
from typing import TYPE_CHECKING
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

if TYPE_CHECKING:
    import argparse


def train(model, train_dataloader, args, optimizer, scheduler, valid_dataloader=None, verbose=False):
    total_steps = args.epochs * len(train_dataloader)
    step = 0
    best_score = None

    for epoch in range(args.epochs):
        step = train_epoch(model, train_dataloader, args, epoch, step, total_steps, optimizer, scheduler, verbose)

        if valid_dataloader is not None:
            metrics = evaluate(model, valid_dataloader, args.metrics, verbose)
            score = metrics[args.metric_for_valid]

        if args.save:
            if args.keep_best_model and compare_scores(best_score, score, args.bigger_better):
                save_model(model, args)
                best_score = score
            elif not args.keep_best_model:
                save_model(model, args)


def train_epoch(model, train_dataloader, args, epoch, global_step, total_steps, optimizer, scheduler, verbose=False):
    progress_bar = tqdm(initial=global_step, total=total_steps)

    for input_data, attention_mask, labels in train_dataloader:
        optimizer.zero_grad()

        logits = model(input_data, attention_mask)  # loss = model(input_data, attention_mask, labels)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        optimizer.step()
        scheduler.step()

        metrics = calculate_metrics(logits, labels, args.metrics)

        metrics_string = [f"{key}: {value}" for key, value in metrics].join(", ")

        progress_bar.update()

        if verbose:
            progress_bar.set_postfix_str(metrics_string)

        global_step += 1

    progress_bar.close()

    return global_step


@torch.no_grad
def evaluate(model, valid_dataloader, metrics_to_calculate, verbose=False):
    progress_bar = tqdm(total=len(valid_dataloader))

    labels = []
    logits = []

    for input_data, attention_mask, label in valid_dataloader:
        logit = model(input_data, attention_mask)

        logits.append(logit)
        labels.append(label)

        progress_bar.update()

    labels = torch.cat(labels, dim=0)
    logits = torch.cat(logits, dim=0)

    metrics = calculate_metrics(logits, labels, metrics_to_calculate)

    progress_bar.close()

    if verbose:
        metrics_string = [f"{key}: {value}" for key, value in metrics].join("\n")
        print(metrics_string)

    return metrics


# TODO
def predict_classification(model, pred_dataloader, verbose=False):
    pass


def save_model(model: torch.Tensor, args: argparse.NameSpace) -> None:
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), args.output_path)


def compare_scores(best: float, current: float, bigger_better: bool) -> bool:
    if best is None:
        return True
    else:
        if current > best and bigger_better:
            return True
        elif current < best and not bigger_better:
            return True
        return False


def calculate_metrics(logits: torch.Tensor, labels: torch.Tensor, metrics_to_calculate: list) -> dict:
    predictions = logits.argmax(dim=-1).detach().numpy()
    labels = labels.detach().numpy()
    metrics = dict()

    for metric in metrics_to_calculate:
        if metric == "f1":
            metrics["f1"] = f1_score(labels, predictions)
        elif metric == "accuracy":
            metrics["accuracy"] = accuracy_score(labels, predictions)
        elif metric == "mcc":
            metrics["mcc"] = matthews_corrcoef(labels, predictions)

    return metrics
