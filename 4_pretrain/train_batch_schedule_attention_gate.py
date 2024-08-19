# coding=utf-8

import os
import os.path
import argparse
from tqdm import tqdm
from itertools import count
from socket import gethostname
from tokenizers import Tokenizer
from statistics import mean
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from lamb import Lamb
from model_attention_gate import Bert
from utils import cosine_schedule_with_warmup_cooldown, is_main_process, get_rank, seed_everything, get_world_size
from dataset import Dataset, ValidationDataset
from model_logging import ModelLogger


if int(os.environ["SLURM_PROCID"]) == 0:
    import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="/pfs/lustrep1/scratch/project_465000144/dasamuel/babylm-v2/data/train_100M_tokenized.bin", type=str, help="Path to the training data.")
    parser.add_argument("--valid_path", default="/pfs/lustrep1/scratch/project_465000144/dasamuel/babylm-v2/data/dev_100M_tokenized.bin", type=str, help="Path to the validation data.")
    parser.add_argument("--name", default="base_attention_gate", type=str, help="Name of the run.")
    parser.add_argument("--config_file", default="/pfs/lustrep1/scratch/project_465000144/dasamuel/babylm-v2/configs/base.json", type=str, help="The BERT model config")
    parser.add_argument("--tokenizer_path", default="/pfs/lustrep1/scratch/project_465000144/dasamuel/babylm-v2/tokenizer_100M.json", type=str, help="Path to the tokenizer.")
    parser.add_argument("--output_dir", default="/pfs/lustrep1/scratch/project_465000144/dasamuel/babylm-v2/checkpoints", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--optimizer", default="lamb", type=str)
    parser.add_argument("--seq_length", default=128, help="Sequence length for training.")
    parser.add_argument("--batch_size", default=128, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--batch_reduction", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=31_250 // 4, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--validate_every", default=1_000, type=int, help="Run validation after every X training shards.")
    parser.add_argument("--validation_steps", default=1, type=int, help="Run validation after every X training shards.")
    parser.add_argument("--log_stats_every", default=100, type=int, help="Run validation after every X training shards.")
    parser.add_argument("--warmup_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--cooldown_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate cooldown for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--save_every', type=int, default=1_000, help="save every X steps")
    parser.add_argument("--mask_p_start", default=0.3, type=float, help="Masking probability.")
    parser.add_argument("--mask_p_end", default=0.15, type=float, help="Masking probability.")
    parser.add_argument("--mask_random_p", default=0.1, type=float, help="Masking probability.")
    parser.add_argument("--mask_keep_p", default=0.1, type=float, help="Masking probability.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--optimizer_eps", default=1e-8, type=float, help="Optimizer epsilon.")
    parser.add_argument("--optimizer_beta1", default=0.9, type=float, help="Optimizer beta1.")
    parser.add_argument("--optimizer_beta2", default=0.98, type=float, help="Optimizer beta2.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--n_special_tokens', default=16, type=int, help="Number of special tokens.")
    parser.add_argument('--z_loss_weight', default=1e-4, type=float, help="Weight for the z loss.")
    args = parser.parse_args()

    args.output_path = f"{args.output_dir}/{args.name}.bin"

    return args


def setup_training(args, tokenizer):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["SLURM_PROCID"])
    args.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert args.gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {args.rank} of {args.world_size} on {gethostname()} where there are {args.gpus_per_node} allocated GPUs per node.", flush=True)

    seed_everything(args.seed + args.rank)

    torch.distributed.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size)
    if args.rank == 0:
        print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)

    args.local_rank = args.rank - args.gpus_per_node * (args.rank // args.gpus_per_node)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    print(f"RCCL started on device {args.device}", flush=True)
    print(f"host: {gethostname()}, rank: {args.rank}, local_rank: {args.local_rank}")

    if is_main_process():
        print(f"Training for {args.max_steps:,} steps with {get_world_size()} GPUs")
        print(f"In total, the model will be trained on 'steps'({args.max_steps:,}) x 'GPUs'({get_world_size()}) x 'batch_size'({args.batch_size:,}) x 'seq_len'({args.seq_length:,}) = {args.max_steps * get_world_size() * args.batch_size * args.seq_length:,} subword instances")

    args.vocab_size = tokenizer.get_vocab_size()

    if is_main_process():
        wandb.init(
            name=args.name,
            project="BabyLM-v2",
            entity="nor-ret"
        )

def load_config(args):
    with open(args.config_file, "r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


def prepare_model_and_optimizer(args):
    args = load_config(args)
    model = Bert(args)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(args)
        wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    model.to(args.device)

    no_decay = ['bias', 'layer_norm']
    decay_params = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
    ]

    if is_main_process():
        print("Parameters without weight decay:")
        for n, _ in no_decay_params:
            print(n)
        print()
        print("Parameters with weight decay:")
        for n, _ in decay_params:
            print(n)
        print(flush=True)

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
 
    scheduler = cosine_schedule_with_warmup_cooldown(
        optimizer,
        int(args.max_steps * args.warmup_proportion),
        int(args.max_steps * args.cooldown_proportion),
        args.max_steps,
        0.1
    )

    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        bucket_cap_mb=torch.cuda.get_device_properties(args.device).total_memory,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True
    )

    return model, optimizer, scheduler


def get_batch(dataloader, device, global_step):
    dataloader._dataset.set_global_step(global_step)
    batch = next(dataloader)
    input_ids, target_ids, attention_mask, mask_p = [t.pin_memory().to(device, non_blocking=True) for t in batch]
    input_ids, target_ids = input_ids.t(), target_ids.t()
    mask_p = mask_p.mean()

    return input_ids, attention_mask, target_ids, mask_p


def training_epoch(model, train_dataloader, valid_dataloader, optimizer, scheduler, global_step, epoch, args):
    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    num_steps = min(len(train_dataloader), args.max_steps - global_step)
    train_dataloader = iter(train_dataloader)
    input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(train_dataloader, args.device, global_step)
    for local_step in tqdm(range(num_steps), desc="Train iteration", initial=global_step, total=args.max_steps, disable=not is_main_process()):
        
        input_ids, attention_mask, target_ids, mask_p = input_ids_, attention_mask_, target_ids_, mask_p_
        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            with ModelLogger(enable=global_step % 100 == 0, module=model):
                loss, accuracy, z_loss, num_tokens = model(input_ids, attention_mask, target_ids)

        if local_step < num_steps - 1: 
            input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(train_dataloader, args.device, global_step)

        total_tokens = torch.tensor(num_tokens, device=args.device, dtype=torch.long)
        torch.distributed.all_reduce(total_tokens, torch.distributed.ReduceOp.SUM)
        weight = args.world_size * num_tokens / total_tokens

        ((loss + args.z_loss_weight * z_loss) * weight).backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            metrics = torch.stack([loss * weight, accuracy * weight, z_loss * weight, mask_p])
            torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
            loss, accuracy, z_loss, mask_p = metrics.tolist()

        if is_main_process():
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": loss,
                    "train/z_loss": z_loss,
                    "train/perplexity": math.exp(loss),
                    "train/accuracy": accuracy * 100.0,
                    "stats/learning_rate": optimizer.param_groups[0]['lr'],
                    "stats/grad_norm": grad_norm,
                    "stats/seq_length": args.seq_length,
                    "stats/batch_size": args.current_batch_size,
                    "stats/mask_p": mask_p,
                },
                commit=False
            )

        optimizer.zero_grad(set_to_none=True)

        if global_step % args.save_every == 0:
            save(model, args)
            
        if global_step % args.validate_every == 0:
            validation_epoch(model, valid_dataloader, epoch, args)
            model.train()

        if is_main_process():
            wandb.log({"global_step": global_step}, commit=True)
        
        global_step += 1

        # Exiting the training due to hitting max steps
        if global_step >= args.max_steps:
            return global_step

    return global_step


@torch.no_grad()
def validation_epoch(model, valid_dataloader, epoch, args, commit=False):
    model = model.eval()

    losses, accuracies = [], []
    valid_dataloader = iter(valid_dataloader)
    input_ids, attention_mask, target_ids, mask_p = get_batch(valid_dataloader, args.device, 0)
    for local_step in tqdm(range(args.validation_steps), desc="Valid iteration", disable=not is_main_process()):

        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            with ModelLogger(enable=global_step % 100 == 0, module=model):
                loss, accuracy, _, num_tokens = model(input_ids, attention_mask, target_ids)

        if local_step < args.validation_steps - 1:
            input_ids, attention_mask, target_ids, mask_p = get_batch(valid_dataloader, args.device, 0)

        total_tokens = torch.tensor(num_tokens, device=args.device, dtype=torch.long)
        torch.distributed.all_reduce(total_tokens, torch.distributed.ReduceOp.SUM)
        weight = args.world_size * num_tokens / total_tokens
        
        metrics = torch.stack([loss * weight, accuracy * weight])
        torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
        loss, accuracy = metrics.tolist()

        losses.append(loss)
        accuracies.append(accuracy)

    if is_main_process():
        wandb.log(
            {
                "epoch": epoch,
                "validation/loss": mean(losses),
                "validation/accuracy": mean(accuracies) * 100.0,
                "validation/perplexity": math.exp(mean(losses))
            },
            commit=commit
        )


def save(model, args):
    if is_main_process():
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        torch.save(model_to_save.state_dict(), args.output_path)


def load_datasets(args, tokenizer, epoch, global_step, train_dataloader, valid_dataloader):
    train_seed = args.seed + get_rank() + epoch * get_world_size()

    if (global_step + 1) / args.max_steps >= 0.9:
        args.seq_length = 512
        batch_size = args.batch_size // 4
    elif (global_step + 1) / args.max_steps >= 0.7:
        args.seq_length = 256
        batch_size = args.batch_size // 2
    else:
        args.seq_length = 128
        batch_size = args.batch_size

    if train_dataloader is None or train_dataloader.dataset.seq_length != args.seq_length:
        train_data = Dataset(args.train_path, tokenizer, args)
        if is_main_process():
            train_data.show_random_item(tokenizer)
    else:
        train_data = train_dataloader.dataset

    # linear batch size scaling
    args.current_batch_size = int(batch_size / args.batch_reduction * (1 - global_step / args.max_steps) + batch_size * (global_step / args.max_steps) + 0.5)

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.current_batch_size,
        num_workers=0,  # non-zero num_workers causes segmenation fault
        generator=torch.Generator().manual_seed(train_seed),
        drop_last=True,
        pin_memory=True,
    )

    if valid_dataloader is None:
        valid_data = ValidationDataset(args.valid_path, tokenizer, args)

        valid_dataloader = DataLoader(
            valid_data,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=0,  # non-zero num_workers causes segmenation fault
            generator=torch.Generator().manual_seed(42),
            drop_last=True,
            pin_memory=True,
        )

    return train_dataloader, valid_dataloader


@torch.no_grad()
def measure_execution_time(model, args, num_runs=10):
    input_ids = torch.randint(0, 1000, (args.seq_length, args.batch_size)).to(model.device)
    attention_mask = torch.zeros(args.batch_size, args.seq_length, args.seq_length, dtype=torch.bool).to(model.device)
    target_ids = torch.randint_like(input_ids, 0, 1000)

    model.module(input_ids, attention_mask, target_ids)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    execution_times = []

    for _ in range(num_runs):
        torch.cuda.synchronize()
        start_event.record()

        model.module(input_ids, attention_mask, target_ids)

        end_event.record()
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        execution_times.append(execution_time)

    # Calculate average execution time
    average_time = sum(execution_times) / len(execution_times)

    return average_time


if __name__ == "__main__":
    args = parse_arguments()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    setup_training(args, tokenizer)
    model, optimizer, scheduler = prepare_model_and_optimizer(args)

    average_time = measure_execution_time(model, args)
    if is_main_process():
        print(f"Average execution time: {average_time:.2f} ms", flush=True)
        wandb.config.update({"average_execution_time": average_time})

    global_step, train_dataloader, valid_dataloader = 0, None, None
    for epoch in count():
        train_dataloader, valid_dataloader = load_datasets(args, tokenizer, epoch, global_step, train_dataloader, valid_dataloader)
        global_step = training_epoch(model, train_dataloader, valid_dataloader, optimizer, scheduler, global_step, epoch, args)

        if global_step >= args.max_steps:
            break

    save(model, args)
    validation_epoch(model, valid_dataloader, epoch, args, commit=True)
