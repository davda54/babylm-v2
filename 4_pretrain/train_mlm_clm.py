import os
import os.path
import argparse
from tqdm import tqdm
from itertools import count
from tokenizers import Tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from lamb import Lamb
from config import BertConfig
from model_mlm_clm import EncoderModel
from utils import cosine_schedule_with_warmup, is_main_process, get_rank, seed_everything, get_world_size
from datasets_mlm_clm import DatasetMLM, DatasetCausal

import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_path", default="../data/processed/cached_128_10M_104.txt", type=str, help="The input data dir. Should contain .hdf5 files for the task.")
    parser.add_argument("--para_path", default="../data/paraphrase/paraphrased.txt", type=str, help="The input data dir. Should contain .hdf5 files for the task.")
    parser.add_argument("--mirror_input_path", default="../data/processed/cached_{mirror_length}_10M_104.txt", type=str, help="The input data dir. Should contain .hdf5 files for the task.")
    parser.add_argument("--name", default="XS_MCLM_Unbalanced_Inputs", type=str)
    parser.add_argument("--config_file", default="../configs/xs.json", type=str, help="The BERT model config")
    parser.add_argument("--output_dir", default="../checkpoints/xs_mclm_unblanced_inputs", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--tokenizer_path", default="../tokenizer_small_104.json", type=str, help="The vocabulary the BERT model will train on.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to a previous checkpointed training state.")

    # Other parameters
    parser.add_argument("--optimizer", default="lamb", type=str)
    parser.add_argument("--use_paraphrases", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_dummy_decoder", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_memory", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--mirror_length", default=32, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=512, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=31250 // 2, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--log_freq', type=int, default=10, help='frequency of logging loss.')
    parser.add_argument("--mask_p", default=0.15, type=float, help="Masking probability.")
    parser.add_argument("--short_p", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument("--mlm_ratio", default=(1.0 / 1.15), type=float)
    parser.add_argument("--gradient_accumulation", default=1, type=int)
    parser.add_argument("--causal_mult", default=1.25, type=float)
    parser.add_argument("--learned", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    return args


@torch.no_grad()
def log_parameter_histograms(model, step):
    for name, param in model.named_parameters():
        wandb.log(
            {
                f"parameters/norm_{name}": torch.linalg.norm(param.data).cpu().item(),
                f"parameters/std_{name}": param.data.std().cpu().item(),
            },
            step=step,
            commit=False
        )
        if param.requires_grad and param.grad is not None:
            wandb.log(
                {
                    f"gradients/norm_{name}": torch.linalg.norm(param.grad).cpu().item(),
                    f"gradients/std_{name}": param.grad.std().cpu().item(),
                },
                step=step,
                commit=False
            )
        if "beta" in name:
            # d =param.data.cpu().numpy()
            d = F.hardsigmoid(param.data.cpu()).numpy()
            param_dict = {f"layer_weights/{name}_{i}": d[i] for i in range(len(d))}
            wandb.log(
                param_dict,
                step=step,
                commit=False
            )
        if "gamma" in name:
            # d =param.data.cpu().numpy()
            d = F.hardsigmoid(param.data.cpu()).numpy()
            param_dict = {f"layer_weights/{name}_{i}": d[i] for i in range(len(d))}
            wandb.log(
                param_dict,
                step=step,
                commit=False
            )
        if "alpha" in name:
            # d =param.data.cpu().numpy()
            d = F.hardsigmoid(param.data.cpu()).numpy()
            param_dict = {f"layer_weights/{name}_emb": d[0]}
            wandb.log(
                param_dict,
                step=step,
                commit=False
            )
        if "delta" in name:
            # d =param.data.cpu().numpy()
            d = F.hardsigmoid(param.data.cpu()).numpy()
            param_dict = {f"layer_weights/{name}_att": d[0]}
            wandb.log(
                param_dict,
                step=step,
                commit=False
            )


def setup_training(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    local_rank = 0

    seed_everything(args.seed)

    if is_main_process():
        os.system(f"mkdir -p {args.output_dir}")


    if is_main_process():
        print(f"Training for {args.max_steps:,} steps with {get_world_size()} GPUs")
        print(f"In total, the model will be trained on 'steps'({args.max_steps:,}) x 'GPUs'({get_world_size()}) x 'batch_size'({args.batch_size:,}) x 'seq_len'({args.seq_length:,}) = {args.max_steps * get_world_size() * args.batch_size * args.seq_length:,} subword instances")

    args.device_max_steps = args.max_steps

    if is_main_process():
        wandb.init(
            name=args.name,
            config=args,
            project="MLM Shift",
            entity="lgcharpe"
        )

    return device, local_rank


def prepare_model_and_optimizer(args, device, local_rank, checkpoint):
    config = BertConfig(args.config_file)

    model = EncoderModel(config, args.causal_mult)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(config.to_dict())
        wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    if checkpoint is not None:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)

    no_decay = ['bias', 'layer_norm', '_embedding']
    decay_params = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad]
    no_decay_params = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
    ]

    if is_main_process():
        param_names = []
        print("Parameters with weight decay:")
        for n, _ in decay_params:
            print(n)
            param_names.append(n)
        print()
        print("Parameters without weight decay:")
        for n, _ in no_decay_params:
            print(n)
            param_names.append(n)
        print(flush=True)

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            param_names=param_names,
        )
 
    scheduler = cosine_schedule_with_warmup(optimizer, int(args.max_steps * args.warmup_proportion), args.max_steps, 0.1)

    return model, config, optimizer, scheduler


def training_epoch(model, train_dataloader, optimizer, scheduler, global_step, epoch, args, device, max_local_steps, causal_ratio):
    model = model.train()
    optimizer.zero_grad(set_to_none=True)
    iterations, total_loss, total_causal_loss, total_causal_perplexity, total_causal_accuracy, total_mlm_loss, total_mlm_perplexity, total_mlm_accuracy = 0, 0, 0, 0, 0, 0, 0, 0

    if is_main_process():
        train_iter = tqdm(train_dataloader, desc="Train iteration", initial=global_step*args.gradient_accumulation, total=args.device_max_steps*args.gradient_accumulation)
    else:
        train_iter = train_dataloader

    for local_step, batch in enumerate(train_iter):
        original_inputs, original_attention_mask, original_outputs, causal_pos = [t.to(device, non_blocking=True) for t in batch]
        original_inputs, original_outputs = original_inputs.t(), original_outputs.t()
        causal_pos = causal_pos.squeeze()

        loss, causal_loss, mlm_loss, causal_perplexity, mlm_perplexity, causal_accuracy, mlm_accuracy = model(original_inputs, original_attention_mask, original_outputs, causal_pos, causal_ratio)
        
        total_loss += loss.item()

        total_causal_loss += causal_loss
        total_causal_perplexity += causal_perplexity
        total_causal_accuracy += causal_accuracy

        total_mlm_loss += mlm_loss
        total_mlm_perplexity += mlm_perplexity
        total_mlm_accuracy += mlm_accuracy
        
        loss = loss / args.gradient_accumulation
        loss.backward()

        iterations += 1

        if iterations == args.gradient_accumulation:

            total_loss /= iterations

            total_causal_loss /= iterations
            total_causal_perplexity /= iterations
            total_causal_accuracy /= iterations

            total_mlm_loss /= iterations
            total_mlm_perplexity /= iterations
            total_mlm_accuracy /= iterations

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)

            optimizer.step(step=global_step)
            optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            global_step += 1
            #with torch.no_grad():
            #    metrics = torch.stack([total_loss, total_perplexity, total_accuracy])
            #    total_loss, total_perplexity, total_accuracy = metrics.tolist()
            
            if global_step % 20 == 0 and args.learned and sum(causal_pos):
                temp_loss = model(original_inputs, original_attention_mask, original_outputs, causal_pos, causal_ratio, only_causal=True)
                temp_loss.backward()
                causal_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 100000)
                optimizer.zero_grad(set_to_none=True)

                temp_loss = model(original_inputs, original_attention_mask, original_outputs, causal_pos, causal_ratio, only_mlm=True)
                temp_loss.backward()
                mlm_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 100000)
                optimizer.zero_grad(set_to_none=True)

                causal_ratio = max(0.0, mlm_grad_norm / causal_grad_norm)

            if is_main_process():
                train_iter.set_postfix_str(f"loss: {total_loss:.2f}, mlm_accuracy: {total_mlm_accuracy * 100.0:.2f}, grad_norm: {grad_norm:.2f}, lr: {optimizer.param_groups[0]['lr']:.5f}")

                if global_step % 10 == 0:
                    log_parameter_histograms(model, global_step)

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": total_loss,
                        "train/causal_loss": total_causal_loss,
                        "train/causal_perplexity": total_causal_perplexity,
                        "train/causal_accuracy": total_causal_accuracy * 100.0,
                        "train/mlm_loss": total_mlm_loss,
                        "train/mlm_perplexity": total_mlm_perplexity,
                        "train/mlm_accuracy": total_mlm_accuracy * 100.0,
                        "stats/learning_rate": optimizer.param_groups[0]['lr'],
                        "stats/grad_norm": grad_norm,
                        "stats/causal_ratio": causal_ratio,
                    },
                    step=global_step,
                )

            iterations, total_loss, total_causal_loss, total_causal_perplexity, total_causal_accuracy, total_mlm_loss, total_mlm_perplexity, total_mlm_accuracy = 0, 0, 0, 0, 0, 0, 0, 0

        # Exiting the training due to hitting max steps
        if global_step >= args.device_max_steps or local_step >= max_local_steps - 1:
            optimizer.zero_grad()
            return global_step, causal_ratio
    
    optimizer.zero_grad()
    return global_step, causal_ratio


def save(model, optimizer, scheduler, global_step, epoch, args):
    checkpoint_path = f"{args.output_dir}/model.bin"
    if is_main_process():
        if os.path.exists(checkpoint_path):
            os.rename(checkpoint_path, f"{checkpoint_path}_tmp")

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        torch.save(
            model_to_save.state_dict(),
            # {
            #     "model": model_to_save.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            #     "grad_scaler": grad_scaler.state_dict(),
            #     "scheduler": scheduler.state_dict(),
            #     "global_step": global_step,
            #     "epoch": epoch,
            #     "args": args,
            # },
            checkpoint_path
        )

    return checkpoint_path


def training(model,
             train_dataloader_mlm,
             train_dataloader_causal,
             optimizer,
             scheduler,
             global_step,
             epoch,
             args,
             device,
             max_local_steps,
             causal_ratio,
             mlm_ratio):
    mlm_epoch = 0
    causal_epoch = 0

    model = model.train()
    optimizer.zero_grad(set_to_none=True)
    iterations, total_loss, total_causal_loss, total_causal_perplexity, total_causal_accuracy, total_mlm_loss, total_mlm_perplexity, total_mlm_accuracy = 0, 0, 0, 0, 0, 0, 0, 0

    train_progress_bar = tqdm(total=args.max_steps)
    train_mlm_iter = iter(train_dataloader_mlm)
    train_causal_iter = iter(train_dataloader_causal)

    for _ in range(args.max_steps*args.gradient_accumulation):
        try:
            batch_mlm = next(train_mlm_iter)
        except StopIteration:
            train_mlm_iter = iter(train_dataloader_mlm)
            batch_mlm = next(train_mlm_iter)
            mlm_epoch += 1
            save(model, optimizer, scheduler, global_step, epoch, args)

        try:
            batch_causal = next(train_causal_iter)
        except StopIteration:
            train_causal_iter = iter(train_dataloader_causal)
            batch_causal = next(train_causal_iter)
            causal_epoch += 1
            save(model, optimizer, scheduler, global_step, epoch, args)

        mlm_inputs, mlm_attention_mask, mlm_outputs = batch_mlm
        causal_inputs, causal_attention_mask, causal_outputs = batch_causal
        inputs = torch.cat([mlm_inputs, causal_inputs], dim=0).to(device, non_blocking=True)
        outputs = torch.cat([mlm_outputs, causal_outputs], dim=0).to(device, non_blocking=True)
        attention_mask = torch.cat([mlm_attention_mask, causal_attention_mask], dim=0).to(device, non_blocking=True)
        causal_pos = torch.cat([torch.zeros(args.mlm_batch_size, dtype=torch.bool), torch.ones(args.causal_batch_size, dtype=torch.bool)]).to(device, non_blocking=True)
        inputs, outputs = inputs.t(), outputs.t()
        causal_pos = causal_pos.squeeze()

        loss, causal_loss, mlm_loss, causal_perplexity, mlm_perplexity, causal_accuracy, mlm_accuracy = model(inputs, attention_mask, outputs, causal_pos, causal_ratio, args.mask_p * mlm_ratio)

        total_loss += loss.item()

        total_causal_loss += causal_loss
        total_causal_perplexity += causal_perplexity
        total_causal_accuracy += causal_accuracy

        total_mlm_loss += mlm_loss
        total_mlm_perplexity += mlm_perplexity
        total_mlm_accuracy += mlm_accuracy

        loss = loss / args.gradient_accumulation
        loss.backward()

        iterations += 1

        if iterations == args.gradient_accumulation:

            total_loss /= iterations

            total_causal_loss /= iterations
            total_causal_perplexity /= iterations
            total_causal_accuracy /= iterations

            total_mlm_loss /= iterations
            total_mlm_perplexity /= iterations
            total_mlm_accuracy /= iterations

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)

            optimizer.step(step=global_step)
            optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            global_step += 1

            if global_step % 20 == 0 and args.learned and sum(causal_pos):
                temp_loss = model(inputs, attention_mask, outputs, causal_pos, causal_ratio, only_causal=True)
                temp_loss.backward()
                causal_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 100000)
                optimizer.zero_grad(set_to_none=True)

                temp_loss = model(inputs, attention_mask, outputs, causal_pos, causal_ratio, only_mlm=True)
                temp_loss.backward()
                mlm_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 100000)
                optimizer.zero_grad(set_to_none=True)

                causal_ratio = max(0.0, mlm_grad_norm / causal_grad_norm)

            if is_main_process():
                train_progress_bar.update()
                train_progress_bar.set_postfix_str(f"loss: {total_loss:.2f}, mlm_accuracy: {total_mlm_accuracy * 100.0:.2f}, grad_norm: {grad_norm:.2f}, lr: {optimizer.param_groups[0]['lr']:.5f}")

                if global_step % 10 == 0:
                    log_parameter_histograms(model, global_step)

                wandb.log(
                    {
                        "mlm_epoch": mlm_epoch,
                        "causal_epoch": causal_epoch,
                        "train/loss": total_loss,
                        "train/causal_loss": total_causal_loss,
                        "train/causal_perplexity": total_causal_perplexity,
                        "train/causal_accuracy": total_causal_accuracy * 100.0,
                        "train/mlm_loss": total_mlm_loss,
                        "train/mlm_perplexity": total_mlm_perplexity,
                        "train/mlm_accuracy": total_mlm_accuracy * 100.0,
                        "stats/learning_rate": optimizer.param_groups[0]['lr'],
                        "stats/grad_norm": grad_norm,
                        "stats/causal_ratio": causal_ratio,
                    },
                    step=global_step,
                )

            iterations, total_loss, total_causal_loss, total_causal_perplexity, total_causal_accuracy, total_mlm_loss, total_mlm_perplexity, total_mlm_accuracy = 0, 0, 0, 0, 0, 0, 0, 0

        # Exiting the training due to hitting max steps
        if global_step >= args.device_max_steps:
            optimizer.zero_grad()
            return global_step, causal_ratio

    optimizer.zero_grad()
    return global_step, causal_ratio


def load_datasets(args, tokenizer, epoch, device):
    args.mlm_batch_size = int(args.batch_size * args.mlm_ratio)
    args.causal_batch_size = args.batch_size - args.mlm_batch_size

    train_data_mlm = DatasetMLM(
        args.input_path, 0, 1, tokenizer, args.seq_length, args.mask_p, args.short_p
    )
    train_data_causal = DatasetCausal(
        args.input_path, 0, 1, tokenizer, args.seq_length
    )
    min_length = torch.tensor(len(train_data_mlm) // args.batch_size, dtype=torch.long, device=device)

    print("Loaded training file", flush=True)

    train_dataloader_mlm = DataLoader(
        train_data_mlm,
        shuffle=True,
        batch_size=args.mlm_batch_size,
        num_workers=1,
        generator=torch.Generator().manual_seed(args.seed),
        drop_last=True,
        pin_memory=True,
    )

    train_dataloader_causal = DataLoader(
        train_data_causal,
        shuffle=True,
        batch_size=args.causal_batch_size,
        num_workers=1,
        generator=torch.Generator().manual_seed(args.seed),
        drop_last=True,
        pin_memory=True,
    )

    return train_dataloader_mlm, train_dataloader_causal, min_length


if __name__ == "__main__":
    args = parse_arguments()

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        initial_epoch, global_step = 0, 0
    else:
        checkpoint, initial_epoch, global_step = None, 0, 0

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    device, local_rank = setup_training(args)
    model, config, optimizer, scheduler = prepare_model_and_optimizer(args, device, local_rank, checkpoint)
    train_dataloader_mlm, train_dataloader_causal, min_length = load_datasets(args, tokenizer, 0, device)
    causal_ratio = 1 - args.mlm_ratio
    mlm_ratio = args.mlm_ratio
    global_step, causal_ratio = training(model, train_dataloader_mlm, train_dataloader_causal, optimizer, scheduler, global_step, 0, args, device, min_length, causal_ratio, mlm_ratio)
    save(model, optimizer, scheduler, global_step, 0, args)
