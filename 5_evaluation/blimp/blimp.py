# coding=utf-8

import argparse
import torch
import tqdm
import json
from collections import Counter
from transformers import AutoTokenizer, AutoModelForMaskedLM
# import wandb
import os
# from tqdm import tqdm

from lm_score import rank_mlm, rank_causal, rank_mlm_shift, rank_fused
# from tokenizers import Tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_path", default="data", type=str, help="Path to BLiMP.")
    # parser.add_argument("--name", default="blimp", type=str)
    parser.add_argument("--output_dir", default="blimp_results", type=str, help="The output directory where the model checkpoints will be written.")
    # parser.add_argument("--tokenizer_path", default="../tokenizer_wiki.json", type=str, help="The vocabulary the BERT model will train on.")
    parser.add_argument("--model_path_or_name", default="bert-base-cased", type=str, help="Path to a previous checkpointed training state.")
    parser.add_argument("--backend", default="mlm", type="str", help="The evaluation backend strategy, options: (mlm, causal, mlm_shift, fused)")
    parser.add_argument("--batch_size", default=64, type=int)

    args = parser.parse_args()

    return args


@torch.no_grad()
def evaluate(model, tokenizer, device, args):
    temperatures = torch.arange(0.0, 3.05, 0.05, device=device).clamp(min=1e-6)

    field_count = {"correct": [Counter() for _ in range(temperatures.size(0))], "total": [Counter() for _ in range(temperatures.size(0))]}
    uid_count = {"correct": [Counter() for _ in range(temperatures.size(0))], "total": [Counter() for _ in range(temperatures.size(0))]}
    linguistics_term_count = {"correct": [Counter() for _ in range(temperatures.size(0))], "total": [Counter() for _ in range(temperatures.size(0))]}

    # iterate through all .jsonl files in ./data/ directory
    for filename in os.listdir(args.input_path):
        if not filename.endswith(".jsonl"):
            continue

        # open file
        with open(os.path.join(args.input_path, filename), "r") as file:
            # iterate through each line in file
            for line in tqdm(file):
                # parse line
                line = json.loads(line.strip())

                # add to pairs
                pair = {
                    "good": line["sentence_good"],
                    "bad": line["sentence_bad"],
                    "field": line["field"],
                    "UID": line["UID"],
                    "linguistics_term": line["linguistics_term"]
                }
                if pair["field"] == "syntax_semantics":
                    pair["field"] = "syntax/semantics"

                # rank
                if args.backend == "mlm":
                    _, finegrained_ranking = rank_mlm([pair["good"], pair["bad"]], model, tokenizer, device, args.batch_size, temperatures=temperatures)
                elif args.backend == "causal":
                    _, finegrained_ranking = rank_causal([pair["good"], pair["bad"]], model, tokenizer, device, args.batch_size, temperatures=temperatures)
                elif args.backend == "mlm_shit":
                    _, finegrained_ranking = rank_mlm_shift([pair["good"], pair["bad"]], model, tokenizer, device, args.batch_size, temperatures=temperatures)
                elif args.backend == "fused":
                    _, finegrained_ranking = rank_fused([pair["good"], pair["bad"]], model, tokenizer, device, args.batch_size, temperatures=temperatures)
                else:
                    raise ValueError(f"Backend {args.backend} is not implemented!")

                for i, ranking in enumerate(finegrained_ranking):
                    if ranking[0] == 0:
                        field_count["correct"][i][pair["field"]] += 1
                        uid_count["correct"][i][pair["UID"]] += 1
                        linguistics_term_count["correct"][i][pair["linguistics_term"]] += 1
                    field_count["total"][i][pair["field"]] += 1
                    uid_count["total"][i][pair["UID"]] += 1
                    linguistics_term_count["total"][i][pair["linguistics_term"]] += 1

            print(f'Accuracy of {pair["UID"]} at temperature 1 is: {uid_count["correct"][20][pair["UID"]] / uid_count["total"][20][pair["UID"]] * 100:.2f}')

    # compute accuracy

    field_accuracy = [{key: field_count["correct"][i][key] / field_count["total"][i][key] * 100.0 for key in field_count["correct"][i].keys()} for i in range(len(finegrained_ranking))]
    uid_accuracy = [{key: uid_count["correct"][i][key] / uid_count["total"][i][key] * 100.0 for key in uid_count["correct"][i].keys()} for i in range(len(finegrained_ranking))]
    linguistics_term_accuracy = [{key: linguistics_term_count["correct"][i][key] / linguistics_term_count["total"][i][key] * 100.0 for key in linguistics_term_count["correct"][i].keys()} for i in range(len(finegrained_ranking))]

    average_accuracies = [sum(uid_accuracy[i].values()) / len(uid_accuracy[i].values()) for i in range(len(finegrained_ranking))]

    for temperature, acc in zip(temperatures.tolist(), average_accuracies):
        print(f"{temperature}\t{acc:.2f}")
    print()

    average_accuracies = torch.tensor(average_accuracies)
    max_temp = torch.argmax(average_accuracies)
    print(f"BEST TEMPERATURE: {max_temp * 0.05}")
    print()

    # print
    print("### FIELD ACCURACY")
    for key in field_accuracy[max_temp].keys():
        print(f"{key}: {field_accuracy[max_temp][key]:.2f}")
    print()

    print("### LINGUISTIC TERM ACCURACY")
    for key in linguistics_term_accuracy[max_temp].keys():
        print(f"{key}: {linguistics_term_accuracy[max_temp][key]:.2f}")
    print()

    print("### UID ACCURACY")
    for key in uid_accuracy[max_temp].keys():
        print(f"{key}: {uid_accuracy[max_temp][key]:.2f}")
    print()

    print("### AVERAGE ACCURACY")
    print(f"{average_accuracies[max_temp]:.2f}")
    print()

    # save report
    with open(f"{args.output_dir}/report_{args.model_path_or_name.split('/')[-1]}.txt", "w") as file:
        file.write("### BEST TEMPERATURE\n")
        file.write(f"{max_temp * 0.05:.2f}\n")

        file.write("### FIELD ACCURACY\n")
        for key in field_accuracy[max_temp].keys():
            file.write(f"{key}: {field_accuracy[max_temp][key]:.2f}\n")
        file.write("\n")

        file.write("### LINGUISTIC TERM ACCURACY\n")
        for key in linguistics_term_accuracy[max_temp].keys():
            file.write(f"{key}: {linguistics_term_accuracy[max_temp][key]:.2f}\n")
        file.write("\n")

        file.write("### UID ACCURACY\n")
        for key in uid_accuracy[max_temp].keys():
            file.write(f"{key}: {uid_accuracy[max_temp][key]:.2f}\n")
        file.write("\n")

        file.write("### AVERAGE ACCURACY\n")
        file.write(f"{average_accuracies[max_temp]:.2f}\n")
        file.write("\n")


if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, trust_remote_code=True)

    model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name, trust_remote_code=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    model.to(device)
    model.eval()

    evaluate(model, tokenizer, device, args)
