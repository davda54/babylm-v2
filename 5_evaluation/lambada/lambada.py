import torch

from tqdm import tqdm
import argparse
import json

from tokenizers import Tokenizer
# from transformers import AutoModelForMaskedLM, AutoTokenizer
from model import BertPred

from evaluator import evaluate_mlm, evaluate_mlm_shift, evaluate_causal, evaluate_prefix


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required Parameters
    parser.add_argument("--output_dir", default="lambada_results", type=str, help="The output directory where the results will be written.")
    parser.add_argument("--data", default="data/lambada.jsonl", type=str, help="Path to file containing the lambada dataset, we expect it to be in a JSONL format.")
    parser.add_argument("--model_path_or_name", default="baseline/baseline.bin", type=str, help="The local path to the model binary.")
    parser.add_argument("--tokenizer_path", default="../../tokenizer_100M.json", type=str, help="The vocabulary the model was trained on.")
    parser.add_argument("--backend", default="mlm", type=str, help="The evaluation backend strategy, options: (mlm, mlm_shift, causal, prefix).")
    parser.add_argument("--config_file", default="../../configs/base.json", type=str)

    # Optinal Parameters
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Outputs the prompt, answer and prediction of the model. Stops after num_prompts prompts.")
    parser.add_argument("--num_prompts", default=10, type=int, help="Number of verbose prompts to output. Only used when verbose is True.")

    args = parser.parse_args()

    return args


def load_config(args):
    with open(args.config_file, "r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


if __name__ == "__main__":

    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.data, "r") as f:
        new_texts = [json.loads(line) for line in f if len(line.strip()) > 0]

    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name)
    args = load_config(args)
    model = BertPred(args)

    model.load_state_dict(torch.load(args.model_path_or_name, map_location="cpu"))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    model.to(device)
    model.eval()

    verbose = args.verbose
    num_prompts = args.num_prompts

    correct_answers = 0
    total_answers = 0
    perplexity = 0.0

    progress_bar = tqdm(new_texts)

    for i, text in enumerate(progress_bar):
        answer = text["answer"]
        prompt = text["prompt"]

        match args.backend:
            case "mlm":
                prediction, loss = evaluate_mlm(prompt, answer, tokenizer, model, device, verbose)
            case "mlm_shift":
                prediction, loss = evaluate_mlm_shift(prompt, answer, tokenizer, model, device, verbose)
            case "causal":
                prediction, loss, _ = evaluate_causal(prompt, answer, tokenizer, model, device, verbose)
            case "prefix":
                prediction, loss, _ = evaluate_prefix(prompt, answer, tokenizer, model, device, verbose)
            case _:
                raise ValueError(f"The architecture cannot be {args.architecture}, it has to be one of the following: encoder, decoder, encoder_decoder.")

        perplexity += loss

        if prediction.strip() == answer.strip():
            correct_answers += 1

        total_answers += 1

        accuracy = correct_answers/total_answers * 100.0
        avg_perplexity = torch.exp(perplexity/total_answers)

        progress_bar.set_description(f"Accuracy: {accuracy:.2f}%, Perplexity: {avg_perplexity:.2f}")

        if verbose and i == num_prompts:
            break

    print(f"Accuracy: {correct_answers/total_answers * 100.0}")
    print(f"Perplexity: {torch.exp(perplexity/total_answers)}")

    if not verbose:
        with open(f"{args.output_dir}/report_{args.model_path_or_name.split('/')[-2]}_{args.backend}.txt", "w") as file:
            file.write(f"ACCURACY: {correct_answers/total_answers * 100.0}\n")
            file.write(f"PERPLEXITY: {torch.exp(perplexity/total_answers)}\n")
            file.write("\n")
