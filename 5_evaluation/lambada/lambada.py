import torch

from tqdm import tqdm
import argparse
import json

from tokenizers import Tokenizer
from transformers import AutoModelForMaskedLM

from evaluator import evaluate_mlm, evaluate_mlm_shift, evaluate_causal, evaluate_prefix


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required Parameters
    parser.add_argument("--data", default="../../data/lambada/lambada.jsonl", type=str, help="Path to file containing the lambada dataset, we expect it to be in a JSONL format.")
    parser.add_argument("--model_path_or_name", default=None, type=str, help="The local path to the model binary.")
    parser.add_argument("--tokenizer_path", default="../tokenizers/baseline_tokenizer.json", type=str, help="The vocabulary the model was trained on.")
    parser.add_argument("--backend", default="mlm", type=str, help="The evaluation backend strategy, options: (mlm, mlm_shift, causal, prefix).")

    # Optinal Parameters
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Outputs the prompt, answer and prediction of the model. Stops after num_prompts prompts.")
    parser.add_argument("--num_prompts", default=10, type=int, help="Number of verbose prompts to output. Only used when verbose is True.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_arguments()

    with open(args.data, "r") as f:
        new_texts = [json.loads(line) for line in f if len(line.strip()) > 0]

    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name)

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
                prediction, loss = evaluate_mlm(prompt, answer, tokenizer, model, verbose)
            case "mlm_shift":
                prediction, loss = evaluate_mlm_shift(prompt, answer, tokenizer, model, verbose)
            case "causal":
                prediction, loss, _ = evaluate_causal(prompt, answer, tokenizer, model, verbose)
            case "prefix":
                prediction, loss, _ = evaluate_prefix(prompt, answer, tokenizer, model, verbose)
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

    # TODO: Write to file