# takes in the input directory, output directory, path to the tokenizer, and the max sequence length
# the input directory is the directory containing N sharded jsonl files
# the output directory is the directory where the each file is tokenized

from tokenizers import Tokenizer
import json
import os
import argparse
import re
from smart_open import open
import time
import torch
import gzip
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train_path', type=str, default="../data/train_100M.jsonl")
    parser.add_argument('--input_valid_path', type=str, default="../data/dev.jsonl")
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer_100M.json")
    return parser.parse_args()


def tokenize_text(tokenizer, text):
    text = text.strip()
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    ids = torch.tensor(ids, dtype=torch.int16)
    return ids


def tokenize_file(input_filename, output_filename, tokenizer):
    tokenized_documents = []
    n_subwords = 0

    for i, line in enumerate(tqdm(open(input_filename, 'rt'))):
        document = json.loads(line)
        tokenized_document = tokenize_text(tokenizer, document)
        tokenized_documents.append(tokenized_document)
        n_subwords += len(tokenized_document)

        if i == 0:
            print("Example tokenized document:")
            print(document)
            for token in tokenized_document:
                print(tokenizer.id_to_token(token), end=" ")
            print(flush=True)
    
    torch.save(tokenized_documents, output_filename)
    print(f"Tokenized {len(tokenized_documents)} documents with {n_subwords} subwords in total")


if __name__ == "__main__":
    args = parse_args()

    # load the tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    input_train_path = args.input_train_path
    input_valid_path = args.input_valid_path
    output_train_path = input_train_path.replace(".jsonl", "_tokenized.bin")
    output_valid_path = input_valid_path.replace(".jsonl", "100M_tokenized.bin")

    tokenize_file(input_train_path, output_train_path, tokenizer)
    tokenize_file(input_valid_path, output_valid_path, tokenizer)
