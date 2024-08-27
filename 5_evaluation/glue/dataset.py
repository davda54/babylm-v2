from __future__ import annotations

import torch
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_file: str) -> None:
        self.premises = []
        self.hypotheses = []
        self.labels = []

        with open(input_file, "r") as file:
            for line in file:
                data = json.loads(line)
                self.premises.append(data["premise"])
                self.hypotheses.append(data["hypothesis"])
                self.labels.append(data["label"])

    def __len__(self) -> None:
        return len(self.premises)

    def __getitem__(self, index: int) -> None:
        premise = self.premises[index]
        hypothesis = self.hypotheses[index]
        label = self.labels[index]

        return (premise, hypothesis), label


def collate_function(tokenizer: Tokenizer, data: list[tuple[str, str] | int]) -> tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor]:

    texts = []
    labels = []

    for text, label in data:
        texts.append(text)
        labels.append(label)

    labels = torch.LongTensor(labels)
    encodings = tokenizer.encode_batch(texts)

    input_ids = []
    attention_mask = []

    for enc in encodings:
        input_ids.append(enc.ids)
        attention_mask.append(enc.attention_mask)

    input_ids = torch.LongTensor(input_ids)
    attention_mask = ~torch.BoolTensor(attention_mask)

    return input_ids, attention_mask, labels
