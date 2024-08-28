from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


class ClassifierHead(nn.Module):

    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__()
        self.nonlinearity: nn.Sequential = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.num_labels)
        )

    def forward(self, eembeddings: torch.Tensor) -> torch.Tensor:
        return self.nonlinearity(eembeddings)


def import_architecture(architecture):
    match architecture:
        case "base":
            from model import Bert
        case "attglu":
            from model_attentionglu_2 import Bert
        case "attgate":
            from model_attention_gate import Bert
        case "densemod":
            from model_denseformer_module import Bert
        case "densesubmod":
            from model_denseformer import Bert
        case "densecont":
            from model_denseformer_2 import Bert
        case "elc":
            from model_elc import Bert
        case "qkln":
            from model_qk_layernorm import Bert
        case _:
            raise ValueError(f"The architecture cannot be {architecture}, it has to be one of the following: base, attglu, attgate, densemod, densesubmod, densecont, elc, qkln.")

    return Bert


class ModelForSequenceClassification(nn.Module):

    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__()
        self.transformer: nn.Module = import_architecture(config.architecture)(config)
        self.classifier: nn.Module = ClassifierHead(config)

    def forward(self, input_data: torch.LongTensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        head_embedding = self.transformer.get_contextualized(input_data.t(), attention_mask.unsqueeze(1))[0]
        logits = self.classifier(head_embedding)

        return logits
