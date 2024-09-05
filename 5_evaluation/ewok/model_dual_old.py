from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import _softmax_backward_data as _softmax_backward_data

import math

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import argparse


class GeGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate: torch.Tensor

        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate, approximate='tanh')
        return x


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx: nn.Parameter, x: torch.Tensor, mask: torch.BoolTensor, dim: int) -> torch.Tensor:
        ctx.dim = dim
        x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, ctx.dim)
        x.masked_fill_(mask, 0.0)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx: nn.Parameter, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        output: torch.Tensor

        output, = ctx.saved_tensors
        inputGrad: torch.Tensor = _softmax_backward_data(grad_output, output, ctx.dim, output.dtype)
        return inputGrad, None, None


class BertPred(nn.Module):
    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__()
        self.embedding: Embedding = Embedding(config)
        self.transformer: Encoder = Encoder(config)
        self.classifier: Classifier = Classifier(config, self.embedding.word_embedding.weight)

    def get_contextualized(self, input_ids: torch.LongTensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        static_embeddings: torch.Tensor
        v_static_embeddings: torch.Tensor
        relative_embedding: torch.Tensor

        static_embeddings, v_static_embeddings, relative_embedding = self.embedding(input_ids)
        contextualized_embeddings: torch.Tensor = self.transformer(static_embeddings, v_static_embeddings, attention_mask.unsqueeze(1), relative_embedding)

        return contextualized_embeddings

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.BoolTensor, masked_lm_labels: torch.LongTensor | None = None) -> torch.Tensor:
        contextualized_embeddings: torch.Tensor = self.get_contextualized(input_ids, attention_mask)
        logits: torch.Tensor = self.classifier(contextualized_embeddings, masked_lm_labels)

        return logits


class Encoder(nn.Module):

    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__()
        self.layers: nn.ModuleList[EncoderLayer] = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

        for i, layer in enumerate(self.layers):
            layer.mlp.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

    def forward(self, hidden_states: torch.Tensor, v_hidden_states: torch.Tensor, attention_mask: torch.BoolTensor, relative_embedding: torch.Tensor) -> torch.Tensor:
        hidden_states: torch.Tensor
        v_hidden_states: torch.Tensor

        for layer in self.layers:
            hidden_states, v_hidden_states = layer(hidden_states, v_hidden_states, attention_mask, relative_embedding)

        output: torch.Tensor = torch.cat([hidden_states, v_hidden_states], dim=-1)

        return output


class EncoderLayer(nn.Module):

    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__()
        self.attention: Attention = Attention(config)
        self.mlp: FeedForward = FeedForward(config)

    def forward(self, x: torch.Tensor, vx: torch.Tensor, attention_mask: torch.BoolTensor, relative_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention: torch.Tensor
        v_attention: torch.Tensor
        mlp: torch.Tensor
        v_mlp: torch.Tensor

        attention, v_attention = self.attention(x, vx, attention_mask, relative_embedding)
        x = x + attention
        vx = vx + v_attention

        mlp, v_mlp = self.mlp(x, vx)
        x = x + mlp
        vx = vx + v_mlp

        return x, vx


class Attention(nn.Module):

    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_size: int = config.hidden_size // config.num_attention_heads
        self.position_bucket_size: int = config.position_bucket_size

        self.in_proj_qk: nn.Linear = nn.Linear(config.hidden_size, 2*config.hidden_size, bias=True)
        self.in_proj_v: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj: nn.Linear = nn.Linear(config.hidden_size, 2*config.hidden_size, bias=True)

        self.pre_layer_norm: nn.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm: nn.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)
        self.m_post_layer_norm: nn.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        self.register_position_indices(config.max_position_embeddings)

        self.dropout: nn.Dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale: float = 1.0 / math.sqrt(3 * self.head_size)
        self.initialize()

    def register_position_indices(self, max_sequence_length: int) -> None:
        position_indices: torch.LongTensor = torch.arange(max_sequence_length, dtype=torch.long).unsqueeze(1) \
            - torch.arange(max_sequence_length, dtype=torch.long).unsqueeze(0)
        position_indices = self.make_log_bucket_position(position_indices, self.position_bucket_size, max_sequence_length)
        position_indices = self.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=True)

    def make_log_bucket_position(self, relative_pos: torch.LongTensor, bucket_size: int, max_sequence_length: int) -> torch.LongTensor:
        sign: torch.Tensor = torch.sign(relative_pos)
        mid: int = bucket_size // 2
        abs_pos: torch.Tensor = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos.clamp(max=max_sequence_length-1)))
        log_pos: torch.Tensor = torch.ceil(torch.log(abs_pos / mid) / math.log((max_sequence_length-1) / mid) * (mid - 1)).int() + mid
        bucket_pos: torch.LongTensor = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()

        return bucket_pos

    def initialize(self) -> None:
        std: float = math.sqrt(2.0 / (5.0 * self.hidden_size))

        nn.init.trunc_normal_(self.in_proj_qk.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_v.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)

        self.in_proj_qk.bias.data.zero_()
        self.in_proj_v.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def create_relative_positioning(self, relative_embedding: torch.Tensor, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        relative_query_pos: torch.Tensor
        relative_key_pos: torch.Tensor

        relative_positions: torch.Tensor = self.in_proj_qk(self.dropout(relative_embedding))  # shape: [2T-1, 2D]
        relative_positions = F.embedding(self.position_indices[:sequence_length, :sequence_length], relative_positions)
        relative_query_pos, relative_key_pos = relative_positions.chunk(2, dim=-1)
        relative_query_pos = relative_query_pos.view(sequence_length, sequence_length, self.num_heads, self.head_size)
        relative_key_pos = relative_key_pos.view(sequence_length, sequence_length, self.num_heads, self.head_size)

        return relative_query_pos, relative_key_pos

    def calculate_attention_score(self, query: torch.Tensor, key: torch.Tensor, relative_query_pos: torch.Tensor, relative_key_pos: torch.Tensor, sequence_length: int, batch_size: int) -> torch.Tensor:
        query = query.reshape(sequence_length, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        key = key.reshape(sequence_length, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        attention_scores: torch.Tensor = torch.bmm(query, key.transpose(1, 2) * self.scale)

        query = query.view(batch_size, self.num_heads, sequence_length, self.head_size)
        key = key.view(batch_size, self.num_heads, sequence_length, self.head_size)
        attention_scores = attention_scores.view(batch_size, self.num_heads, sequence_length, sequence_length)
        attention_scores.add_(torch.einsum("bhqd, qkhd -> bhqk", query, relative_key_pos * self.scale))
        attention_scores.add_(torch.einsum("bhkd, qkhd -> bhqk", key * self.scale, relative_query_pos))

        return attention_scores

    def calculate_output(self, attention_probabilities: torch.Tensor, value: torch.Tensor, sequence_length: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        context: torch.Tensor
        v_context: torch.Tensor

        value = value.view(sequence_length, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        output: torch.Tensor = torch.bmm(attention_probabilities.flatten(0, 1), value)  # shape: [B*H, Q, D]
        output = output.transpose(0, 1).reshape(output.size(1), -1, self.hidden_size)  # shape: [Q, B, H*D]
        context, v_context = self.out_proj(output).chunk(2, dim=-1)
        context = self.dropout(self.post_layer_norm(context))
        v_context = self.dropout(self.m_post_layer_norm(v_context))

        return context, v_context

    def forward(self, hidden_states: torch.Tensor, v_hidden_states: torch.Tensor, attention_mask: torch.BoolTensor, relative_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length: int
        batch_size: int
        query: torch.Tensor
        key: torch.Tensor
        relative_query_pos: torch.Tensor
        relative_key_pos: torch.Tensor
        context: torch.Tensor
        v_context: torch.Tensor

        sequence_length, batch_size, _ = hidden_states.size()

        if self.position_indices.size(0) < sequence_length:
            self.register_position_indices(sequence_length)

        hidden_states = self.pre_layer_norm(hidden_states)
        v_hidden_states = self.pre_layer_norm(v_hidden_states)

        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=-1)
        value: torch.Tensor = self.in_proj_v(v_hidden_states)

        relative_query_pos, relative_key_pos = self.create_relative_positioning(relative_embedding, sequence_length)

        attention_scores: torch.Tensor = self.calculate_attention_score(query, key, relative_query_pos, relative_key_pos, sequence_length, batch_size)
        attention_probabilities: torch.Tensor = MaskedSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probabilities = self.dropout(attention_probabilities)

        context, v_context = self.calculate_output(attention_probabilities, value, sequence_length, batch_size)

        return context, v_context


class FeedForward(nn.Module):

    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__()

        self.mlp: nn.Sequential = nn.Sequential(
            nn.LayerNorm(2*config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(2*config.hidden_size, 2*config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, 2*config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size: int) -> None:
        std: float = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x: torch.Tensor, vx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joint: torch.Tensor = torch.cat([x, vx], dim=-1)
        return self.mlp(joint).chunk(2, dim=-1)


class Embedding(nn.Module):

    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__()

        self.word_embedding: nn.Embedding = nn.Embedding(config.vocab_size, 2*config.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(config.hidden_dropout_prob)
        self.scale: float = math.sqrt(config.hidden_size)

        self.relative_embedding: nn.Parameter = nn.Parameter(torch.zeros(2 * config.position_bucket_size - 1, config.hidden_size))
        self.relative_layer_norm: nn.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.initialize(config.hidden_size)

    def initialize(self, hidden_size: int) -> None:
        std: float = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        word_embedding: torch.Tensor
        v_word_embedding: torch.Tensor

        word_embedding, v_word_embedding = self.dropout(self.word_embedding(input_ids) * self.scale).chunk(2, dim=-1)
        relative_embedding: torch.Tensor = self.relative_layer_norm(self.relative_embedding)

        return word_embedding, v_word_embedding, relative_embedding


class Classifier(nn.Module):

    def __init__(self, config: argparse.Namespace, subword_embedding: torch.Tensor) -> None:
        super().__init__()
        self.nonlinearity: nn.Sequential = nn.Sequential(
            nn.LayerNorm(2*config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(2*config.hidden_size, 2*config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(2*config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(subword_embedding.size(1), subword_embedding.size(0))
        )
        self.initialize(config.hidden_size, subword_embedding)

    def initialize(self, hidden_size: int, subword_embedding: torch.Tensor) -> None:
        std: float = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[-1].weight = subword_embedding
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x: torch.Tensor, masked_lm_labels: torch.LongTensor | None = None) -> torch.Tensor:
        if masked_lm_labels is not None:
            x = torch.index_select(x.flatten(0, 1), 0, torch.nonzero(masked_lm_labels.flatten() != -100).squeeze())
        x = self.nonlinearity(x)

        return x
