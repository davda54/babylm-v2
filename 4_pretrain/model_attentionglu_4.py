import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _softmax_backward_data as _softmax_backward_data
from torch.utils import checkpoint


class Bert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.transformer = Encoder(config)
        self.classifier = MaskClassifier(config, self.embedding.word_embedding.weight)

    def forward(self, x, x_mask, masked_lm_labels=None):

        static_embeddings, relative_embedding = self.embedding(x)
        contextualized_embeddings = self.transformer(static_embeddings, x_mask.unsqueeze(1), relative_embedding)

        subword_prediction = self.classifier(contextualized_embeddings, masked_lm_labels)

        gold_labels = masked_lm_labels.flatten()

        gold_labels = gold_labels[gold_labels != -100]

        loss = F.cross_entropy(subword_prediction, gold_labels)
        z_loss = torch.logsumexp(subword_prediction, dim=-1).pow(2).mean()

        with torch.no_grad():
            accuracy = (subword_prediction.argmax(-1) == gold_labels).float().mean()

        num_tokens = gold_labels.size(0)

        return loss, accuracy, z_loss, num_tokens

class Encoder(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

        for i, layer in enumerate(self.layers):
            layer.attention.out_proj.weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.attention.value_projection.weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

        self.activation_checkpointing = activation_checkpointing

    def forward(self, hidden_states, attention_mask, relative_embedding):
        memory = hidden_states
        for layer in self.layers:
            if self.activation_checkpointing:
                hidden_states = checkpoint.checkpoint(layer, hidden_states, attention_mask, relative_embedding)
            else:
                hidden_states = layer(hidden_states, attention_mask, relative_embedding)

        return hidden_states

class MaskClassifier(nn.Module):
    def __init__(self, config, subword_embedding):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(subword_embedding.size(1), subword_embedding.size(0))
        )
        self.initialize(config.hidden_size, subword_embedding)

    def initialize(self, hidden_size, embedding):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[-1].weight = embedding
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x, masked_lm_labels=None):
        if masked_lm_labels is not None:
            x = torch.index_select(x.flatten(0, 1), 0, torch.nonzero(masked_lm_labels.flatten() != -100).squeeze())
        x = self.nonlinearity(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AttentionGLU(config)

    def forward(self, x, padding_mask, relative_embedding):
        return x + self.attention(x, padding_mask, relative_embedding)


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self, x, mask, dim):
        self.dim = dim
        x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, self.dim)
        x.masked_fill_(mask, 0.0)
        self.save_for_backward(x)
        return x

    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output.dtype)
        return inputGrad, None, None


class AttentionGLU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.position_bucket_size = config.position_bucket_size
        self.num_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.v_head_size = config.intermediate_size // config.num_attention_heads
        self.qk_head_size = config.hidden_size // config.num_attention_heads

        self.value_projection = nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=False)
        self.in_proj_qk = nn.Linear(config.hidden_size, 2*config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.l_skip = nn.Parameter(torch.zeros(config.intermediate_size, dtype=torch.float32))

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm = nn.LayerNorm(config.intermediate_size, config.layer_norm_eps, elementwise_affine=False)

        position_indices = torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(1) \
            - torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0)
        position_indices = self.make_log_bucket_position(position_indices, config.position_bucket_size, config.max_position_embeddings)
        position_indices = config.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.qk_head_size)
        self.initialize(config.hidden_size)

    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos).clamp(max=max_position - 1))
        log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position-1) / mid) * (mid - 1)).int() + mid
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        return bucket_pos

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.value_projection.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_qk.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.in_proj_qk.bias.data.zero_()

    def forward(self, hidden_states, attention_mask, relative_embedding):
        key_len, batch_size, _ = hidden_states.size()
        query_len = key_len

        if self.position_indices.size(0) < query_len:
            position_indices = torch.arange(query_len, dtype=torch.long).unsqueeze(1) \
                - torch.arange(query_len, dtype=torch.long).unsqueeze(0)
            position_indices = self.make_log_bucket_position(position_indices, self.position_bucket_size, 512)
            position_indices = self.position_bucket_size - 1 + position_indices
            self.register_buffer("position_indices", position_indices.to(hidden_states.device), persistent=True)

        hidden_states = self.pre_layer_norm(hidden_states)

        value, gate = self.value_projection(hidden_states).chunk(2, dim=-1)
        gate = F.gelu(gate)
        value_skip = F.gelu(value)
        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)  # shape: [T, B, D]

        pos = self.in_proj_qk(self.dropout(relative_embedding))  # shape: [2T-1, 2D]
        pos = F.embedding(self.position_indices[:query_len, :key_len], pos)  # shape: [T, T, 2D]
        query_pos, key_pos = pos.chunk(2, dim=-1)
        query_pos = query_pos.view(query_len, key_len, self.num_heads, self.qk_head_size)
        key_pos = key_pos.view(query_len, key_len, self.num_heads, self.qk_head_size)

        query = query.reshape(query_len, batch_size * self.num_heads, self.qk_head_size).transpose(0, 1)
        key = key.reshape(key_len, batch_size * self.num_heads, self.qk_head_size).transpose(0, 1)
        value = value.reshape(key_len, batch_size * self.num_heads, self.v_head_size).transpose(0, 1)

        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)

        query = query.view(batch_size, self.num_heads, query_len, self.qk_head_size)
        key = key.view(batch_size, self.num_heads, query_len, self.qk_head_size)
        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        attention_scores.add_(torch.einsum("bhqd,qkhd->bhqk", query, key_pos * self.scale))
        attention_scores.add_(torch.einsum("bhkd,qkhd->bhqk", key * self.scale, query_pos))

        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)

        attention_probs = self.dropout(attention_probs)
        context = torch.bmm(attention_probs.flatten(0, 1), value)  # shape: [B*H, Q, D]
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.intermediate_size)  # shape: [Q, B, H*D]
        context = context + F.sigmoid(self.l_skip) * value_skip
        context = context * gate
        context = self.post_layer_norm(context)
        context = self.out_proj(context)
        context = self.out_dropout(context)

        return context

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.hidden_size))
        self.relative_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids):
        word_embedding = self.dropout(self.word_layer_norm(self.word_embedding(input_ids)))
        relative_embeddings = self.relative_layer_norm(self.relative_embedding)
        return word_embedding, relative_embeddings