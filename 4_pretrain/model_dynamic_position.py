import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _softmax_backward_data as _softmax_backward_data
from torch.utils import checkpoint


class Bert(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.embedding = Embedding(config)
        self.transformer = Encoder(config, activation_checkpointing)
        self.classifier = MaskClassifier(config, self.embedding.word_embedding.weight)

    def get_contextualized(self, input_ids, attention_mask):
        static_embeddings = self.embedding(input_ids)
        contextualized_embeddings = self.transformer(static_embeddings, attention_mask.unsqueeze(1))
        return contextualized_embeddings

    def forward(self, input_ids, attention_mask, masked_lm_labels=None):
        contextualized_embeddings = self.get_contextualized(input_ids, attention_mask)
        subword_prediction = self.classifier(contextualized_embeddings, masked_lm_labels)

        gold_labels = masked_lm_labels.flatten()
        gold_labels = gold_labels[gold_labels != -100]

        loss = F.cross_entropy(subword_prediction, gold_labels, reduction="none").mean()
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
            layer.mlp.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

        self.activation_checkpointing = activation_checkpointing

    def forward(self, hidden_states, attention_mask):
        for layer in self.layers:
            if self.activation_checkpointing:
                hidden_states = checkpoint.checkpoint(layer, hidden_states, attention_mask)
            else:
                hidden_states = layer(hidden_states, attention_mask)

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
        self.attention = Attention(config)
        self.mlp = FeedForward(config)

    def forward(self, x, padding_mask):
        x = x + self.attention(x, padding_mask)
        x = x + self.mlp(x)
        return x


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate, approximate='tanh')
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return self.mlp(x)


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
    

class DynamicPosition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_projection = nn.Linear(config.hidden_size, config.num_attention_heads, bias=True)
        self.q_relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.num_attention_heads, config.hidden_size // config.num_attention_heads))
        self.k_relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.num_attention_heads, config.hidden_size // config.num_attention_heads))

        self.bucket_size = config.position_bucket_size
        self.max_position = config.max_position_embeddings
        self.scale = 1.0 / math.sqrt(3 * config.hidden_size / config.num_attention_heads)

        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.q_relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.k_relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.pos_projection.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.pos_projection.bias.data.fill_(0.0)

    def forward(self, hidden_states, query, key):
        floor_buckets, ceil_buckets, delta_weight = self.create_position_buckets(hidden_states)  # shape: [B, H, T_q, T_k]

        query_pos_bias = torch.einsum("lhd,qbhd->bhql", self.k_relative_embedding * self.scale, query)  # shape: [B, H, T_q, L]
        key_pos_bias = torch.einsum("lhd,kbhd->bhlk", self.q_relative_embedding, key * self.scale)  # shape: [B, H, L, T_k]

        floor_query_pos_bias = torch.gather(query_pos_bias, 3, floor_buckets)  # shape: [B, H, T_q, T_k]
        ceil_query_pos_bias = torch.gather(query_pos_bias, 3, ceil_buckets)  # shape: [B, H, T_q, T_k]
        floor_key_pos_bias = torch.gather(key_pos_bias, 2, floor_buckets)  # shape: [B, H, T_q, T_k]
        ceil_key_pos_bias = torch.gather(key_pos_bias, 2, ceil_buckets)  # shape: [B, H, T_q, T_k]

        query_pos_bias = floor_query_pos_bias + delta_weight * (ceil_query_pos_bias - floor_query_pos_bias)  # shape: [B, H, T_q, T_k]
        key_pos_bias = floor_key_pos_bias + delta_weight * (ceil_key_pos_bias - floor_key_pos_bias)  # shape: [B, H, T_q, T_k]

        return query_pos_bias, key_pos_bias
        
    def create_position_buckets(self, hidden_states):
        soft_positions = self.pos_projection(hidden_states).sigmoid()  # shape: [T, B, H]
        soft_positions = soft_positions * 1.2 - 0.1  # shape: [T, B, H]
        soft_positions = soft_positions.permute(1, 2, 0)  # shape: [B, H, T]
        cum_soft_positions = soft_positions.cumsum(dim=-1)  # shape: [B, H, T]

        soft_position_indices = cum_soft_positions.unsqueeze(2) - cum_soft_positions.unsqueeze(3)  # shape: [B, H, T, T]
        soft_position_indices = soft_position_indices + self.static_position_bias(hidden_states)  # shape: [B, H, T, T]
        floor_position_indices = soft_position_indices.floor().long()  # shape: [B, H, T, T]
        ceil_position_indices = soft_position_indices.ceil().long()  # shape: [B, H, T, T]
        delta_position_indices = soft_position_indices - floor_position_indices  # shape: [B, H, T, T]

        floor_position_buckets = self.make_log_bucket_position(floor_position_indices)  # shape: [B, H, T, T]
        ceil_position_buckets = self.make_log_bucket_position(ceil_position_indices)  # shape: [B, H, T, T]

        return floor_position_buckets, ceil_position_buckets, delta_position_indices

    def static_position_bias(self, hidden_states):
        tril = torch.tril(torch.ones(hidden_states.size(0), hidden_states.size(0), device=hidden_states.device), diagonal=-1)  # shape: [T, T]
        triu = torch.triu(torch.ones(hidden_states.size(0), hidden_states.size(0), device=hidden_states.device), diagonal=1)  # shape: [T, T]
        position_bias = triu - tril  # shape: [T, T]
        position_bias = position_bias.view(1, 1, hidden_states.size(0), hidden_states.size(0))  # shape: [1, 1, T, T]
        return position_bias
    
    def make_log_bucket_position(self, relative_pos):
        sign = torch.sign(relative_pos)
        mid = self.bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos).clamp(max=self.max_position - 1))
        log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((self.max_position - 1) / mid) * (mid - 1)).int() + mid
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        bucket_pos = self.bucket_size - 1 + bucket_pos
        return bucket_pos

        
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.in_proj_qk = nn.Linear(config.hidden_size, 2*config.hidden_size, bias=True)
        self.in_proj_v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.q_layer_norm = nn.LayerNorm(self.head_size, config.layer_norm_eps, elementwise_affine=True)
        self.k_layer_norm = nn.LayerNorm(self.head_size, config.layer_norm_eps, elementwise_affine=True)
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        self.dynamic_position = DynamicPosition(config)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)
        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.in_proj_qk.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_v.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.in_proj_qk.bias.data.zero_()
        self.in_proj_v.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def forward(self, hidden_states, attention_mask):
        key_len, batch_size, _ = hidden_states.size()
        query_len = key_len

        hidden_states = self.pre_layer_norm(hidden_states)
        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)  # shape: [T, B, D]
        value = self.in_proj_v(hidden_states)  # shape: [T, B, D]

        query = self.q_layer_norm(query.reshape(query_len, batch_size, self.num_heads, self.head_size))
        key = self.k_layer_norm(key.reshape(key_len, batch_size, self.num_heads, self.head_size))
        value = value.view(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        query_pos, key_pos = self.dynamic_position(hidden_states, query, key)

        query = query.flatten(1, 2).transpose(0, 1)
        key = key.flatten(1, 2).transpose(0, 1)
        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)

        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        attention_scores.add_(query_pos + key_pos)

        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)

        attention_probs = self.dropout(attention_probs)
        context = torch.bmm(attention_probs.flatten(0, 1), value)  # shape: [B*H, Q, D]
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.hidden_size)  # shape: [Q, B, H*D]
        context = self.out_proj(context)
        context = self.post_layer_norm(context)
        context = self.dropout(context)

        return context


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids):
        word_embedding = self.dropout(self.word_layer_norm(self.word_embedding(input_ids)))
        return word_embedding
