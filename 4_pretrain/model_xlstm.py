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

        static_embeddings = self.embedding(x)
        contextualized_embeddings = self.transformer(static_embeddings, x_mask.unsqueeze(1))

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
            layer.attention.in_proj.weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

        self.activation_checkpointing = activation_checkpointing

    def forward(self, hidden_states, attention_mask):
        memory = hidden_states
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
        self.attention = AttentionGLU(config)

    def forward(self, x, padding_mask):
        return x + self.attention(x, padding_mask)


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


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=512, base=1_000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def swish(x):
    return x * x.sigmoid()


class BlockDiagonalLinear(nn.Module):
    def __init__(self, in_features, out_features, num_blocks, bias=True):
        super().__init__()
        self.in_features_block_size = in_features // num_blocks
        self.out_features_block_size = out_features // num_blocks
        self.num_blocks = num_blocks
        self.weight = nn.Parameter(torch.Tensor(num_blocks, self.out_features_block_size, self.in_features_block_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_blocks, self.out_features_block_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()
    
    def forward(self, x):
        T, B, D = x.shape
        x = x.view(T, B, self.num_blocks, self.in_features_block_size)
        x = torch.einsum('tbhi,hoi->tbho', x, self.weight)
        x = x.flatten(2, 3)
        if self.bias is not None:
            x = x + self.bias.flatten(0, 1)
        return x


class AttentionGLU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.intermediate_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.intermediate_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.position_bucket_size = config.position_bucket_size
        self.num_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.head_size = config.intermediate_size // config.num_attention_heads

        self.in_proj = nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=False)
        self.out_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.query_proj = BlockDiagonalLinear(config.intermediate_size, config.intermediate_size, config.intermediate_size // 4, bias=True)
        self.key_proj = BlockDiagonalLinear(config.intermediate_size, config.intermediate_size, config.intermediate_size // 4, bias=False)
        self.value_proj = BlockDiagonalLinear(config.intermediate_size, config.intermediate_size, config.intermediate_size // 4, bias=False)
        self.time_conv = nn.Conv1d(config.intermediate_size, config.intermediate_size, kernel_size=5, padding=2, groups=config.intermediate_size)

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm = nn.LayerNorm(self.intermediate_size, config.layer_norm_eps, elementwise_affine=False)
        self.key_layer_norm = nn.LayerNorm(self.head_size, config.layer_norm_eps, elementwise_affine=True)
        self.query_layer_norm = nn.LayerNorm(self.head_size, config.layer_norm_eps, elementwise_affine=True)
        self.l_skip = nn.Parameter(torch.ones(config.intermediate_size, dtype=torch.float32))
        self.rotary_emb = RotaryEmbedding(self.head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.scale = 1.0 / math.sqrt(self.head_size)
        self.initialize(config.hidden_size, self.head_size)


    def initialize(self, hidden_size, head_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.in_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x, attention_mask):
        key_len, batch_size, _ = x.size()
        query_len = key_len

        x = self.pre_layer_norm(x)  # shape: [T, B, D]
        x, gate = self.in_proj(x).chunk(2, dim=-1)  # shape: 2 x [T, B, I]
        
        gate = swish(gate)  # shape: [T, B, I]

        value = self.value_proj(x)
        value = value.view(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        x = self.time_conv(x.permute(1, 2, 0)).permute(2, 0, 1)  # shape: [T, B, I]
        x = swish(x)
        key = self.key_proj(x).view(key_len, batch_size, self.num_heads, self.head_size)
        key = self.key_layer_norm(key).view(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        query = self.query_proj(x).view(key_len, batch_size, self.num_heads, self.head_size)
        query = self.query_layer_norm(query).view(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        cos, sin = self.rotary_emb(value, torch.arange(query_len, device=x.device).expand(batch_size * self.num_heads, -1))
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)
        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)

        value = torch.bmm(attention_probs.flatten(0, 1), value)  # shape: [B*H, Q, D]
        value = value.transpose(0, 1).reshape(value.size(1), -1, self.intermediate_size)  # shape: [Q, B, H*D]

        x = x * self.l_skip + value
        x = x * gate
        x = self.post_layer_norm(x)
        x = self.out_proj(x)
        x = self.out_dropout(x)

        return x


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