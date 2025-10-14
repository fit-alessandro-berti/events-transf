
import torch
import torch.nn as nn

from utils import TOKEN_TYPES


class EventEmbed(nn.Module):
    def __init__(self, num_activities: int, num_resources: int, num_dims: int, d_model: int):
        super().__init__()
        self.act_emb = nn.Embedding(num_activities, d_model)
        self.res_emb = nn.Embedding(num_resources, d_model)
        self.num_mlp = nn.Sequential(nn.Linear(num_dims, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.time_mlp = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.proj = nn.Linear(d_model * 4, d_model)

    def forward(self, act_ids, res_ids, num_feats, time_feats):
        a = self.act_emb(act_ids)
        r = self.res_emb(res_ids)
        n = self.num_mlp(num_feats)
        t = self.time_mlp(time_feats)
        x = torch.cat([a, r, n, t], dim=-1)
        return self.proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T, d = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return x + self.pos_emb(pos)


class SegmentEmbedding(nn.Module):
    def __init__(self, d_model: int, num_segments: int = 2):
        super().__init__()
        self.seg_emb = nn.Embedding(num_segments, d_model)

    def forward(self, x, segment_ids):
        return x + self.seg_emb(segment_ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att + attn_mask  # broadcast over batch & heads
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        for blk in self.layers:
            x = blk(x, attn_mask)
        return self.ln_f(x)


class SpecialTokenEmbedding(nn.Module):
    def __init__(self, n_special: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(n_special, d_model)
    def forward(self, special_ids):
        return self.emb(special_ids)


class ClassLabelEmbedding(nn.Module):
    def __init__(self, n_classes: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(n_classes, d_model)
    def forward(self, class_ids):
        return self.emb(class_ids)


class NumericLabelEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
    def forward(self, values):
        return self.mlp(values.unsqueeze(-1))
