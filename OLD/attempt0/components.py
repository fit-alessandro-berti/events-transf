# components.py

import torch
import torch.nn as nn
import math
from data_generator import SPECIAL_TOKENS


class EventEmbedder(nn.Module):
    """
    Token embeddings with rich event features.

    For <EVENT> tokens, we ADD (not replace) a learned projection of
    (categorical + numeric + time) features to the base token embedding.
    We also add:
      - type embeddings (event/label/query/task/etc.)
      - case-id embeddings computed from <CASE_SEP> markers

    Numeric/time streams are LayerNorm'ed before their MLPs for stability.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        cat_cardinalities,
        num_num_features: int,
        num_time_features: int,
        d_cat: int = 64,
        dropout: float = 0.1,
        max_cases: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_num_features = num_num_features
        self.num_time_features = num_time_features
        self.num_cat = len(cat_cardinalities)
        self.d_cat = d_cat
        self.max_cases = max_cases

        # Token embedding for all tokens (specials + class tokens)
        self.token_embed = nn.Embedding(
            vocab_size, d_model, padding_idx=SPECIAL_TOKENS['<PAD>']
        )

        # Compact categorical embeddings per categorical field
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(card, d_cat) for card in cat_cardinalities]
        )

        # Normalization before MLPs to stabilize scale
        self.num_norm = nn.LayerNorm(num_num_features)
        self.time_norm = nn.LayerNorm(num_time_features)

        # Numeric + time MLPs (to half-width)
        hid = max(32, d_model // 2)
        self.num_mlp = nn.Sequential(
            nn.Linear(num_num_features, hid), nn.GELU(), nn.Linear(hid, hid)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(num_time_features, hid), nn.GELU(), nn.Linear(hid, hid)
        )

        # Project concatenated [cats || nums || time] -> d_model, then normalize
        concat_dim = self.num_cat * d_cat + hid + hid
        self.event_proj = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Token type embeddings:
        # 0=other/pad, 1=EVENT, 2=LABEL (the <LABEL> marker), 3=QUERY,
        # 4=TASK, 5=CASE_SEP, 6=LABEL_VALUE (token AFTER <LABEL> in support)
        self.type_embed = nn.Embedding(7, d_model)

        # Case id embedding (computed from cumsum of <CASE_SEP>)
        self.case_embed = nn.Embedding(self.max_cases, d_model)

        # Learnable scales to balance contributions
        self.event_scale = nn.Parameter(torch.tensor(1.0))
        self.type_scale = nn.Parameter(torch.tensor(1.0))
        self.case_scale = nn.Parameter(torch.tensor(1.0))

        self.dropout = nn.Dropout(dropout)

    def _build_type_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        type_ids = torch.zeros_like(token_ids)

        is_event = token_ids == SPECIAL_TOKENS['<EVENT>']
        is_label = token_ids == SPECIAL_TOKENS['<LABEL>']
        is_query = token_ids == SPECIAL_TOKENS['<QUERY>']
        is_task = (token_ids == SPECIAL_TOKENS['<TASK_NEXT_ACTIVITY>']) | (
            token_ids == SPECIAL_TOKENS['<TASK_REMAINING_TIME>']
        )
        is_case_sep = token_ids == SPECIAL_TOKENS['<CASE_SEP>']
        is_label_value = token_ids >= len(SPECIAL_TOKENS)

        type_ids = torch.where(is_event, torch.full_like(type_ids, 1), type_ids)
        type_ids = torch.where(is_label, torch.full_like(type_ids, 2), type_ids)
        type_ids = torch.where(is_query, torch.full_like(type_ids, 3), type_ids)
        type_ids = torch.where(is_task, torch.full_like(type_ids, 4), type_ids)
        type_ids = torch.where(is_case_sep, torch.full_like(type_ids, 5), type_ids)
        type_ids = torch.where(is_label_value, torch.full_like(type_ids, 6), type_ids)

        return type_ids

    def _compute_case_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Case ids are computed as the cumulative count of <CASE_SEP> tokens.
        The initial tokens (before the first <CASE_SEP>) have case_id = 0,
        first case after that has id=1, etc. Clamped to [0, max_cases-1].
        """
        B, T = token_ids.shape
        is_sep = (token_ids == SPECIAL_TOKENS['<CASE_SEP>']).to(torch.long)  # [B,T]
        case_ids = torch.cumsum(is_sep, dim=1)  # increments at each separator
        case_ids = case_ids.clamp(max=self.max_cases - 1)
        return case_ids  # [B,T] ints in [0, max_cases-1]

    def forward(self, token_ids, cat_feats, num_feats, time_feats):
        """
        token_ids: [B, T]
        cat_feats: [B, T, num_cat]
        num_feats: [B, T, num_num_features]
        time_feats:[B, T, num_time_features]
        """
        B, T = token_ids.shape

        # Base token embeddings for all tokens
        base_emb = self.token_embed(token_ids)  # [B, T, d_model]

        # Event feature embedding for <EVENT> positions
        if self.num_cat > 0:
            cat_parts = [emb(cat_feats[:, :, i]) for i, emb in enumerate(self.cat_embs)]
            cat_concat = torch.cat(cat_parts, dim=-1)  # [B, T, num_cat*d_cat]
        else:
            cat_concat = torch.zeros(B, T, 0, device=token_ids.device)

        # Normalize numeric/time features before MLPs for stability
        num_emb = self.num_mlp(self.num_norm(num_feats))     # [B, T, hid]
        time_emb = self.time_mlp(self.time_norm(time_feats)) # [B, T, hid]

        event_feature_emb = self.event_proj(torch.cat([cat_concat, num_emb, time_emb], dim=-1))

        # ADD event features at <EVENT> positions (gated), don't replace
        is_event = (token_ids == SPECIAL_TOKENS['<EVENT>']).unsqueeze(-1).float()  # [B,T,1]
        mixed = base_emb + self.event_scale * is_event * event_feature_emb

        # Add type embeddings to every token (scaled)
        type_ids = self._build_type_ids(token_ids)
        mixed = mixed + self.type_scale * self.type_embed(type_ids)

        # Add case-id embeddings (computed from <CASE_SEP>)
        case_ids = self._compute_case_ids(token_ids)  # [B,T]
        mixed = mixed + self.case_scale * self.case_embed(case_ids)

        return self.dropout(mixed)  # [B, T, d_model]


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encodings with dropout.
    """
    def __init__(self, d_model, dropout=0.1, max_len=8192):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)         # [max_len,1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)                 # [max_len,1,d_model]
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        # x: [B, T, d_model]
        T = x.size(1)
        x = x + self.pe[:T].transpose(0, 1)  # [1,T,D] broadcast to [B,T,D]
        return self.dropout(x)


class DecoderOnlyTransformer(nn.Module):
    """
    Causal Transformer implemented via TransformerEncoder + causal mask.
    Uses norm_first=True and a final LayerNorm for stability.
    """
    def __init__(self, d_model, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, src_emb, attn_mask, padding_mask):
        """
        src_emb: [B,T,D]
        attn_mask: [T,T] boolean mask (True blocks attention)
        padding_mask: [B,T] boolean mask (True masks padding positions)
        """
        src_with_pos = self.pos_encoder(src_emb)
        h = self.transformer_encoder(
            src=src_with_pos,
            mask=attn_mask,
            src_key_padding_mask=padding_mask
        )
        return self.final_norm(h)
