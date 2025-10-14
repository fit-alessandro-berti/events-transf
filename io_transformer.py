
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from model_components import (
    EventEmbed, PositionalEncoding, SegmentEmbedding, DecoderOnlyTransformer,
    SpecialTokenEmbedding, ClassLabelEmbedding, NumericLabelEmbedding
)
from utils import TOKEN_TYPES, SPECIAL_TOKENS


@dataclass
class IOTransformerConfig:
    num_activities: int
    num_resources: int
    num_dims: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.1
    n_classes: int = 128
    n_special: int = len(SPECIAL_TOKENS)


class IOTransformer(nn.Module):
    def __init__(self, cfg: IOTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.event_embed = EventEmbed(cfg.num_activities, cfg.num_resources, cfg.num_dims, cfg.d_model)
        self.special_embed = SpecialTokenEmbedding(cfg.n_special, cfg.d_model)
        self.class_label_embed = ClassLabelEmbedding(cfg.n_classes, cfg.d_model)
        self.num_label_embed = NumericLabelEmbedding(cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.max_seq_len, cfg.d_model)
        self.seg_enc = SegmentEmbedding(cfg.d_model, num_segments=2)
        self.backbone = DecoderOnlyTransformer(cfg.n_layers, cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
        self.cls_head = nn.Linear(cfg.d_model, cfg.n_classes)
        self.reg_head = nn.Linear(cfg.d_model, 2)  # (mu, log_sigma)

    def forward(self, batch, predict_task: Literal["cls", "reg", "both"] = "both"):
        B, T = batch.activity_ids.size(0), batch.activity_ids.size(1)
        device = batch.activity_ids.device
        d = self.cfg.d_model
        x = torch.zeros(B, T, d, device=device)

        is_event = (batch.token_types == TOKEN_TYPES["EVENT"]).unsqueeze(-1)
        ev = self.event_embed(batch.activity_ids, batch.resource_ids, batch.num_feats, batch.time_feats)
        x = x + ev * is_event

        is_spec = (batch.token_types == TOKEN_TYPES["SPECIAL"]).unsqueeze(-1)
        se = self.special_embed(batch.special_ids)
        x = x + se * is_spec

        is_cls = (batch.token_types == TOKEN_TYPES["CLS_LABEL"]).unsqueeze(-1)
        cle = self.class_label_embed(batch.class_label_ids)
        x = x + cle * is_cls

        is_num = (batch.token_types == TOKEN_TYPES["NUM_LABEL"]).unsqueeze(-1)
        nle = self.num_label_embed(batch.num_label_values)
        x = x + nle * is_num

        x = self.pos_enc(x)
        x = self.seg_enc(x, batch.segment_ids)
        h = self.backbone(x, batch.attn_mask)

        out = {}
        if predict_task in ("cls", "both"):
            out["logits"] = self.cls_head(h)
        if predict_task in ("reg", "both"):
            mu_logsigma = self.reg_head(h)
            out["mu"], out["log_sigma"] = mu_logsigma[..., 0], mu_logsigma[..., 1]
        return out
