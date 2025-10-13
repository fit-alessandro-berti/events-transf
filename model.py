# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from components import EventEmbedder, DecoderOnlyTransformer
from data_generator import VOCAB_SIZE, ACTIVITY_VOCAB_SIZE, TIME_BUCKET_VOCAB_SIZE, SPECIAL_TOKENS


class IOTransformer(nn.Module):
    """
    Decoder-only backbone with:
      - Task conditioning (adds a task embedding to all tokens)
      - Weight-tied class heads (aligns class tokens with outputs)
      - Retrieval/copy head at *every* <LABEL> position using earlier support label-value tokens
        (causal: each <LABEL> only sees keys strictly before it)
    """
    def __init__(self, d_model, n_layers, n_heads, cat_cardinalities, num_num_features, num_time_features, dropout=0.1):
        super().__init__()

        self.embedder = EventEmbedder(
            vocab_size=VOCAB_SIZE, d_model=d_model, cat_cardinalities=cat_cardinalities,
            num_num_features=num_num_features, num_time_features=num_time_features, dropout=dropout
        )

        self.backbone = DecoderOnlyTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout
        )

        # Parametric heads (learned linear layers)
        self.param_next = nn.Linear(d_model, ACTIVITY_VOCAB_SIZE)
        self.param_time = nn.Linear(d_model, TIME_BUCKET_VOCAB_SIZE)

        # Learned task embedding added to ALL tokens (0=next_activity, 1=remaining_time)
        self.task_embed = nn.Embedding(2, d_model)

        # Scales for tied-weight logits and retrieval logits (positivized by softplus)
        # Start copy scales higher to encourage early use of support examples.
        self.tied_scale_act = nn.Parameter(torch.tensor(0.0))   # softplus -> ~0.693
        self.tied_scale_time = nn.Parameter(torch.tensor(0.0))
        self.copy_scale_act = nn.Parameter(torch.tensor(1.5))   # softplus -> ~1.70
        self.copy_scale_time = nn.Parameter(torch.tensor(1.5))

        # Learned temperatures for cosine sims in copy logits
        self.copy_temp_act = nn.Parameter(torch.tensor(1.0))    # softplus -> ~1.31
        self.copy_temp_time = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _generate_causal_mask(sz: int, device: torch.device):
        """
        Return a boolean causal mask where True indicates positions that should be masked.
        """
        # upper triangular (excluding diagonal) is masked
        return torch.ones(sz, sz, dtype=torch.bool, device=device).triu(1)

    def _task_ids(self, tasks):
        # Map list[str] -> tensor ids
        ids = [0 if t == 'next_activity' else 1 for t in tasks]
        return torch.tensor(ids, dtype=torch.long, device=self.tied_scale_act.device)

    def _tied_logits(self, h: torch.Tensor):
        """
        Compute logits by tying output weights to the class token embeddings.
        h: [B,T,D]
        returns: (logits_act, logits_time)
        """
        # Grab the token embedding table
        E = self.embedder.token_embed.weight  # [VOCAB, D]
        act_start = len(SPECIAL_TOKENS)
        time_start = act_start + ACTIVITY_VOCAB_SIZE

        W_act = E[act_start: act_start + ACTIVITY_VOCAB_SIZE]       # [A, D]
        W_time = E[time_start: time_start + TIME_BUCKET_VOCAB_SIZE] # [B, D]

        tied_act = F.linear(h, W_act)    # [B,T,A]
        tied_time = F.linear(h, W_time)  # [B,T,B]
        return tied_act, tied_time

    def _retrieval_logits_all_labels(self, h: torch.Tensor, tokens: torch.Tensor):
        """
        Build copy/retrieval logits at *every* <LABEL> position using only keys from the *past*.

        Keys = positions whose previous token is <LABEL> (i.e., the support label-value tokens).
        For each label position L, we select keys j where j < L to preserve causality.

        h: [B,T,D], tokens: [B,T]
        Returns two tensors shaped like full-logit maps:
          act_logits:  [B,T,ACTIVITY_VOCAB_SIZE]
          time_logits: [B,T,TIME_BUCKET_VOCAB_SIZE]
        Non-<LABEL> positions are zeros.
        """
        B, T, D = h.shape
        device = h.device

        act_logits = torch.zeros(B, T, ACTIVITY_VOCAB_SIZE, device=device)
        time_logits = torch.zeros(B, T, TIME_BUCKET_VOCAB_SIZE, device=device)

        act_start = len(SPECIAL_TOKENS)
        time_start = act_start + ACTIVITY_VOCAB_SIZE

        # cosine temperature (positive)
        tau_act = F.softplus(self.copy_temp_act)
        tau_time = F.softplus(self.copy_temp_time)

        # Precompute normalized states for cosine similarity
        h_norm = F.normalize(h, dim=-1)  # [B,T,D]

        for b in range(B):
            tok_b = tokens[b]       # [T]
            h_b = h_norm[b]         # [T,D]

            # Label markers
            is_label = (tok_b == SPECIAL_TOKENS['<LABEL>'])  # [T]
            label_pos = torch.nonzero(is_label, as_tuple=False).squeeze(-1)  # [N_l] (possibly 0)

            if label_pos.numel() == 0:
                continue

            # Value tokens = tokens whose previous token is <LABEL>
            value_mask = torch.zeros_like(is_label)
            value_mask[1:] = is_label[:-1]

            val_act_mask = value_mask & (tok_b >= act_start) & (tok_b < time_start)
            val_time_mask = value_mask & (tok_b >= time_start) & (tok_b < time_start + TIME_BUCKET_VOCAB_SIZE)

            key_pos_act = torch.nonzero(val_act_mask, as_tuple=False).squeeze(-1)   # [K_a]
            key_pos_time = torch.nonzero(val_time_mask, as_tuple=False).squeeze(-1) # [K_t]

            # Class indices for keys
            if key_pos_act.numel() > 0:
                key_cls_act = (tok_b.index_select(0, key_pos_act) - act_start).long()  # [K_a]
            if key_pos_time.numel() > 0:
                key_cls_time = (tok_b.index_select(0, key_pos_time) - time_start).long()  # [K_t]

            # For each label L, use only keys strictly before L to keep it causal
            for L in label_pos.tolist():
                if key_pos_act.numel() > 0:
                    mask_before = key_pos_act < L
                    if mask_before.any():
                        K = h_b.index_select(0, key_pos_act[mask_before])       # [kL, D]
                        cls_idx = key_cls_act[mask_before]                       # [kL]
                        q = h_b[L].unsqueeze(0)                                   # [1, D]
                        sims = (q @ K.t()).squeeze(0) * tau_act                  # [kL] cosine * temperature
                        accum = torch.zeros(ACTIVITY_VOCAB_SIZE, device=device)
                        accum = accum.index_add(0, cls_idx, sims)                # sum similarities per class
                        act_logits[b, L] = accum

                if key_pos_time.numel() > 0:
                    mask_before = key_pos_time < L
                    if mask_before.any():
                        K = h_b.index_select(0, key_pos_time[mask_before])       # [kL, D]
                        cls_idx = key_cls_time[mask_before]                      # [kL]
                        q = h_b[L].unsqueeze(0)                                   # [1, D]
                        sims = (q @ K.t()).squeeze(0) * tau_time                 # [kL]
                        accum = torch.zeros(TIME_BUCKET_VOCAB_SIZE, device=device)
                        accum = accum.index_add(0, cls_idx, sims)
                        time_logits[b, L] = accum

        return act_logits, time_logits

    def forward(self, batch):
        token_ids = batch['tokens']                # [B,T]
        cat_feats = batch['cat_feats']            # [B,T,C]
        num_feats = batch['num_feats']            # [B,T,Fn]
        time_feats = batch['time_feats']          # [B,T,Ft]
        padding_mask = (batch['attention_mask'] == 0)  # [B,T] bool
        tasks = batch.get('tasks', None)

        device = token_ids.device
        seq_len = token_ids.size(1)
        causal_mask = self._generate_causal_mask(seq_len, device)  # [T,T] bool

        # Base embeddings (event/type/case-aware)
        x_emb = self.embedder(token_ids, cat_feats, num_feats, time_feats)

        # Task conditioning: add the same task vector to all tokens in the episode
        if tasks is not None:
            task_ids = self._task_ids(tasks)                        # [B]
            task_vec = self.task_embed(task_ids).unsqueeze(1)       # [B,1,D]
            x_emb = x_emb + task_vec                                # broadcast over T

        # Backbone
        h = self.backbone(x_emb, attn_mask=causal_mask, padding_mask=padding_mask)  # [B,T,D]

        # Parametric logits
        param_next = self.param_next(h)      # [B,T,A]
        param_time = self.param_time(h)      # [B,T,B]

        # Tied-weight logits (use class-token embeddings)
        tied_act, tied_time = self._tied_logits(h)                  # [B,T,A], [B,T,B]

        # Retrieval/copy logits at *all* <LABEL> positions (zeros elsewhere)
        copy_act, copy_time = self._retrieval_logits_all_labels(h, token_ids)

        # Combine with learned positive scales (softplus)
        s_tied_act = F.softplus(self.tied_scale_act)
        s_tied_time = F.softplus(self.tied_scale_time)
        s_copy_act = F.softplus(self.copy_scale_act)
        s_copy_time = F.softplus(self.copy_scale_time)

        activity_logits = param_next + s_tied_act * tied_act + s_copy_act * copy_act
        time_logits = param_time + s_tied_time * tied_time + s_copy_time * copy_time

        return activity_logits, time_logits
