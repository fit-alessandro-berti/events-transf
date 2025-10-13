# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from components import EventEmbedder, DecoderOnlyTransformer
from data_generator import VOCAB_SIZE, ACTIVITY_VOCAB_SIZE, TIME_BUCKET_VOCAB_SIZE, SPECIAL_TOKENS


class IOTransformer(nn.Module):
    """
    Decoder-only backbone with three complementary heads:
      1) Parametric heads (learned linear classifiers)
      2) Weight-tied heads (dot with class-token embeddings)
      3) Prototype head trained at *all* <LABEL> positions:
         - builds class prototypes from earlier SUPPORT <LABEL> positions (causally)
         - adds a Bayesian prior from class-token embeddings (helps at small K)
         - compares each <LABEL> hidden state to prototypes via cosine similarity

    All heads are combined with learned positive gates (softplus).
    A task embedding conditions the whole sequence (next_activity vs remaining_time).
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

        # Scales for tied-weight and prototype logits (positivized by softplus)
        self.tied_scale_act = nn.Parameter(torch.tensor(0.0))   # ~0.69 after softplus
        self.tied_scale_time = nn.Parameter(torch.tensor(0.0))
        # Start prototype gates >0 so the model uses supports early
        self.proto_scale_act = nn.Parameter(torch.tensor(1.0))  # ~1.31 after softplus
        self.proto_scale_time = nn.Parameter(torch.tensor(1.0))

        # Prototype priors: how much class-token embedding contributes to the prototype
        self.proto_prior_act = nn.Parameter(torch.tensor(0.5))  # ~1.13 after softplus
        self.proto_prior_time = nn.Parameter(torch.tensor(0.5))

        # Temperatures for cosine similarities in prototype logits
        self.proto_temp_act = nn.Parameter(torch.tensor(1.0))   # ~1.31 after softplus
        self.proto_temp_time = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _generate_causal_mask(sz: int, device: torch.device):
        # upper triangular (excluding diagonal) is masked
        return torch.ones(sz, sz, dtype=torch.bool, device=device).triu(1)

    def _task_ids(self, tasks):
        # Map list[str] -> tensor ids
        ids = [0 if t == 'next_activity' else 1 for t in tasks]
        return torch.tensor(ids, dtype=torch.long, device=self.tied_scale_act.device)

    def _tied_logits(self, h: torch.Tensor):
        """
        Weight-tied logits using class-token embeddings.
        """
        E = self.embedder.token_embed.weight  # [VOCAB, D]
        act_start = len(SPECIAL_TOKENS)
        time_start = act_start + ACTIVITY_VOCAB_SIZE

        W_act = E[act_start: act_start + ACTIVITY_VOCAB_SIZE]       # [A, D]
        W_time = E[time_start: time_start + TIME_BUCKET_VOCAB_SIZE] # [B, D]

        tied_act = F.linear(h, W_act)    # [B,T,A]
        tied_time = F.linear(h, W_time)  # [B,T,B]
        return tied_act, tied_time

    def _prototype_logits_all_labels(self, h: torch.Tensor, tokens: torch.Tensor):
        """
        Build prototype logits at *every* <LABEL> position using only earlier
        SUPPORT <LABEL> positions (causal). Non-<LABEL> positions are zeros.

        For each <LABEL> at position L:
          - collect support label positions S where S < L and S is a support label
          - use NEXT token at S to get the class id
          - compute per-class prototype as (sum of normalized h[S] + alpha * class_emb) / (count + alpha)
          - similarity = temperature * cosine(h[L], prototype[c])
        """
        B, T, D = h.shape
        device = h.device

        act_logits = torch.zeros(B, T, ACTIVITY_VOCAB_SIZE, device=device)
        time_logits = torch.zeros(B, T, TIME_BUCKET_VOCAB_SIZE, device=device)

        act_start = len(SPECIAL_TOKENS)
        time_start = act_start + ACTIVITY_VOCAB_SIZE

        # Normalize hidden states and class-token embeddings for cosine
        h_norm = F.normalize(h, dim=-1)  # [B,T,D]
        E = self.embedder.token_embed.weight.detach()  # [VOCAB, D]
        E_act = F.normalize(E[act_start: act_start + ACTIVITY_VOCAB_SIZE], dim=-1)                 # [A,D]
        E_time = F.normalize(E[time_start: time_start + TIME_BUCKET_VOCAB_SIZE], dim=-1)           # [B,D]

        alpha_act = F.softplus(self.proto_prior_act)    # >0
        alpha_time = F.softplus(self.proto_prior_time)  # >0
        tau_act = F.softplus(self.proto_temp_act)       # >0
        tau_time = F.softplus(self.proto_temp_time)     # >0

        # Identify label positions
        is_label = (tokens == SPECIAL_TOKENS['<LABEL>'])        # [B,T]
        # Identify the very next token (class token) for each position
        next_token_ids = torch.roll(tokens, shifts=-1, dims=1)  # [B,T]

        # Query label is the very last <LABEL> in each episode by construction,
        # but we don't need to know which one it is here; we always restrict to earlier supports.
        for b in range(B):
            tok_b = tokens[b]                 # [T]
            h_b = h_norm[b]                   # [T,D]

            label_pos = torch.nonzero(is_label[b], as_tuple=False).squeeze(-1)  # [N_l]
            if label_pos.numel() == 0:
                continue

            # Support label candidates: all label positions; we'll filter "earlier than L" inside the loop.
            sup_pos_all = label_pos
            nxt_ids_all = next_token_ids[b].index_select(0, sup_pos_all)  # [N_l]
            h_sup_all = h_b.index_select(0, sup_pos_all)                  # [N_l, D]

            # Pre-split by class type to avoid doing it many times
            sup_is_act = (nxt_ids_all >= act_start) & (nxt_ids_all < time_start)
            sup_is_time = (nxt_ids_all >= time_start) & (nxt_ids_all < time_start + TIME_BUCKET_VOCAB_SIZE)

            for L in label_pos.tolist():
                # mask supports strictly before L
                before_mask = (sup_pos_all < L)
                if before_mask.any():
                    # ---- Activities
                    mask_a = before_mask & sup_is_act
                    if mask_a.any():
                        sup_idx = torch.nonzero(mask_a, as_tuple=False).squeeze(-1)
                        cls_idx = (nxt_ids_all.index_select(0, sup_idx) - act_start).long()       # [kA]
                        H = h_sup_all.index_select(0, sup_idx)                                     # [kA, D]

                        Hsum = torch.zeros(ACTIVITY_VOCAB_SIZE, D, device=device)
                        Hsum.index_add_(0, cls_idx, H)
                        Cnt = torch.zeros(ACTIVITY_VOCAB_SIZE, device=device)
                        Cnt.index_add_(0, cls_idx, torch.ones_like(cls_idx, dtype=torch.float))

                        # Bayesian-smoothed prototypes (normalize again for cosine)
                        proto = (Hsum + alpha_act * E_act) / (Cnt.unsqueeze(-1) + alpha_act + 1e-8)
                        proto = F.normalize(proto, dim=-1)                                         # [A,D]

                        q = h_b[L].unsqueeze(0)                                                    # [1,D]
                        sims = (q @ proto.t()).squeeze(0) * tau_act                                # [A]
                        act_logits[b, L] = sims

                    # ---- Time buckets
                    mask_t = before_mask & sup_is_time
                    if mask_t.any():
                        sup_idx = torch.nonzero(mask_t, as_tuple=False).squeeze(-1)
                        cls_idx = (nxt_ids_all.index_select(0, sup_idx) - time_start).long()       # [kB]
                        H = h_sup_all.index_select(0, sup_idx)                                     # [kB, D]

                        Hsum = torch.zeros(TIME_BUCKET_VOCAB_SIZE, D, device=device)
                        Hsum.index_add_(0, cls_idx, H)
                        Cnt = torch.zeros(TIME_BUCKET_VOCAB_SIZE, device=device)
                        Cnt.index_add_(0, cls_idx, torch.ones_like(cls_idx, dtype=torch.float))

                        proto = (Hsum + alpha_time * E_time) / (Cnt.unsqueeze(-1) + alpha_time + 1e-8)
                        proto = F.normalize(proto, dim=-1)                                         # [B,D]

                        q = h_b[L].unsqueeze(0)                                                    # [1,D]
                        sims = (q @ proto.t()).squeeze(0) * tau_time                               # [B]
                        time_logits[b, L] = sims

                # If no prior supports, we leave logits at zeros here; the param/tied heads
                # (and their priors) carry the prediction for early labels.

        return act_logits, time_logits  # zeros at non-<LABEL> positions

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

        # Prototype logits at *all* <LABEL> positions (zeros elsewhere)
        proto_act, proto_time = self._prototype_logits_all_labels(h, token_ids)

        # Combine with learned positive scales (softplus)
        s_tied_act = F.softplus(self.tied_scale_act)
        s_tied_time = F.softplus(self.tied_scale_time)
        s_proto_act = F.softplus(self.proto_scale_act)
        s_proto_time = F.softplus(self.proto_scale_time)

        activity_logits = param_next + s_tied_act * tied_act + s_proto_act * proto_act
        time_logits = param_time + s_tied_time * tied_time + s_proto_time * proto_time

        return activity_logits, time_logits
