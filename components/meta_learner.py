# components/meta_learner.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
# --- MODIFIED IMPORTS ---
from .learned_event_embedder import LearnedEventEmbedder
from .pretrained_event_embedder import PretrainedEventEmbedder
# ------------------------
from .event_encoder import EventEncoder
from .prototypical_head import PrototypicalHead

class MetaLearner(nn.Module):
    """
    Combines embedder, encoder, and prototypical head for meta-learning.
    The internal embedder is chosen based on the specified strategy.
    """
    def __init__(self, strategy: str, num_feat_dim: int, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.strategy = strategy
        self.encoder = EventEncoder(d_model, n_heads, n_layers, dropout)
        self.proto_head = PrototypicalHead()
        self.d_model = d_model

        if self.strategy == 'pretrained':
            # (omitted for brevity - this part is unchanged)
            self.embedding_dim = kwargs['embedding_dim']
            self.embedder = PretrainedEventEmbedder(self.embedding_dim, num_feat_dim, d_model, dropout)
            self.pad_event = {'activity_embedding': np.zeros(self.embedding_dim, dtype=np.float32), 'resource_embedding': np.zeros(self.embedding_dim, dtype=np.float32), 'activity_id': 0, 'cost': 0.0, 'time_from_start': 0.0, 'time_from_previous': 0.0, 'timestamp': 0.0, 'case_id': 'pad'}
        elif self.strategy == 'learned':
            # Instantiate the new character-based embedder
            self.embedder = LearnedEventEmbedder(
                char_vocab_size=kwargs['char_vocab_size'],
                char_emb_dim=kwargs['char_embedding_dim'],
                char_cnn_out_dim=kwargs['char_cnn_output_dim'],
                num_feat_dim=num_feat_dim,
                d_model=d_model,
                dropout=dropout
            )
            # Define padding with empty strings for names
            self.pad_event = {
                'activity_name': '', 'resource_name': '', 'activity_id': -100,
                'cost': 0.0, 'time_from_start': 0.0, 'time_from_previous': 0.0,
                'timestamp': 0.0, 'case_id': 'pad'
            }
        else:
            raise ValueError(f"Unknown embedding strategy: '{self.strategy}'")

    def set_char_vocab(self, char_to_id: dict):
        """Passes the character vocabulary to the learned embedder."""
        if self.strategy == 'learned':
            self.embedder.char_to_id = char_to_id
            print("Character vocabulary set in LearnedEventEmbedder.")

    def _process_batch(self, batch_of_sequences):
        # (omitted for brevity - this function is unchanged)
        device = next(self.parameters()).device
        max_len = max(len(seq) for seq in batch_of_sequences) if batch_of_sequences else 0
        if max_len == 0: return torch.empty(0, self.d_model, device=device)
        padded_dfs, masks = [], []
        for seq in batch_of_sequences:
            pad_len = max_len - len(seq)
            mask = [False] * len(seq) + [True] * pad_len
            df = pd.DataFrame(seq)
            if pad_len > 0:
                pad_df = pd.DataFrame([self.pad_event] * pad_len)
                df = pd.concat([df, pad_df], ignore_index=True)
            padded_dfs.append(df)
            masks.append(mask)
        batch_df = pd.concat(padded_dfs, ignore_index=True)
        all_embeddings = self.embedder(batch_df)
        embeddings_reshaped = all_embeddings.view(len(batch_of_sequences), max_len, -1)
        mask_tensor = torch.tensor(masks, dtype=torch.bool, device=device)
        return self.encoder(embeddings_reshaped, src_key_padding_mask=mask_tensor)

    def forward(self, support_set, query_set, task_type):
        # (omitted for brevity - this function is unchanged)
        support_seqs, query_seqs = [s[0] for s in support_set], [q[0] for q in query_set]
        all_seqs = support_seqs + query_seqs
        if not all_seqs: return None, None
        all_encoded = self._process_batch(all_seqs)
        support_features = all_encoded[:len(support_seqs)]
        query_features = all_encoded[len(support_seqs):]
        device = all_encoded.device
        if task_type == 'classification':
            support_labels = torch.LongTensor([s[1] for s in support_set]).to(device)
            query_labels = torch.LongTensor([q[1] for q in query_set]).to(device)
            predictions, proto_classes = self.proto_head.forward_classification(support_features, support_labels, query_features)
            if predictions is None: return None, None
            label_map = {orig_label.item(): new_label for new_label, orig_label in enumerate(proto_classes)}
            mapped_labels = torch.tensor([label_map.get(l.item(), -100) for l in query_labels], device=device, dtype=torch.long)
            return predictions, mapped_labels
        elif task_type == 'regression':
            support_labels = torch.as_tensor([s[1] for s in support_set], dtype=torch.float32, device=device)
            query_labels = torch.as_tensor([q[1] for q in query_set], dtype=torch.float32, device=device)
            predictions = self.proto_head.forward_regression(support_features, support_labels, query_features)
            return predictions, query_labels
        else: raise ValueError(f"Unknown task type: {task_type}")
