# components/meta_learner.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from .event_embedder import EventEmbedder
from .event_encoder import EventEncoder
from .prototypical_head import PrototypicalHead


class MetaLearner(nn.Module):
    """
    Combines embedder, encoder, and prototypical head for meta-learning.
    """

    def __init__(self, embedding_dim, num_feat_dim, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.embedder = EventEmbedder(embedding_dim, num_feat_dim, d_model, dropout=dropout)
        self.encoder = EventEncoder(d_model, n_heads, n_layers, dropout)
        self.proto_head = PrototypicalHead()
        self.embedding_dim = embedding_dim

    def _process_batch(self, batch_of_sequences):
        """Embed and encode a batch of sequences of varying lengths."""
        device = next(self.parameters()).device

        max_len = max(len(seq) for seq in batch_of_sequences) if batch_of_sequences else 0
        if max_len == 0:
            # Handle case of empty batch
            return torch.empty(0, self.encoder.d_model, device=device)

        padded_dfs = []
        masks = []

        # Define the structure of a padding event
        pad_event = {
            'activity_embedding': np.zeros(self.embedding_dim, dtype=np.float32),
            'resource_embedding': np.zeros(self.embedding_dim, dtype=np.float32),
            'activity_id': 0,
            'cost': 0.0,
            'time_from_start': 0.0,
            'time_from_previous': 0.0,
            'timestamp': 0.0,
            'case_id': 'pad'
        }

        for seq in batch_of_sequences:
            pad_len = max_len - len(seq)
            mask = [False] * len(seq) + [True] * pad_len

            df = pd.DataFrame(seq)
            if pad_len > 0:
                pad_df = pd.DataFrame([pad_event] * pad_len)
                df = pd.concat([df, pad_df], ignore_index=True)

            padded_dfs.append(df)
            masks.append(mask)

        batch_df = pd.concat(padded_dfs, ignore_index=True)
        all_embeddings = self.embedder(batch_df)

        embeddings_reshaped = all_embeddings.view(len(batch_of_sequences), max_len, -1)
        mask_tensor = torch.tensor(masks, dtype=torch.bool, device=device)

        encoded_vectors = self.encoder(embeddings_reshaped, src_key_padding_mask=mask_tensor)
        return encoded_vectors

    def forward(self, support_set, query_set, task_type):
        support_seqs = [s[0] for s in support_set]
        support_labels = [s[1] for s in support_set]
        query_seqs = [q[0] for q in query_set]
        query_labels = [q[1] for q in query_set]

        all_seqs = support_seqs + query_seqs
        if not all_seqs:
            return None, None

        all_encoded = self._process_batch(all_seqs)

        num_support = len(support_seqs)
        support_features = all_encoded[:num_support]
        query_features = all_encoded[num_support:]

        if task_type == 'classification':
            support_labels_tensor = torch.LongTensor(support_labels).to(all_encoded.device)
            query_labels_tensor = torch.LongTensor(query_labels).to(all_encoded.device)

            predictions, proto_classes = self.proto_head.forward_classification(
                support_features, support_labels_tensor, query_features
            )

            if predictions is None:
                return None, None

            label_map = {original_label.item(): new_label for new_label, original_label in enumerate(proto_classes)}
            mapped_labels = [label_map.get(l.item(), -100) for l in query_labels_tensor]
            mapped_query_labels = torch.tensor(mapped_labels, device=all_encoded.device, dtype=torch.long)

            return predictions, mapped_query_labels

        elif task_type == 'regression':
            support_labels_tensor = torch.as_tensor(support_labels, dtype=torch.float32, device=all_encoded.device)
            query_labels_tensor = torch.as_tensor(query_labels, dtype=torch.float32, device=all_encoded.device)

            predictions = self.proto_head.forward_regression(
                support_features, support_labels_tensor, query_features
            )
            return predictions, query_labels_tensor
        else:
            raise ValueError(f"Unknown task type: {task_type}")
