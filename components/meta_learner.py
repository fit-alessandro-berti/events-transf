import torch
import torch.nn as nn
import pandas as pd
from .event_embedder import EventEmbedder
from .event_encoder import EventEncoder
from .prototypical_head import PrototypicalHead

class MetaLearner(nn.Module):
    """
    Combines embedder, encoder, and prototypical head for meta-learning.
    """

    def __init__(self, cat_vocabs, num_feat_dim, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.embedder = EventEmbedder(cat_vocabs, num_feat_dim, d_model, dropout=dropout)
        self.encoder = EventEncoder(d_model, n_heads, n_layers, dropout)
        self.proto_head = PrototypicalHead()

    def _process_batch(self, batch_of_sequences):
        """Embed and encode a batch of sequences of varying lengths."""
        device = next(self.parameters()).device

        # Pad sequences to the same length
        max_len = max(len(seq) for seq in batch_of_sequences)

        padded_dfs = []
        masks = []
        for seq in batch_of_sequences:
            pad_len = max_len - len(seq)
            mask = [False] * len(seq) + [True] * pad_len

            # Create a padded DataFrame
            df = pd.DataFrame(seq)
            if pad_len > 0:
                pad_df = pd.DataFrame([{k: 0 for k in df.columns}] * pad_len)
                df = pd.concat([df, pad_df], ignore_index=True)

            padded_dfs.append(df)
            masks.append(mask)

        # Batch embed and encode
        batch_df = pd.concat(padded_dfs, ignore_index=True)
        all_embeddings = self.embedder(batch_df)  # (batch_size * max_len, d_model)

        # Reshape to (batch_size, max_len, d_model) for batch_first=True transformer
        embeddings_reshaped = all_embeddings.view(len(batch_of_sequences), max_len, -1).to(device)

        mask_tensor = torch.tensor(masks, dtype=torch.bool, device=device)

        # Get final encoded representation for each sequence
        encoded_vectors = self.encoder(embeddings_reshaped, src_key_padding_mask=mask_tensor)
        return encoded_vectors

    def forward(self, support_set, query_set, task_type):
        """
        Processes a meta-learning episode.

        Args:
            support_set (list): List of (sequence, label) tuples.
            query_set (list): List of (sequence, label) tuples.
            task_type (str): 'classification' or 'regression'.

        Returns:
            Tuple: (predictions, true_labels)
        """
        support_seqs = [s[0] for s in support_set]
        support_labels = [s[1] for s in support_set]
        query_seqs = [q[0] for q in query_set]
        query_labels = [q[1] for q in query_set]

        # Encode all sequences
        all_seqs = support_seqs + query_seqs
        all_encoded = self._process_batch(all_seqs)

        num_support = len(support_seqs)
        support_features = all_encoded[:num_support]
        query_features = all_encoded[num_support:]

        if task_type == 'classification':
            support_labels_tensor = torch.LongTensor(support_labels).to(all_encoded.device)
            query_labels_tensor = torch.LongTensor(query_labels).to(all_encoded.device)

            log_probs, proto_classes = self.proto_head.forward_classification(
                support_features, support_labels_tensor, query_features
            )

            # Map original query labels to the order of prototypes
            label_map = {original_label.item(): new_label for new_label, original_label in enumerate(proto_classes)}
            mapped_query_labels = torch.tensor([label_map[l.item()] for l in query_labels_tensor], device=all_encoded.device)

            return log_probs, mapped_query_labels

        elif task_type == 'regression':
            support_labels_tensor = torch.as_tensor(support_labels, dtype=torch.float32, device=all_encoded.device)
            query_labels_tensor = torch.as_tensor(query_labels, dtype=torch.float32, device=all_encoded.device)

            predictions = self.proto_head.forward_regression(
                support_features, support_labels_tensor, query_features
            )
            return predictions, query_labels_tensor
        else:
            raise ValueError(f"Unknown task type: {task_type}")
