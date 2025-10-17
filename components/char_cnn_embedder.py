# components/char_cnn_embedder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CharCNNEmbedder(nn.Module):
    """
    Generates a fixed-size embedding for a word from its characters
    using a 1D Convolutional Neural Network.
    """

    def __init__(self, char_vocab_size: int, char_embedding_dim: int, output_dim: int, max_word_len: int = 30):
        super().__init__()
        self.max_word_len = max_word_len

        # Character embedding layer
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)

        # 1D convolutions that act as n-gram detectors
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_embedding_dim, out_channels=32, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=char_embedding_dim, out_channels=32, kernel_size=4, padding=2),
            nn.Conv1d(in_channels=char_embedding_dim, out_channels=32, kernel_size=5, padding=2),
        ])

        # The output dimension of the concatenated conv layers
        cnn_output_dim = 32 * len(self.convs)

        # Final projection layer
        self.projection = nn.Sequential(
            nn.Linear(cnn_output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def _strings_to_char_ids(self, strings: list[str], char_to_id: dict):
        """Converts a list of strings to a padded tensor of character IDs."""
        device = self.char_embedding.weight.device
        batch_char_ids = []
        unk_id = char_to_id.get('<UNK>', 1)

        for s in strings:
            s = s[:self.max_word_len]  # Truncate long strings
            char_ids = [char_to_id.get(c, unk_id) for c in s]
            # Pad to max_word_len
            padded_ids = char_ids + [0] * (self.max_word_len - len(char_ids))
            batch_char_ids.append(padded_ids)

        return torch.tensor(batch_char_ids, dtype=torch.long, device=device)

    def forward(self, strings: list[str], char_to_id: dict):
        """
        Args:
            strings (list[str]): A list of activity or resource names.
            char_to_id (dict): The mapping from character to integer ID.

        Returns:
            torch.Tensor: A tensor of shape (len(strings), output_dim).
        """
        # 1. Convert strings to character ID tensor: (batch_size, max_word_len)
        char_ids = self._strings_to_char_ids(strings, char_to_id)

        # 2. Get character embeddings: (batch_size, max_word_len, char_embedding_dim)
        embedded_chars = self.char_embedding(char_ids)

        # 3. Reshape for Conv1d: (batch_size, char_embedding_dim, max_word_len)
        embedded_chars = embedded_chars.permute(0, 2, 1)

        # 4. Apply convolutions and activation
        conv_outputs = [F.gelu(conv(embedded_chars)) for conv in self.convs]

        # 5. Apply max-pooling over time (the word length dimension)
        pooled_outputs = [F.max_pool1d(out, out.shape[2]).squeeze(2) for out in conv_outputs]

        # 6. Concatenate pooled features: (batch_size, cnn_output_dim)
        concatenated = torch.cat(pooled_outputs, dim=1)

        # 7. Project to the final output dimension
        final_embedding = self.projection(concatenated)

        return final_embedding
