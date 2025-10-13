# data_generator.py

import torch
import numpy as np
import random

# Define special tokens and vocabulary
# ADDED <EVENT> token to represent generic events in the sequence.
SPECIAL_TOKENS = {
    '<PAD>': 0, '<TASK_NEXT_ACTIVITY>': 1, '<TASK_REMAINING_TIME>': 2,
    '<CASE_SEP>': 3, '<LABEL>': 4, '<QUERY>': 5, '<EVENT>': 6
}
# We'll have 10 possible activities (A0-A9)
ACTIVITY_VOCAB_SIZE = 10
# We'll discretize remaining time into 50 buckets
TIME_BUCKET_VOCAB_SIZE = 50

# Total vocabulary size includes special tokens, activities, and time buckets
VOCAB_SIZE = len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + TIME_BUCKET_VOCAB_SIZE


class EpisodeGenerator:
    """
    Generates ICL episodes with support and query sets for event data.
    """

    def __init__(self, num_cases, max_case_len, num_cat_features, num_num_features):
        self.num_cases = num_cases
        self.max_case_len = max_case_len
        self.num_cat_features = num_cat_features
        self.num_num_features = num_num_features
        self.cat_cardinalities = [10] * num_cat_features  # Assume 10 categories per feature
        self.dataset = self._generate_dataset()

    def _generate_dataset(self):
        """Generates a synthetic dataset of event cases."""
        dataset = []
        for _ in range(self.num_cases):
            case_len = random.randint(5, self.max_case_len)
            events = []
            current_time = 0
            for i in range(case_len):
                # Features
                cat_feats = [random.randint(0, c - 1) for c in self.cat_cardinalities]
                num_feats = np.random.rand(self.num_num_features).tolist()
                delta_t = random.uniform(0.5, 5.0)
                current_time += delta_t
                time_feats = [np.log(delta_t + 1e-6), current_time]

                # Targets
                next_activity_label = random.randint(0, ACTIVITY_VOCAB_SIZE - 1)
                remaining_time = (case_len - 1 - i) * np.mean([2.5])  # Avg time step

                events.append({
                    'cat_feats': cat_feats,
                    'num_feats': num_feats,
                    'time_feats': time_feats,
                    'next_activity': next_activity_label,
                    'remaining_time': remaining_time
                })
            dataset.append(events)
        return dataset

    def _discretize_time(self, time_value):
        """Discretizes a continuous time value into a fixed number of buckets."""
        # Simple linear bucketing for demonstration
        # A log-scale might be better in practice
        max_rem_time = self.max_case_len * 5.0
        bucket = int((time_value / max_rem_time) * (TIME_BUCKET_VOCAB_SIZE - 1))
        return min(max(bucket, 0), TIME_BUCKET_VOCAB_SIZE - 1)

    def create_episode(self, k_shots, task):
        """
        Creates a single ICL episode.
        Format: [<TASK>]<SEP>[support_case_1]<LABEL>[y1]<SEP>...<QUERY>[query_case]<LABEL>
        """
        sampled_cases = random.sample(self.dataset, k_shots + 1)
        support_cases, query_case = sampled_cases[:k_shots], sampled_cases[-1]

        # Start with the task token
        task_token = SPECIAL_TOKENS[f'<TASK_{task.upper()}>']
        episode_tokens = [task_token]

        # --- Process Support Cases ---
        for case in support_cases:
            episode_tokens.append(SPECIAL_TOKENS['<CASE_SEP>'])

            # Use a random prefix of the case events
            prefix_len = random.randint(2, len(case) - 1)
            # FIXED: Use the valid <EVENT> token instead of out-of-bounds placeholders.
            event_tokens = [SPECIAL_TOKENS['<EVENT>']] * prefix_len
            episode_tokens.extend(event_tokens)
            episode_tokens.append(SPECIAL_TOKENS['<LABEL>'])

            if task == 'next_activity':
                label = case[prefix_len - 1]['next_activity']
                label_token = len(SPECIAL_TOKENS) + label
            else:  # remaining_time
                label = case[prefix_len - 1]['remaining_time']
                label_token = len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + self._discretize_time(label)
            episode_tokens.append(label_token)

        # --- Process Query Case ---
        episode_tokens.append(SPECIAL_TOKENS['<CASE_SEP>'])
        episode_tokens.append(SPECIAL_TOKENS['<QUERY>'])

        query_prefix_len = random.randint(2, len(query_case) - 1)
        # FIXED: Use the valid <EVENT> token instead of out-of-bounds placeholders.
        query_event_tokens = [SPECIAL_TOKENS['<EVENT>']] * query_prefix_len
        episode_tokens.extend(query_event_tokens)
        episode_tokens.append(SPECIAL_TOKENS['<LABEL>'])

        # Determine the ground truth for the query
        if task == 'next_activity':
            query_true_label = query_case[query_prefix_len - 1]['next_activity']
            query_true_token = len(SPECIAL_TOKENS) + query_true_label
        else:  # remaining_time
            query_true_label_continuous = query_case[query_prefix_len - 1]['remaining_time']
            query_true_token = len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + self._discretize_time(
                query_true_label_continuous)

        # Create a dictionary to pass real features for embedding
        # In a real implementation, you'd map events to unique IDs
        all_events_in_episode = []
        for case in support_cases:
            all_events_in_episode.extend(case)
        all_events_in_episode.extend(query_case)

        # The loss should only be calculated at the query's <LABEL> position
        loss_mask = [0] * (len(episode_tokens) - 1) + [1]

        return {
            "tokens": episode_tokens,
            "events": all_events_in_episode,  # Simplified mapping
            "loss_mask": loss_mask,
            "query_true_token": query_true_token,
            "query_true_continuous": query_true_label_continuous if task == 'remaining_time' else -1,
            "task": task
        }


def collate_batch(batch):
    """Pads sequences in a batch to the same length."""
    max_len = max(len(item['tokens']) for item in batch)

    padded_tokens = []
    padded_loss_masks = []
    attention_masks = []

    for item in batch:
        pad_len = max_len - len(item['tokens'])

        # Pad tokens and loss mask with <PAD> token ID (0)
        padded_tokens.append(item['tokens'] + [SPECIAL_TOKENS['<PAD>']] * pad_len)
        padded_loss_masks.append(item['loss_mask'] + [0] * pad_len)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_masks.append([1] * len(item['tokens']) + [0] * pad_len)

    # We need to handle event features separately. This is a simplification.
    # A robust implementation would map events to a global dictionary.
    return {
        "tokens": torch.tensor(padded_tokens, dtype=torch.long),
        "loss_mask": torch.tensor(padded_loss_masks, dtype=torch.bool),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.float),
        "query_true_tokens": torch.tensor([item['query_true_token'] for item in batch], dtype=torch.long),
        "query_true_continuous": torch.tensor([item['query_true_continuous'] for item in batch], dtype=torch.float),
        "tasks": [item['task'] for item in batch]
    }
