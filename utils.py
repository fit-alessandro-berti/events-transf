# /io_transformer/utils.py

import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class Vocabulary:
    """Manages mapping between tokens and integer IDs."""

    def __init__(self, name):
        self.name = name
        self.token2index = {"<PAD>": 0}
        self.index2token = {0: "<PAD>"}
        self.num_tokens = 1

    def add_token(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.num_tokens
            self.index2token[self.num_tokens] = token
            self.num_tokens += 1

    def __len__(self):
        return self.num_tokens


def build_vocabularies(event_log):
    """Creates vocabularies from the event log."""
    # Special tokens vocabulary
    token_vocab = Vocabulary('special_tokens')
    special_tokens = ["<QUERY>", "<CASE_SEP>", "<LABEL>", "<TASK_NEXT_ACTIVITY>", "<TASK_REMAINING_TIME>"]
    for token in special_tokens:
        token_vocab.add_token(token)

    # Feature vocabularies
    activity_vocab = Vocabulary('activity')
    resource_vocab = Vocabulary('resource')

    for trace in event_log:
        for activity in trace['activity'].unique():
            activity_vocab.add_token(activity)
        for resource in trace['resource'].unique():
            resource_vocab.add_token(resource)

    return {
        'token': token_vocab,
        'activity': activity_vocab,
        'resource': resource_vocab
    }


def build_icl_episode(task, full_log, vocabs, k_shots, is_training=True):
    """
    Builds a single ICL episode with K support examples and 1 query.
    """
    # Sample K+1 unique cases
    sampled_indices = np.random.choice(len(full_log), k_shots + 1, replace=False)
    support_cases = [full_log[i] for i in sampled_indices[:k_shots]]
    query_case = full_log[sampled_indices[-1]]

    # Task token
    task_token = f"<TASK_{task.upper()}>"
    sequence = [{'token': task_token}]

    # 1. Build Support Examples
    for case in support_cases:
        sequence.append({'token': '<CASE_SEP>'})

        # Select a random event in the case to be the prediction point
        if len(case) < 2: continue
        predict_idx = np.random.randint(1, len(case))
        prefix = case.iloc[:predict_idx]

        # Add events from the prefix
        for _, event in prefix.iterrows():
            sequence.append({'activity': event['activity'], 'resource': event['resource'], 'cost': event['cost']})

        sequence.append({'token': '<LABEL>'})

        # Add the true label
        if task == 'next_activity':
            true_label = case.iloc[predict_idx]['activity']
            sequence.append({'activity': true_label})  # Label is an activity token
        elif task == 'remaining_time':
            end_time = case.iloc[-1]['timestamp']
            predict_time = case.iloc[predict_idx - 1]['timestamp']
            remaining_time = (end_time - predict_time).total_seconds() / 3600.0  # In hours
            sequence.append({'cost': remaining_time})  # Use numeric field for label

    # 2. Build Query Example
    sequence.append({'token': '<CASE_SEP>'})
    sequence.append({'token': '<QUERY>'})

    if len(query_case) < 2: return None  # Skip empty/too short queries
    query_predict_idx = np.random.randint(1, len(query_case))
    query_prefix = query_case.iloc[:query_predict_idx]

    for _, event in query_prefix.iterrows():
        sequence.append({'activity': event['activity'], 'resource': event['resource'], 'cost': event['cost']})

    sequence.append({'token': '<LABEL>'})

    # The model must predict what comes after <LABEL>
    if task == 'next_activity':
        query_target = vocabs['activity'].token2index[query_case.iloc[query_predict_idx]['activity']]
    elif task == 'remaining_time':
        end_time = query_case.iloc[-1]['timestamp']
        predict_time = query_case.iloc[query_predict_idx - 1]['timestamp']
        query_target = (end_time - predict_time).total_seconds() / 3600.0

    return sequence, task, query_target


def collate_fn(batch, vocabs):
    """
    Prepares a batch of episodes for the model.
    Handles tokenization, feature extraction, and padding.
    """
    token_sequences, activity_sequences, resource_sequences = [], [], []
    numeric_sequences, time_sequences = [], []
    targets, target_masks = [], []

    # Hardcoded mean/std for normalization - in a real scenario, compute from training set
    cost_mean, cost_std = 55.0, 25.0
    time_mean, time_std = 5.0, 5.0

    for sequence, task, target in batch:
        if sequence is None: continue

        # Initialize lists for features of the current sequence
        tok_ids, act_ids, res_ids = [], [], []
        num_feats, time_feats = [], []

        # Process timestamps and deltas
        timestamps = [pd.Timestamp('1970-01-01')] + [e['timestamp'] for e in sequence if 'timestamp' in e]

        event_idx = 0
        for i, item in enumerate(sequence):
            # 1. Tokenization
            tok_ids.append(vocabs['token'].token2index.get(item.get('token', '<PAD>'), 0))
            act_ids.append(vocabs['activity'].token2index.get(item.get('activity', '<PAD>'), 0))
            res_ids.append(vocabs['resource'].token2index.get(item.get('resource', '<PAD>'), 0))

            # 2. Feature extraction
            is_event = 'activity' in item
            if is_event:
                # Normalize numeric features
                cost = (item.get('cost', cost_mean) - cost_mean) / cost_std
                num_feats.append([cost])

                # Compute time features (log-scaled)
                delta_from_start = (timestamps[event_idx + 1] - timestamps[0]).total_seconds() / 3600.0
                delta_from_prev = (timestamps[event_idx + 1] - timestamps[event_idx]).total_seconds() / 3600.0

                time_feats.append([
                    np.log(delta_from_start + 1e-6),
                    np.log(delta_from_prev + 1e-6)
                ])
                event_idx += 1
            else:  # Special token
                num_feats.append([0.0])  # Z-scored zero
                time_feats.append([0.0, 0.0])

        token_sequences.append(torch.LongTensor(tok_ids))
        activity_sequences.append(torch.LongTensor(act_ids))
        resource_sequences.append(torch.LongTensor(res_ids))
        numeric_sequences.append(torch.FloatTensor(num_feats))
        time_sequences.append(torch.FloatTensor(time_feats))
        targets.append(target)

        # Create a mask to identify the query label position for loss calculation
        target_mask = torch.zeros(len(tok_ids), dtype=torch.bool)
        query_label_pos = len(tok_ids) - 1  # The <LABEL> token is always last in the query part
        target_mask[query_label_pos] = True
        target_masks.append(target_mask)

    # Pad sequences
    batch_dict = {
        'token_ids': pad_sequence(token_sequences, batch_first=True, padding_value=0),
        'activity_ids': pad_sequence(activity_sequences, batch_first=True, padding_value=0),
        'resource_ids': pad_sequence(resource_sequences, batch_first=True, padding_value=0),
        'numeric_features': pad_sequence(numeric_sequences, batch_first=True, padding_value=0.0),
        'time_features': pad_sequence(time_sequences, batch_first=True, padding_value=0.0),
        'targets': torch.tensor(targets),
        'target_mask': pad_sequence(target_masks, batch_first=True, padding_value=False),
        'padding_mask': (pad_sequence(token_sequences, batch_first=True, padding_value=0) == 0)
        # True for padded tokens
    }
    return batch_dict
