# data_generator.py

import torch
import numpy as np
import random

# Define special tokens and vocabulary
SPECIAL_TOKENS = {
    '<PAD>': 0, '<TASK_NEXT_ACTIVITY>': 1, '<TASK_REMAINING_TIME>': 2,
    '<CASE_SEP>': 3, '<LABEL>': 4, '<QUERY>': 5, '<EVENT>': 6
}
ACTIVITY_VOCAB_SIZE = 10
TIME_BUCKET_VOCAB_SIZE = 50
VOCAB_SIZE = len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + TIME_BUCKET_VOCAB_SIZE


class EpisodeGenerator:
    def __init__(self, num_cases, max_case_len, num_cat_features, num_num_features):
        self.num_cases = num_cases
        self.max_case_len = max_case_len
        self.num_cat_features = num_cat_features
        self.num_num_features = num_num_features
        self.time_feat_dim = 2  # delta_t, absolute_time
        self.cat_cardinalities = [10] * num_cat_features
        self.dataset = self._generate_dataset()

    def _generate_dataset(self):
        dataset = []
        for _ in range(self.num_cases):
            case_len = random.randint(5, self.max_case_len)
            events = []
            current_time = 0
            for i in range(case_len):
                cat_feats = [random.randint(0, c - 1) for c in self.cat_cardinalities]
                num_feats = np.random.rand(self.num_num_features).tolist()
                delta_t = random.uniform(0.5, 5.0)
                current_time += delta_t
                time_feats = [np.log(delta_t + 1e-6), current_time / 100.0]  # Scaled time
                next_activity_label = random.randint(0, ACTIVITY_VOCAB_SIZE - 1)
                remaining_time = (case_len - 1 - i) * 2.5
                events.append({
                    'cat_feats': cat_feats, 'num_feats': num_feats, 'time_feats': time_feats,
                    'next_activity': next_activity_label, 'remaining_time': remaining_time
                })
            dataset.append(events)
        return dataset

    def _discretize_time(self, time_value):
        max_rem_time = self.max_case_len * 5.0
        bucket = int((time_value / max_rem_time) * (TIME_BUCKET_VOCAB_SIZE - 1))
        return min(max(bucket, 0), TIME_BUCKET_VOCAB_SIZE - 1)

    def create_episode(self, k_shots, task):
        sampled_cases = random.sample(self.dataset, k_shots + 1)
        support_cases, query_case = sampled_cases[:k_shots], sampled_cases[-1]

        episode_tokens = []
        # NEW: Store feature vectors corresponding to each token
        cat_feature_list, num_feature_list, time_feature_list = [], [], []

        # Helper to pad features for non-event tokens
        def pad_features():
            cat_feature_list.append([0] * self.num_cat_features)
            num_feature_list.append([0] * self.num_num_features)
            time_feature_list.append([0] * self.time_feat_dim)

        # Start with task token
        episode_tokens.append(SPECIAL_TOKENS[f'<TASK_{task.upper()}>'])
        pad_features()

        for case in support_cases:
            episode_tokens.append(SPECIAL_TOKENS['<CASE_SEP>'])
            pad_features()
            prefix_len = random.randint(2, len(case) - 1)
            for i in range(prefix_len):
                episode_tokens.append(SPECIAL_TOKENS['<EVENT>'])
                cat_feature_list.append(case[i]['cat_feats'])
                num_feature_list.append(case[i]['num_feats'])
                time_feature_list.append(case[i]['time_feats'])

            episode_tokens.append(SPECIAL_TOKENS['<LABEL>'])
            pad_features()

            if task == 'next_activity':
                label = case[prefix_len - 1]['next_activity']
                label_token = len(SPECIAL_TOKENS) + label
            else:
                label = case[prefix_len - 1]['remaining_time']
                label_token = len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + self._discretize_time(label)
            episode_tokens.append(label_token)
            pad_features()

        episode_tokens.append(SPECIAL_TOKENS['<CASE_SEP>'])
        pad_features()
        episode_tokens.append(SPECIAL_TOKENS['<QUERY>'])
        pad_features()

        query_prefix_len = random.randint(2, len(query_case) - 1)
        for i in range(query_prefix_len):
            episode_tokens.append(SPECIAL_TOKENS['<EVENT>'])
            cat_feature_list.append(query_case[i]['cat_feats'])
            num_feature_list.append(query_case[i]['num_feats'])
            time_feature_list.append(query_case[i]['time_feats'])

        episode_tokens.append(SPECIAL_TOKENS['<LABEL>'])
        pad_features()

        if task == 'next_activity':
            query_true_label_continuous = -1
            query_true_token = len(SPECIAL_TOKENS) + query_case[query_prefix_len - 1]['next_activity']
        else:
            query_true_label_continuous = query_case[query_prefix_len - 1]['remaining_time']
            query_true_token = len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + self._discretize_time(
                query_true_label_continuous)

        loss_mask = [0] * (len(episode_tokens) - 1) + [1]

        return {
            "tokens": episode_tokens, "loss_mask": loss_mask,
            "cat_feats": cat_feature_list, "num_feats": num_feature_list, "time_feats": time_feature_list,
            "query_true_token": query_true_token, "query_true_continuous": query_true_label_continuous,
            "task": task
        }


def collate_batch(batch):
    max_len = max(len(item['tokens']) for item in batch)

    padded_tokens, padded_loss_masks, attention_masks = [], [], []
    padded_cat, padded_num, padded_time = [], [], []

    num_cat_features = len(batch[0]['cat_feats'][0])
    num_num_features = len(batch[0]['num_feats'][0])
    num_time_features = len(batch[0]['time_feats'][0])

    for item in batch:
        pad_len = max_len - len(item['tokens'])

        padded_tokens.append(item['tokens'] + [SPECIAL_TOKENS['<PAD>']] * pad_len)
        padded_loss_masks.append(item['loss_mask'] + [0] * pad_len)
        attention_masks.append([1] * len(item['tokens']) + [0] * pad_len)

        # NEW: Pad the feature sequences
        padded_cat.append(item['cat_feats'] + [[0] * num_cat_features] * pad_len)
        padded_num.append(item['num_feats'] + [[0] * num_num_features] * pad_len)
        padded_time.append(item['time_feats'] + [[0] * num_time_features] * pad_len)

    return {
        "tokens": torch.tensor(padded_tokens, dtype=torch.long),
        "loss_mask": torch.tensor(padded_loss_masks, dtype=torch.bool),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.float),
        "cat_feats": torch.tensor(padded_cat, dtype=torch.long),
        "num_feats": torch.tensor(padded_num, dtype=torch.float),
        "time_feats": torch.tensor(padded_time, dtype=torch.float),
        "query_true_tokens": torch.tensor([item['query_true_token'] for item in batch], dtype=torch.long),
        "query_true_continuous": torch.tensor([item['query_true_continuous'] for item in batch], dtype=torch.float),
        "tasks": [item['task'] for item in batch]
    }
