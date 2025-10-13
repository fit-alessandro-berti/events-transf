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
    """
    Synthetic dataset generator with learnable structure.

    - First categorical feature = current activity in [0, ACTIVITY_VOCAB_SIZE-1].
    - Second categorical feature (if present) = segment in [0, n_segments-1].
    - A 'phase' (derived from normalized progress) influences transitions.
    - next_activity is a function of (current_activity, segment, phase) with light noise.
    - remaining_time correlates with normalized progress.

    This makes both tasks (next activity and remaining time) learnable from the prefix.
    """
    def __init__(self, num_cases, max_case_len, num_cat_features, num_num_features,
                 n_phases: int = 5, n_segments: int = 4, seed: int = 42):
        self.num_cases = num_cases
        self.max_case_len = max_case_len
        self.num_cat_features = num_cat_features
        self.num_num_features = num_num_features

        # Time feature dim is fixed at 2: [log_delta_t, normalized_progress]
        self.time_feat_dim = 2

        # Make the first categorical feature the current activity (cardinality = ACTIVITY_VOCAB_SIZE)
        # Remaining categorical features (if any) have cardinality 10.
        self.cat_cardinalities = [ACTIVITY_VOCAB_SIZE] + [10] * max(0, num_cat_features - 1)

        self.n_phases = n_phases
        self.n_segments = n_segments

        # Reproducibility
        random.seed(seed)
        np.random.seed(seed)

        self.dataset = self._generate_dataset()

    # --- Utilities ---

    def _transition(self, activity, segment, phase):
        """
        Deterministic + light-noise transition rule:
        next = (activity + step) % ACTIVITY_VOCAB_SIZE
        where step depends on segment and phase (both small integers).
        """
        base = 1 + (segment % 3)        # 1..3
        phase_effect = (phase % 2)      # 0 or 1
        step = base + phase_effect      # 1..4

        # small noise: 10% chance of adding +1 more step
        if random.random() < 0.10:
            step += 1
        return (activity + step) % ACTIVITY_VOCAB_SIZE

    def _discretize_time(self, time_value):
        max_rem_time = self.max_case_len * 3.0  # scale consistent with avg durations below
        bucket = int((time_value / max_rem_time) * (TIME_BUCKET_VOCAB_SIZE - 1))
        return min(max(bucket, 0), TIME_BUCKET_VOCAB_SIZE - 1)

    def _generate_dataset(self):
        dataset = []
        for _ in range(self.num_cases):
            case_len = random.randint(8, self.max_case_len)

            # Segment used to control transition dynamics & average durations
            segment = random.randint(0, self.n_segments - 1) if self.num_cat_features >= 2 else 0

            # Start activity
            activity = random.randint(0, ACTIVITY_VOCAB_SIZE - 1)

            # Average duration depends on segment (so remaining_time isn't just event count)
            avg_duration = 1.5 + 0.6 * segment  # 1.5, 2.1, 2.7, 3.3 for 4 segments

            current_time = 0.0
            events = []
            for i in range(case_len):
                # progress & phase
                denom = max(case_len - 1, 1)
                progress = i / denom  # 0..1
                phase = min(int(progress * self.n_phases), self.n_phases - 1)

                # delta time log-normal around segment-controlled average
                delta_t = float(np.random.lognormal(mean=np.log(avg_duration + 1e-3), sigma=0.25))
                current_time += delta_t

                # categorical features
                cat_feats = [activity]
                if self.num_cat_features >= 2:
                    cat_feats.append(segment)
                # fill any remaining categorical dims
                for _extra in range(self.num_cat_features - len(cat_feats)):
                    # use phase (mod 10) to make it semi-informative but bounded to 10
                    cat_feats.append(phase % 10)

                # numeric features (choose informative ones first)
                num_feats = [
                    progress,                      # directly informative for remaining_time
                    (phase + 1) / self.n_phases,   # coarse progress
                    (activity + 1) / ACTIVITY_VOCAB_SIZE  # weakly tied to next activity
                ]
                # pad/truncate to requested dimension
                if self.num_num_features <= len(num_feats):
                    num_feats = num_feats[:self.num_num_features]
                else:
                    num_feats = num_feats + [0.0] * (self.num_num_features - len(num_feats))

                # time features (log delta and normalized progress)
                time_feats = [np.log(delta_t + 1e-6), progress]

                # labels
                next_activity_label = self._transition(activity, segment, phase)
                remaining_time = (case_len - 1 - i) * avg_duration

                events.append({
                    'cat_feats': cat_feats,
                    'num_feats': num_feats,
                    'time_feats': time_feats,
                    'next_activity': next_activity_label,
                    'remaining_time': remaining_time
                })

                # advance activity for next event
                activity = next_activity_label

            dataset.append(events)
        return dataset

    # --- Episode building (IO-style) ---

    def create_episode(self, k_shots, task):
        """
        Builds an in-context episode:
          [<TASK>]  (support: ... <LABEL> y) x K  [<CASE_SEP>] [<QUERY>] (query: ... <LABEL>)
        NOTE: For the query, we DO NOT append the label token; the model must predict at <LABEL>.
        """
        sampled_cases = random.sample(self.dataset, k_shots + 1)
        support_cases, query_case = sampled_cases[:k_shots], sampled_cases[-1]

        episode_tokens = []
        cat_feature_list, num_feature_list, time_feature_list = [], [], []

        # Helper to pad features for non-event tokens
        def pad_features():
            cat_feature_list.append([0] * self.num_cat_features)
            num_feature_list.append([0.0] * self.num_num_features)
            time_feature_list.append([0.0] * self.time_feat_dim)

        # Start with task token
        episode_tokens.append(SPECIAL_TOKENS[f'<TASK_{task.upper()}>'])
        pad_features()

        # --- Support cases (K-shot) ---
        for case in support_cases:
            episode_tokens.append(SPECIAL_TOKENS['<CASE_SEP>']); pad_features()
            prefix_len = random.randint(2, len(case) - 1)

            for i in range(prefix_len):
                episode_tokens.append(SPECIAL_TOKENS['<EVENT>'])
                cat_feature_list.append(case[i]['cat_feats'])
                num_feature_list.append(case[i]['num_feats'])
                time_feature_list.append(case[i]['time_feats'])

            # place the label inline (IO-format)
            episode_tokens.append(SPECIAL_TOKENS['<LABEL>']); pad_features()
            if task == 'next_activity':
                label = case[prefix_len - 1]['next_activity']
                label_token = len(SPECIAL_TOKENS) + label
            else:
                label = case[prefix_len - 1]['remaining_time']
                label_token = len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + self._discretize_time(label)

            # append the concrete label token for support
            episode_tokens.append(label_token); pad_features()

        # --- Query case (no target token appended) ---
        episode_tokens.append(SPECIAL_TOKENS['<CASE_SEP>']); pad_features()
        episode_tokens.append(SPECIAL_TOKENS['<QUERY>']); pad_features()

        query_prefix_len = random.randint(2, len(query_case) - 1)
        for i in range(query_prefix_len):
            episode_tokens.append(SPECIAL_TOKENS['<EVENT>'])
            cat_feature_list.append(query_case[i]['cat_feats'])
            num_feature_list.append(query_case[i]['num_feats'])
            time_feature_list.append(query_case[i]['time_feats'])

        # place <LABEL> where the model must predict
        episode_tokens.append(SPECIAL_TOKENS['<LABEL>']); pad_features()

        # store true target (not appended to tokens)
        if task == 'next_activity':
            query_true_label_continuous = -1.0
            query_true_token = len(SPECIAL_TOKENS) + query_case[query_prefix_len - 1]['next_activity']
        else:
            query_true_label_continuous = query_case[query_prefix_len - 1]['remaining_time']
            query_true_token = len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + self._discretize_time(
                query_true_label_continuous)

        # loss only at the query <LABEL> position (the final token in this construction)
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

        padded_cat.append(item['cat_feats'] + [[0] * num_cat_features] * pad_len)
        padded_num.append(item['num_feats'] + [[0.0] * num_num_features] * pad_len)
        padded_time.append(item['time_feats'] + [[0.0] * num_time_features] * pad_len)

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
