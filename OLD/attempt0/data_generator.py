# data_generator.py
#
# Synthetic process-mining style data generator for training.
# Adds DENSE per-event targets to strengthen supervision:
#   - dense_next_targets / dense_time_targets with masks to avoid query leakage.

import math
import random
from typing import List, Dict, Any

import numpy as np
import torch

# ----------------------------
# Vocabulary / Special Tokens
# ----------------------------
SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<TASK_NEXT_ACTIVITY>': 1,
    '<TASK_REMAINING_TIME>': 2,
    '<CASE_SEP>': 3,
    '<LABEL>': 4,
    '<QUERY>': 5,
    '<EVENT>': 6,
}
ACTIVITY_VOCAB_SIZE = 10
TIME_BUCKET_VOCAB_SIZE = 50
VOCAB_SIZE = len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + TIME_BUCKET_VOCAB_SIZE


def get_reverse_vocab():
    rev_vocab = {v: k for k, v in SPECIAL_TOKENS.items()}
    for i in range(ACTIVITY_VOCAB_SIZE):
        rev_vocab[len(SPECIAL_TOKENS) + i] = f"ACTIVITY_{i}"
    for i in range(TIME_BUCKET_VOCAB_SIZE):
        rev_vocab[len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + i] = f"TIME_BUCKET_{i}"
    return rev_vocab


# ----------------------------
# Simulation model
# ----------------------------
class ProcSimModel:
    def __init__(
        self,
        variant_id: int,
        num_activities: int = ACTIVITY_VOCAB_SIZE,
        num_resources: int = 12,
        num_groups: int = 4,
        seed: int = 0,
    ):
        self.rng = np.random.RandomState(seed + 17 * (variant_id + 1))
        self.A = num_activities
        self.R = num_resources
        self.G = num_groups

        self.resource_group = np.array([(r + variant_id) % self.G for r in range(self.R)], dtype=int)
        self.trans_mat = self._build_transition_matrix(variant_id)

        base_dur = 1.2 + 0.25 * self.rng.rand(self.A) + 0.15 * (np.arange(self.A) % 3)
        self.dur_means = base_dur * (1.0 + 0.15 * variant_id)
        self.dur_sigma = 0.25

        base_amt = 80.0 + 40.0 * self.rng.rand(self.A) + 15.0 * (np.arange(self.A) % 4)
        self.amt_means = base_amt * (1.0 + 0.10 * variant_id)
        self.amt_sigma = 0.40

        self.res_pref = self._build_resource_preferences(variant_id)

    def _build_transition_matrix(self, variant_id: int) -> np.ndarray:
        A = self.A
        mat = np.zeros((A, A), dtype=float)
        fwd = 0.60 + 0.05 * self.rng.rand()
        skip = 0.12 + 0.03 * (variant_id % 3)
        self_loop = 0.04 + 0.02 * self.rng.rand()
        for i in range(A):
            mat[i, (i + 1) % A] += fwd
            mat[i, (i + 2) % A] += skip
            mat[i, i] += self_loop
            others = [j for j in range(A) if j not in {(i + 1) % A, (i + 2) % A, i}]
            self.rng.shuffle(others)
            w_remaining = 1.0 - (fwd + skip + self_loop)
            w1 = w_remaining * (0.35 + 0.3 * self.rng.rand())
            w2 = w_remaining - w1
            mat[i, others[0]] += w1
            mat[i, others[1]] += w2
        mat = mat / mat.sum(axis=1, keepdims=True)
        return mat

    def _build_resource_preferences(self, variant_id: int) -> np.ndarray:
        A, R, G = self.A, self.R, self.G
        res_group = self.resource_group
        prefs = np.zeros((A, R), dtype=float)
        for a in range(A):
            preferred_group = (a + variant_id) % G
            base = np.full(R, 0.1)
            base[res_group == preferred_group] = 0.7
            noise = 0.05 * self.rng.rand(R)
            w = base + noise
            w = np.maximum(w, 1e-6)
            prefs[a] = w / w.sum()
        return prefs

    def _sample_next_activity(self, a_curr: int) -> int:
        return int(self.rng.choice(self.A, p=self.trans_mat[a_curr]))

    def _sample_resource_for_activity(self, a: int) -> int:
        return int(self.rng.choice(self.R, p=self.res_pref[a]))

    def _sample_delta_t(self, a: int) -> float:
        mu = math.log(self.dur_means[a] + 1e-6)
        return float(self.rng.lognormal(mean=mu, sigma=self.dur_sigma))

    def _sample_amount(self, a: int) -> float:
        mu = math.log(self.amt_means[a] + 1e-6)
        return float(self.rng.lognormal(mean=mu, sigma=self.amt_sigma))

    def generate_trace(self, max_len: int) -> List[Dict[str, Any]]:
        L = self.rng.randint(8, max(9, max_len + 1))
        a = int(self.rng.randint(0, self.A))
        events = []
        for _ in range(L):
            r = self._sample_resource_for_activity(a)
            g = int(self.resource_group[r])
            delta_t = self._sample_delta_t(a)
            amount = self._sample_amount(a)
            events.append({
                "activity": a,
                "resource": r,
                "group": g,
                "delta_t": delta_t,
                "amount": amount,
            })
            a = self._sample_next_activity(a)
        return events


class EpisodeGenerator:
    """
    Dataset made of multiple ProcSimModel variants. Categorical features:
    [activity, resource, group]; numeric: [amount, amount_norm, ...]; time: [log_dt, progress].
    Adds dense per-event supervision to accelerate/regularize training.
    """
    def __init__(
        self,
        num_cases: int,
        max_case_len: int,
        num_cat_features: int,   # >=3 (activity, resource, group)
        num_num_features: int,   # >=1 (amount + optional)
        n_models: int = 3,
        num_resources: int = 12,
        num_groups: int = 4,
        seed: int = 42,
    ):
        assert num_cat_features >= 3, "num_cat_features must be >= 3 (activity, resource, group)"
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.num_cases = num_cases
        self.max_case_len = max_case_len
        self.num_cat_features = 3
        self.num_num_features = num_num_features
        self.time_feat_dim = 2

        self.num_resources = num_resources
        self.num_groups = num_groups
        self.cat_cardinalities = [ACTIVITY_VOCAB_SIZE, self.num_resources, self.num_groups]

        self.models = [
            ProcSimModel(
                variant_id=i,
                num_activities=ACTIVITY_VOCAB_SIZE,
                num_resources=self.num_resources,
                num_groups=self.num_groups,
                seed=seed + 100 * i,
            )
            for i in range(n_models)
        ]

        self.dataset = self._generate_dataset()
        self._finalize_time_buckets()

    def _build_num_feats(self, amount: float, case_amount_mean: float) -> List[float]:
        feats = [amount]
        if self.num_num_features > 1:
            feats.append(amount / (case_amount_mean + 1e-8))
        if self.num_num_features > len(feats):
            feats += [0.0] * (self.num_num_features - len(feats))
        return feats[: self.num_num_features]

    def _generate_dataset(self) -> List[List[Dict[str, Any]]]:
        dataset: List[List[Dict[str, Any]]] = []
        per_model = max(1, self.num_cases // max(1, len(self.models)))
        for m in self.models:
            for _ in range(per_model):
                raw = m.generate_trace(self.max_case_len)
                if len(raw) < 3:
                    continue
                total_time = sum(ev["delta_t"] for ev in raw)
                cum_time = 0.0
                case_amount_mean = float(np.mean([ev["amount"] for ev in raw]))

                events = []
                for i, ev in enumerate(raw):
                    progress = i / (len(raw) - 1)
                    cum_time += ev["delta_t"]
                    remaining_time = max(total_time - cum_time, 0.0)
                    next_activity = raw[i + 1]["activity"] if i + 1 < len(raw) else ev["activity"]

                    cat_feats = [ev["activity"], ev["resource"], ev["group"]]
                    num_feats = self._build_num_feats(ev["amount"], case_amount_mean)
                    time_feats = [float(np.log(ev["delta_t"] + 1e-6)), progress]

                    events.append({
                        "cat_feats": cat_feats,
                        "num_feats": num_feats,
                        "time_feats": time_feats,
                        "next_activity": next_activity,
                        "remaining_time": remaining_time,
                    })
                dataset.append(events)
        if len(dataset) == 0:
            raise RuntimeError("No traces generated; reduce constraints or check simulator.")
        return dataset

    def _finalize_time_buckets(self):
        all_rem = [ev["remaining_time"] for case in self.dataset for ev in case]
        q95 = float(np.percentile(all_rem, 95)) if len(all_rem) > 0 else self.max_case_len * 3.0
        self.max_remaining_time = max(q95, 1e-3)

    def _discretize_time(self, time_value: float) -> int:
        b = int((time_value / self.max_remaining_time) * (TIME_BUCKET_VOCAB_SIZE - 1))
        return max(0, min(TIME_BUCKET_VOCAB_SIZE - 1, b))

    def bucket_to_continuous(self, bucket: int) -> float:
        bucket = max(0, min(TIME_BUCKET_VOCAB_SIZE - 1, int(bucket)))
        width = self.max_remaining_time / TIME_BUCKET_VOCAB_SIZE
        return (bucket + 0.5) * width

    def create_episode(self, k_shots: int, task: str) -> Dict[str, Any]:
        """
        [<TASK>] (support K cases with inline <LABEL> y)  [<CASE_SEP>] [<QUERY>] (prefix ... <LABEL>)
        Also returns dense per-event targets (next activity & remaining time bucket).
        """
        import copy

        sampled_cases = random.sample(self.dataset, k_shots + 1)
        support_cases, query_case = sampled_cases[:k_shots], sampled_cases[-1]

        episode_tokens: List[int] = []
        cat_feature_list: List[List[int]] = []
        num_feature_list: List[List[float]] = []
        time_feature_list: List[List[float]] = []

        # Dense supervision holders
        dense_next_targets: List[int] = []
        dense_time_targets: List[int] = []
        dense_mask_next: List[int] = []
        dense_mask_time: List[int] = []

        def pad_features():
            cat_feature_list.append([0] * self.num_cat_features)
            num_feature_list.append([0.0] * self.num_num_features)
            time_feature_list.append([0.0] * self.time_feat_dim)

        def pad_dense():
            dense_next_targets.append(0)
            dense_time_targets.append(0)
            dense_mask_next.append(0)
            dense_mask_time.append(0)

        # Task token
        episode_tokens.append(SPECIAL_TOKENS[f'<TASK_{task.upper()}>'])
        pad_features(); pad_dense()

        # --- support cases
        for case in support_cases:
            episode_tokens.append(SPECIAL_TOKENS['<CASE_SEP>']); pad_features(); pad_dense()
            prefix_len = random.randint(2, len(case) - 1)

            for i in range(prefix_len):
                ev = case[i]
                episode_tokens.append(SPECIAL_TOKENS['<EVENT>'])
                cat_feature_list.append(ev['cat_feats'])
                num_feature_list.append(ev['num_feats'])
                time_feature_list.append(ev['time_feats'])

                # dense labels at every event in support
                dense_next_targets.append(ev['next_activity'])
                dense_time_targets.append(self._discretize_time(ev['remaining_time']))
                dense_mask_next.append(1)
                dense_mask_time.append(1)

            episode_tokens.append(SPECIAL_TOKENS['<LABEL>']); pad_features(); pad_dense()
            if task == 'next_activity':
                label = case[prefix_len - 1]['next_activity']
                label_token = len(SPECIAL_TOKENS) + label
            else:
                rem = case[prefix_len - 1]['remaining_time']
                label_token = len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + self._discretize_time(rem)
            episode_tokens.append(label_token); pad_features(); pad_dense()

        # --- query case
        episode_tokens.append(SPECIAL_TOKENS['<CASE_SEP>']); pad_features(); pad_dense()
        episode_tokens.append(SPECIAL_TOKENS['<QUERY>']); pad_features(); pad_dense()

        q_prefix_len = random.randint(2, len(query_case) - 1)
        for i in range(q_prefix_len):
            ev = query_case[i]
            episode_tokens.append(SPECIAL_TOKENS['<EVENT>'])
            cat_feature_list.append(ev['cat_feats'])
            num_feature_list.append(ev['num_feats'])
            time_feature_list.append(ev['time_feats'])

            # dense supervision on query prefix EXCEPT the last event to avoid leaking query target
            is_last = (i == q_prefix_len - 1)
            dense_next_targets.append(ev['next_activity'])
            dense_time_targets.append(self._discretize_time(ev['remaining_time']))
            dense_mask_next.append(0 if is_last else 1)
            dense_mask_time.append(0 if is_last else 1)

        episode_tokens.append(SPECIAL_TOKENS['<LABEL>']); pad_features(); pad_dense()

        if task == 'next_activity':
            query_true_cont = -1.0
            query_true_token = len(SPECIAL_TOKENS) + query_case[q_prefix_len - 1]['next_activity']
        else:
            query_true_cont = query_case[q_prefix_len - 1]['remaining_time']
            bucket = self._discretize_time(query_true_cont)
            query_true_token = len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + bucket

        loss_mask = [0] * (len(episode_tokens) - 1) + [1]

        return {
            "tokens": episode_tokens,
            "loss_mask": loss_mask,
            "cat_feats": cat_feature_list,
            "num_feats": num_feature_list,
            "time_feats": time_feature_list,
            "query_true_token": query_true_token,
            "query_true_continuous": query_true_cont,
            "task": task,
            # dense supervision
            "dense_next_targets": dense_next_targets,
            "dense_time_targets": dense_time_targets,
            "dense_mask_next": dense_mask_next,
            "dense_mask_time": dense_mask_time,
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_len = max(len(item['tokens']) for item in batch)

    padded_tokens, padded_loss_masks, attention_masks = [], [], []
    padded_cat, padded_num, padded_time = [], [], []

    num_cat_features = len(batch[0]['cat_feats'][0])
    num_num_features = len(batch[0]['num_feats'][0])
    num_time_features = len(batch[0]['time_feats'][0])

    # Optional dense supervision fields (present in training generator)
    has_dense = ('dense_next_targets' in batch[0])

    if has_dense:
        pad_dense_next = []
        pad_dense_time = []
        pad_mask_next = []
        pad_mask_time = []

    for item in batch:
        pad_len = max_len - len(item['tokens'])

        padded_tokens.append(item['tokens'] + [SPECIAL_TOKENS['<PAD>']] * pad_len)
        padded_loss_masks.append(item['loss_mask'] + [0] * pad_len)
        attention_masks.append([1] * len(item['tokens']) + [0] * pad_len)

        padded_cat.append(item['cat_feats'] + [[0] * num_cat_features] * pad_len)
        padded_num.append(item['num_feats'] + [[0.0] * num_num_features] * pad_len)
        padded_time.append(item['time_feats'] + [[0.0] * num_time_features] * pad_len)

        if has_dense:
            pad_dense_next.append(item['dense_next_targets'] + [0] * pad_len)
            pad_dense_time.append(item['dense_time_targets'] + [0] * pad_len)
            pad_mask_next.append(item['dense_mask_next'] + [0] * pad_len)
            pad_mask_time.append(item['dense_mask_time'] + [0] * pad_len)

    out = {
        "tokens": torch.tensor(padded_tokens, dtype=torch.long),
        "loss_mask": torch.tensor(padded_loss_masks, dtype=torch.bool),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.float),
        "cat_feats": torch.tensor(padded_cat, dtype=torch.long),
        "num_feats": torch.tensor(padded_num, dtype=torch.float),
        "time_feats": torch.tensor(padded_time, dtype=torch.float),
        "query_true_tokens": torch.tensor([item['query_true_token'] for item in batch], dtype=torch.long),
        "query_true_continuous": torch.tensor([item['query_true_continuous'] for item in batch], dtype=torch.float),
        "tasks": [item['task'] for item in batch],
    }

    if has_dense:
        out.update({
            "dense_next_targets": torch.tensor(pad_dense_next, dtype=torch.long),
            "dense_time_targets": torch.tensor(pad_dense_time, dtype=torch.long),
            "dense_mask_next": torch.tensor(pad_mask_next, dtype=torch.bool),
            "dense_mask_time": torch.tensor(pad_mask_time, dtype=torch.bool),
        })

    return out


if __name__ == '__main__':
    def print_episode_details(episode, rev_vocab):
        print("-" * 80)
        print(f"TASK: {episode['task']}")
        print(f"TOTAL TOKENS: {len(episode['tokens'])}")
        print("-" * 80)

        for i, token_id in enumerate(episode['tokens']):
            token_name = rev_vocab.get(token_id, f"ID_{token_id}")
            loss_marker = " <--- LOSS" if episode['loss_mask'][i] == 1 else ""
            print(f"[{i:03d}] Token: {token_name:<25}{loss_marker}")
            if token_name == '<EVENT>':
                cat_f = episode['cat_feats'][i]
                num_f = [f"{x:.2f}" for x in episode['num_feats'][i]]
                time_f = [f"{x:.2f}" for x in episode['time_feats'][i]]
                print(f"      - Cat Feats [act,res,grp]: {cat_f}")
                print(f"      - Num Feats [amount,...] : {num_f}")
                print(f"      - Time Feats [log_dt, p] : {time_f}")

        true_token_id = episode['query_true_token']
        true_token_name = rev_vocab.get(true_token_id, f"ID_{true_token_id}")
        print("-" * 80)
        print(f"QUERY GROUND TRUTH TOKEN: {true_token_name} (ID: {true_token_id})")
        if episode['query_true_continuous'] != -1.0:
            print(f"QUERY GROUND TRUTH (Continuous): {episode['query_true_continuous']:.2f}")
        print("-" * 80)

    print("\n" + "=" * 25 + " GENERATING SAMPLE EPISODES " + "=" * 25)
    K_SHOTS = 2
    NUM_CAT_FEATURES = 3
    NUM_NUM_FEATURES = 3

    gen = EpisodeGenerator(
        num_cases=60,
        max_case_len=24,
        num_cat_features=NUM_CAT_FEATURES,
        num_num_features=NUM_NUM_FEATURES,
        n_models=4,
        seed=7,
    )
    rev = get_reverse_vocab()

    ep1 = gen.create_episode(K_SHOTS, 'next_activity')
    ep2 = gen.create_episode(K_SHOTS, 'remaining_time')

    print("\n--- EXAMPLE 1: NEXT ACTIVITY ---")
    print_episode_details(ep1, rev)
    print("\n--- EXAMPLE 2: REMAINING TIME ---")
    print_episode_details(ep2, rev)
