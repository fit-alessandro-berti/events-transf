# test_data_generator.py
#
# XES-based data generator for evaluation/inference.
# Reads an XES event log via pm4py and converts each trace to the same
# IO-format used by the training pipeline. Features include:
#   - categorical: [activity, resource, org_group]
#   - numeric: [amount, ... (padded)]
#   - time: [log_delta_t, normalized_progress]
#
# Notes:
#  - Activities are mapped into a fixed ACTIVITY_VOCAB_SIZE. By default the
#    last index is an OOV/OTHER bucket so you can evaluate logs with many
#    activities against a fixed head size.
#  - Resources and groups take their natural cardinality from the log.
#  - Remaining time is computed from actual timestamps.

from typing import List, Dict, Any, Optional
import math
import random
import numpy as np

# pm4py import (support both "read_xes" and the classic importer)
try:
    import pm4py
    _HAS_PM4PY = True
except Exception:  # pragma: no cover
    _HAS_PM4PY = False

from data_generator import (
    SPECIAL_TOKENS,
    ACTIVITY_VOCAB_SIZE,
    TIME_BUCKET_VOCAB_SIZE,
)

class XesEpisodeGenerator:
    def __init__(
        self,
        xes_path: str,
        num_num_features: int,
        activity_key: str = "concept:name",
        timestamp_key: str = "time:timestamp",
        resource_key: str = "org:resource",
        group_key: str = "org:group",
        amount_keys: Optional[List[str]] = None,     # e.g. ["amount", "cost", "value"]
        reserve_oov_activity: bool = True,
        seed: int = 123,
    ):
        if not _HAS_PM4PY:
            raise ImportError(
                "pm4py is required for XES ingestion. Install with `pip install pm4py`."
            )
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.num_num_features = num_num_features
        self.time_feat_dim = 2

        self.activity_key = activity_key
        self.timestamp_key = timestamp_key
        self.resource_key = resource_key
        self.group_key = group_key
        self.amount_keys = amount_keys or ["amount", "cost", "value"]

        # Load log (support both call patterns across pm4py versions)
        try:
            log = pm4py.read_xes(xes_path)
        except Exception:
            from pm4py.objects.log.importer.xes import importer as xes_importer
            log = xes_importer.apply(xes_path)

        # Extract raw traces (skip super-short ones)
        traces = []
        for trace in log:
            events = list(trace)
            # ensure chronological order
            events.sort(key=lambda e: e.get(self.timestamp_key))
            if len(events) >= 3:
                traces.append(events)

        if len(traces) == 0:
            raise RuntimeError("No sufficiently long traces (>=3 events) found in the XES log.")

        # Build mappings
        self._build_activity_mapping(traces, reserve_oov_activity)
        self._build_resource_group_mappings(traces)

        # Build dataset in the required structure
        self.dataset = self._convert_traces(traces)
        self._finalize_time_buckets()

        # cat cardinalities: [activity, resource, group]
        self.cat_cardinalities = [ACTIVITY_VOCAB_SIZE, self.num_resources, self.num_groups]
        self.num_cat_features = 3

    # ---- mappings ----
    def _build_activity_mapping(self, traces, reserve_oov: bool):
        # Frequency count
        freq: Dict[str, int] = {}
        for evs in traces:
            for e in evs:
                a = str(e.get(self.activity_key, ""))
                freq[a] = freq.get(a, 0) + 1
        # Rank by frequency
        items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        if reserve_oov and ACTIVITY_VOCAB_SIZE >= 2:
            keep = ACTIVITY_VOCAB_SIZE - 1
            kept = [a for a, _ in items[:keep]]
            self.activity_oov_idx = ACTIVITY_VOCAB_SIZE - 1
        else:
            keep = ACTIVITY_VOCAB_SIZE
            kept = [a for a, _ in items[:keep]]
            self.activity_oov_idx = None

        self.activity_to_idx: Dict[str, int] = {}
        for i, a in enumerate(kept):
            self.activity_to_idx[a] = i

    def _build_resource_group_mappings(self, traces):
        res_set, grp_set = set(), set()
        for evs in traces:
            for e in evs:
                r = e.get(self.resource_key)
                g = e.get(self.group_key)
                if r is not None:
                    res_set.add(str(r))
                if g is not None:
                    grp_set.add(str(g))
        self.resource_to_idx = {r: i for i, r in enumerate(sorted(res_set))}
        self.group_to_idx = {g: i for i, g in enumerate(sorted(grp_set))}
        self.num_resources = max(1, len(self.resource_to_idx))
        self.num_groups = max(1, len(self.group_to_idx))

    def _map_activity(self, a_str: str) -> int:
        if a_str in self.activity_to_idx:
            return self.activity_to_idx[a_str]
        if self.activity_oov_idx is not None:
            return self.activity_oov_idx
        # fallback (hash into the fixed space)
        return hash(a_str) % ACTIVITY_VOCAB_SIZE

    def _map_resource(self, r) -> int:
        if r is None:
            return 0
        return self.resource_to_idx.get(str(r), 0)

    def _map_group(self, g) -> int:
        if g is None:
            return 0
        return self.group_to_idx.get(str(g), 0)

    # ---- conversion ----
    def _event_amount(self, e) -> float:
        for k in self.amount_keys:
            v = e.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                continue
        return 0.0

    def _convert_traces(self, traces) -> List[List[Dict[str, Any]]]:
        dataset = []
        for evs in traces:
            # deltas and totals from timestamps (seconds -> hours for scale)
            deltas = []
            for i, e in enumerate(evs):
                if i == 0:
                    deltas.append(0.0)
                    continue
                t_curr = e.get(self.timestamp_key)
                t_prev = evs[i - 1].get(self.timestamp_key)
                if t_curr is None or t_prev is None:
                    dt_sec = 0.0
                else:
                    dt_sec = (t_curr - t_prev).total_seconds()
                deltas.append(max(0.0, float(dt_sec) / 3600.0))  # hours

            amounts = [self._event_amount(e) for e in evs]
            amount_mean = float(np.mean(amounts)) if len(amounts) else 0.0

            total_time = float(sum(deltas))
            cum = 0.0
            converted: List[Dict[str, Any]] = []
            for i, e in enumerate(evs):
                a = self._map_activity(str(e.get(self.activity_key, "")))
                r = self._map_resource(e.get(self.resource_key))
                g = self._map_group(e.get(self.group_key))
                dt = deltas[i]
                cum += dt
                progress = i / (len(evs) - 1)
                remaining = max(total_time - cum, 0.0)
                next_a = self._map_activity(str(evs[i + 1].get(self.activity_key, ""))) if i + 1 < len(evs) else a

                num_feats = [amounts[i]]
                if self.num_num_features > 1:
                    num_feats.append(amounts[i] / (amount_mean + 1e-8))
                if self.num_num_features > len(num_feats):
                    num_feats += [0.0] * (self.num_num_features - len(num_feats))
                num_feats = num_feats[: self.num_num_features]

                converted.append({
                    "cat_feats": [a, r, g],
                    "num_feats": num_feats,
                    "time_feats": [float(np.log(dt + 1e-6)), progress],
                    "next_activity": int(next_a),
                    "remaining_time": float(remaining),
                })
            dataset.append(converted)
        return dataset

    def _finalize_time_buckets(self):
        all_rem = [ev["remaining_time"] for case in self.dataset for ev in case]
        q95 = float(np.percentile(all_rem, 95)) if len(all_rem) > 0 else 1.0
        self.max_remaining_time = max(q95, 1e-3)

    def _discretize_time(self, t: float) -> int:
        b = int((t / self.max_remaining_time) * (TIME_BUCKET_VOCAB_SIZE - 1))
        return max(0, min(TIME_BUCKET_VOCAB_SIZE - 1, b))

    def bucket_to_continuous(self, bucket: int) -> float:
        bucket = max(0, min(TIME_BUCKET_VOCAB_SIZE - 1, int(bucket)))
        width = self.max_remaining_time / TIME_BUCKET_VOCAB_SIZE
        return (bucket + 0.5) * width

    # ---- episode builder (same as training format) ----
    def create_episode(self, k_shots: int, task: str) -> Dict[str, Any]:
        sampled_cases = random.sample(self.dataset, k_shots + 1)
        support_cases, query_case = sampled_cases[:k_shots], sampled_cases[-1]

        episode_tokens: List[int] = []
        cat_feature_list: List[List[int]] = []
        num_feature_list: List[List[float]] = []
        time_feature_list: List[List[float]] = []

        def pad_features():
            cat_feature_list.append([0] * 3)
            num_feature_list.append([0.0] * self.num_num_features)
            time_feature_list.append([0.0] * self.time_feat_dim)

        episode_tokens.append(SPECIAL_TOKENS[f'<TASK_{task.upper()}>'])
        pad_features()

        for case in support_cases:
            episode_tokens.append(SPECIAL_TOKENS['<CASE_SEP>']); pad_features()
            prefix_len = random.randint(2, len(case) - 1)

            for i in range(prefix_len):
                episode_tokens.append(SPECIAL_TOKENS['<EVENT>'])
                cat_feature_list.append(case[i]['cat_feats'])
                num_feature_list.append(case[i]['num_feats'])
                time_feature_list.append(case[i]['time_feats'])

            episode_tokens.append(SPECIAL_TOKENS['<LABEL>']); pad_features()
            if task == 'next_activity':
                label = case[prefix_len - 1]['next_activity']
                label_token = len(SPECIAL_TOKENS) + label
            else:
                rem = case[prefix_len - 1]['remaining_time']
                label_token = len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE + self._discretize_time(rem)
            episode_tokens.append(label_token); pad_features()

        episode_tokens.append(SPECIAL_TOKENS['<CASE_SEP>']); pad_features()
        episode_tokens.append(SPECIAL_TOKENS['<QUERY>']); pad_features()

        q_prefix_len = random.randint(2, len(query_case) - 1)
        for i in range(q_prefix_len):
            episode_tokens.append(SPECIAL_TOKENS['<EVENT>'])
            cat_feature_list.append(query_case[i]['cat_feats'])
            num_feature_list.append(query_case[i]['num_feats'])
            time_feature_list.append(query_case[i]['time_feats'])

        episode_tokens.append(SPECIAL_TOKENS['<LABEL>']); pad_features()

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
        }
