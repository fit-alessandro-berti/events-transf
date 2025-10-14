
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from utils import EpisodeBatch, SPECIAL_TOKENS, TOKEN_TYPES, make_causal_mask


@dataclass
class NumericScaler:
    mean: float
    std: float
    def encode(self, x: np.ndarray) -> np.ndarray:
        s = self.std if self.std > 1e-6 else 1.0
        return (x - self.mean) / s


class EpisodeBuilder:
    def __init__(self, traces, activity2id: Dict[str, int], resource2id: Dict[str, int], max_seq_len: int = 512, seed: int = 123):
        self.rng = random.Random(seed)
        self.traces = traces
        self.activity2id = activity2id
        self.resource2id = resource2id
        self.max_seq_len = max_seq_len

        costs = []
        deltas = []
        for tr in self.traces:
            if not tr.events:
                continue
            for i, e in enumerate(tr.events):
                costs.append(e.cost)
                if i == 0:
                    dt = 0.0
                else:
                    dt = e.timestamp - tr.events[i - 1].timestamp
                deltas.append(dt)
        import numpy as np
        self.cost_scaler = NumericScaler(float(np.mean(costs)), float(np.std(costs) + 1e-6))
        self.logdt_scaler = NumericScaler(float(np.mean(np.log1p(deltas))), float(np.std(np.log1p(deltas)) + 1e-6))

    def _encode_event(self, tr, idx, abs_anchor_min: float):
        e = tr.events[idx]
        T = len(tr.events)
        act_id = e.activity_id
        res_id = e.resource_id
        cost_z = self.cost_scaler.encode(np.array([e.cost], dtype=np.float32))[0]
        if idx == 0:
            dt = 0.0
        else:
            dt = e.timestamp - tr.events[idx - 1].timestamp
        logdt = np.log1p(dt)
        logdt_z = self.logdt_scaler.encode(np.array([logdt], dtype=np.float32))[0]
        pos_frac = idx / max(T - 1, 1)
        # hour-of-day and dow from an arbitrary anchor
        hour = int(((abs_anchor_min // 60) + e.timestamp // 60) % 24)
        dow = int(((abs_anchor_min // (24 * 60)) + e.timestamp // (24 * 60)) % 7)
        num_feats = [float(cost_z), float(pos_frac)]
        time_feats = [float(logdt_z), float(hour / 23.0), float(dow / 6.0)]
        return act_id, res_id, num_feats, time_feats

    def _append_event_token(self, buffers, tr, idx, seg_id, abs_anchor_min):
        act_id, res_id, num_feats, time_feats = self._encode_event(tr, idx, abs_anchor_min)
        buffers["activity_ids"].append(act_id)
        buffers["resource_ids"].append(res_id)
        buffers["num_feats"].append(num_feats)
        buffers["time_feats"].append(time_feats)
        buffers["token_types"].append(TOKEN_TYPES["EVENT"])
        buffers["special_ids"].append(0)
        buffers["class_label_ids"].append(0)
        buffers["num_label_values"].append(0.0)
        buffers["segment_ids"].append(seg_id)
        buffers["label_pos_mask"].append(0)
        buffers["query_label_pos_mask"].append(0)

    def _append_special(self, buffers, name: str, seg_id: int):
        buffers["activity_ids"].append(0)
        buffers["resource_ids"].append(0)
        buffers["num_feats"].append([0.0, 0.0])
        buffers["time_feats"].append([0.0, 0.0, 0.0])
        buffers["token_types"].append(TOKEN_TYPES["SPECIAL"])
        buffers["special_ids"].append(SPECIAL_TOKENS[name])
        buffers["class_label_ids"].append(0)
        buffers["num_label_values"].append(0.0)
        buffers["segment_ids"].append(seg_id)
        buffers["label_pos_mask"].append(1 if name == "LABEL" else 0)
        buffers["query_label_pos_mask"].append(0)

    def _append_class_label(self, buffers, class_id: int, seg_id: int):
        buffers["activity_ids"].append(0)
        buffers["resource_ids"].append(0)
        buffers["num_feats"].append([0.0, 0.0])
        buffers["time_feats"].append([0.0, 0.0, 0.0])
        buffers["token_types"].append(TOKEN_TYPES["CLS_LABEL"])
        buffers["special_ids"].append(0)
        buffers["class_label_ids"].append(class_id)
        buffers["num_label_values"].append(0.0)
        buffers["segment_ids"].append(seg_id)
        buffers["label_pos_mask"].append(0)
        buffers["query_label_pos_mask"].append(0)

    def _append_num_label(self, buffers, value: float, seg_id: int):
        buffers["activity_ids"].append(0)
        buffers["resource_ids"].append(0)
        buffers["num_feats"].append([0.0, 0.0])
        buffers["time_feats"].append([0.0, 0.0, 0.0])
        buffers["token_types"].append(TOKEN_TYPES["NUM_LABEL"])
        buffers["special_ids"].append(0)
        buffers["class_label_ids"].append(0)
        buffers["num_label_values"].append(float(value))
        buffers["segment_ids"].append(seg_id)
        buffers["label_pos_mask"].append(0)
        buffers["query_label_pos_mask"].append(0)

    def build_episode(self, task: str, K: int, device: torch.device) -> EpisodeBatch:
        # task in {"cls", "reg"}
        abs_anchor_min = self.rng.uniform(0, 60 * 24 * 365)
        q_tr = self.rng.choice(self.traces)
        if len(q_tr.events) < 2:
            return self.build_episode(task, K, device)

        q_idx = self.rng.randint(0, len(q_tr.events) - 2)

        buffers = {k: [] for k in ["activity_ids","resource_ids","num_feats","time_feats",
                                   "token_types","special_ids","class_label_ids","num_label_values",
                                   "segment_ids","label_pos_mask","query_label_pos_mask"]}

        # [TASK]
        self._append_special(buffers, "TASK_NEXT_ACTIVITY" if task=="cls" else "TASK_REMAINING_TIME", seg_id=0)
        # <CASE_SEP>
        self._append_special(buffers, "CASE_SEP", seg_id=0)

        supports = []
        for _ in range(K):
            tr = self.rng.choice(self.traces)
            if len(tr.events) < 2:
                continue
            s_idx = self.rng.randint(0, len(tr.events) - 2)
            supports.append((tr, s_idx))

        for tr, s_idx in supports:
            for i in range(s_idx + 1):
                self._append_event_token(buffers, tr, i, seg_id=0, abs_anchor_min=abs_anchor_min)
            self._append_special(buffers, "LABEL", seg_id=0)
            if task == "cls":
                self._append_class_label(buffers, tr.next_activity_ids[s_idx], seg_id=0)
            else:
                self._append_num_label(buffers, tr.remaining_times[s_idx], seg_id=0)
            self._append_special(buffers, "CASE_SEP", seg_id=0)

        # <QUERY>
        self._append_special(buffers, "QUERY", seg_id=1)
        for i in range(q_idx + 1):
            self._append_event_token(buffers, q_tr, i, seg_id=1, abs_anchor_min=abs_anchor_min)
        self._append_special(buffers, "LABEL", seg_id=1)
        buffers["query_label_pos_mask"][-1] = 1

        y_cls, y_reg = None, None
        if task == "cls":
            y_cls = torch.tensor([q_tr.next_activity_ids[q_idx]], dtype=torch.long, device=device)
        else:
            y_reg = torch.tensor([q_tr.remaining_times[q_idx]], dtype=torch.float32, device=device)

        T = len(buffers["token_types"])
        if T > self.max_seq_len:
            start = T - self.max_seq_len
            for k in list(buffers.keys()):
                buffers[k] = buffers[k][start:]
            T = self.max_seq_len

        import numpy as np
        def to_tensor(name, dtype):
            arr = np.array(buffers[name])
            return torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0)

        return EpisodeBatch(
            activity_ids=to_tensor("activity_ids", torch.long),
            resource_ids=to_tensor("resource_ids", torch.long),
            num_feats=to_tensor("num_feats", torch.float32),
            time_feats=to_tensor("time_feats", torch.float32),
            token_types=to_tensor("token_types", torch.long),
            special_ids=to_tensor("special_ids", torch.long),
            class_label_ids=to_tensor("class_label_ids", torch.long),
            num_label_values=to_tensor("num_label_values", torch.float32),
            segment_ids=to_tensor("segment_ids", torch.long),
            label_pos_mask=to_tensor("label_pos_mask", torch.long),
            query_label_pos_mask=to_tensor("query_label_pos_mask", torch.long),
            y_cls=y_cls, y_reg=y_reg,
            attn_mask=make_causal_mask(T, device=device),
        )

    def build_batch(self, task: str, K: int, batch_size: int, device: torch.device) -> EpisodeBatch:
        batches = [self.build_episode(task, K, device) for _ in range(batch_size)]
        def stack(attr):
            return torch.cat([getattr(b, attr) for b in batches], dim=0)
        return EpisodeBatch(
            activity_ids=stack("activity_ids"),
            resource_ids=stack("resource_ids"),
            num_feats=stack("num_feats"),
            time_feats=stack("time_feats"),
            token_types=stack("token_types"),
            special_ids=stack("special_ids"),
            class_label_ids=stack("class_label_ids"),
            num_label_values=stack("num_label_values"),
            segment_ids=stack("segment_ids"),
            label_pos_mask=stack("label_pos_mask"),
            query_label_pos_mask=stack("query_label_pos_mask"),
            y_cls=torch.cat([b.y_cls if b.y_cls is not None else torch.zeros((1,), dtype=torch.long, device=device) for b in batches], dim=0) if batches[0].y_cls is not None else None,
            y_reg=torch.cat([b.y_reg if b.y_reg is not None else torch.zeros((1,), dtype=torch.float32, device=device) for b in batches], dim=0) if batches[0].y_reg is not None else None,
            attn_mask=batches[0].attn_mask,
        )

    def build_causal_batch(self, batch_size: int, device: torch.device):
        Bs = []
        for _ in range(batch_size):
            tr = self.rng.choice(self.traces)
            if len(tr.events) < 2:
                continue
            L = self.rng.randint(2, min(len(tr.events), self.max_seq_len))
            abs_anchor_min = self.rng.uniform(0, 60 * 24 * 365)
            buffers = {k: [] for k in ["activity_ids","resource_ids","num_feats","time_feats",
                                       "token_types","special_ids","class_label_ids","num_label_values",
                                       "segment_ids","label_pos_mask","query_label_pos_mask"]}
            for i in range(L):
                self._append_event_token(buffers, tr, i, seg_id=0, abs_anchor_min=abs_anchor_min)
            import numpy as np
            def to_tensor(name, dtype):
                arr = np.array(buffers[name])
                return torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0)
            b = EpisodeBatch(
                activity_ids=to_tensor("activity_ids", torch.long),
                resource_ids=to_tensor("resource_ids", torch.long),
                num_feats=to_tensor("num_feats", torch.float32),
                time_feats=to_tensor("time_feats", torch.float32),
                token_types=to_tensor("token_types", torch.long),
                special_ids=to_tensor("special_ids", torch.long),
                class_label_ids=to_tensor("class_label_ids", torch.long),
                num_label_values=to_tensor("num_label_values", torch.float32),
                segment_ids=to_tensor("segment_ids", torch.long),
                label_pos_mask=to_tensor("label_pos_mask", torch.long),
                query_label_pos_mask=to_tensor("query_label_pos_mask", torch.long),
                y_cls=None, y_reg=None,
                attn_mask=make_causal_mask(L, device=device),
            )
            next_acts = [tr.next_activity_ids[i] for i in range(L - 1)]
            rem_times = [tr.remaining_times[i] for i in range(L - 1)]
            b.next_cls_targets = torch.tensor(next_acts, dtype=torch.long, device=device).unsqueeze(0)
            b.next_reg_targets = torch.tensor(rem_times, dtype=torch.float32, device=device).unsqueeze(0)
            Bs.append(b)
        def stack(attr):
            return torch.cat([getattr(b, attr) for b in Bs], dim=0)
        out = EpisodeBatch(
            activity_ids=stack("activity_ids"),
            resource_ids=stack("resource_ids"),
            num_feats=stack("num_feats"),
            time_feats=stack("time_feats"),
            token_types=stack("token_types"),
            special_ids=stack("special_ids"),
            class_label_ids=stack("class_label_ids"),
            num_label_values=stack("num_label_values"),
            segment_ids=stack("segment_ids"),
            label_pos_mask=stack("label_pos_mask"),
            query_label_pos_mask=stack("query_label_pos_mask"),
            y_cls=None, y_reg=None,
            attn_mask=Bs[0].attn_mask,
        )
        out.next_cls_targets = torch.cat([b.next_cls_targets for b in Bs], dim=0)
        out.next_reg_targets = torch.cat([b.next_reg_targets for b in Bs], dim=0)
        return out
