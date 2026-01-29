# File: data_generator.py
import pandas as pd
import random
import pm4py
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

# --- Import from project files ---
from config import CONFIG
# 🔻 Removed: from time_transf import transform_time (no longer needed here)


class XESLogLoader:
    """
    Handles data loading and processing based on the chosen embedding strategy.
    - 'pretrained': Generates fixed semantic embeddings using a pre-trained model.
    - 'learned': Creates a character vocabulary to generate embeddings on-the-fly.
    """

    def __init__(self, strategy: str, sbert_model_name: str = 'all-MiniLM-L6-v2'):
        self.strategy = strategy
        print(f"Data loader initialized with strategy: '{self.strategy}'")

        # --- Artifacts learned from training data ---
        self.activity_to_id = {}  # For classification labels (pretrained) & evaluation
        self.char_to_id = {}  # For learned character embeddings
        self.training_activity_names = []
        self.training_activity_embeddings = None

        # --- 🔻 NEW: Caches for pretrained embeddings 🔻 ---
        self.activity_embedding_map = {}
        self.resource_embedding_map = {}
        # --- 🔺 END NEW 🔺 ---

        # Special tokens for vocabularies
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.pad_id = 0
        self.unk_id = 1  # For characters

        self.sbert_model = None
        if self.strategy == 'pretrained':
            try:
                self.sbert_model = SentenceTransformer(sbert_model_name)
                self.sbert_embedding_dim = self.sbert_model.get_sentence_embedding_dimension()
            except Exception as e:
                raise RuntimeError(f"Pretrained strategy requires SentenceTransformer: {e}")
            self.pad_embedding = np.zeros(self.sbert_embedding_dim, dtype=np.float32)

    def fit(self, training_log_paths: dict, activity_key='concept:name', resource_key='org:resource'):
        """
        Fits on training data to create either embeddings or vocabularies.
        """
        print(f"Fitting on training data (strategy: '{self.strategy}')...")
        all_activities, all_resources = set(), set()
        for _, path in training_log_paths.items():
            if not os.path.exists(path): continue
            try:
                df = pm4py.convert_to_dataframe(pm4py.read_xes(path, variant="iterparse"))
                all_activities.update(df[activity_key].unique())
                all_resources.update(df[resource_key].fillna('Unknown').unique())
            except Exception as e:
                print(f"❌ Error reading file {path}: {e}")

        if not all_activities: raise ValueError("No activities found in training logs.")
        self.training_activity_names = sorted(list(all_activities))
        all_resource_names = sorted(list(all_resources))  # 🔻 --- NEW --- 🔻

        # This map is now only strictly necessary for the 'pretrained' strategy and the
        # embedding quality evaluation, but we create it always for consistency.
        self.activity_to_id = {name: i for i, name in enumerate(self.training_activity_names)}

        if self.strategy == 'learned':
            all_names = all_activities.union(all_resources)
            all_chars = set("".join(all_names))
            self.char_to_id = {char: i + 2 for i, char in enumerate(sorted(list(all_chars)))}
            self.char_to_id[self.PAD_TOKEN] = self.pad_id
            self.char_to_id[self.UNK_TOKEN] = self.unk_id
            print(f"  - Created character vocabulary of size {len(self.char_to_id)}.")
        elif self.strategy == 'pretrained':
            print("  - Generating and storing embeddings for all training activities...")
            self.training_activity_embeddings = self.sbert_model.encode(
                self.training_activity_names, show_progress_bar=True, normalize_embeddings=True
            )
            # --- 🔻 NEW: Populate activity embedding map 🔻 ---
            self.activity_embedding_map = {name: emb for name, emb in zip(
                self.training_activity_names, self.training_activity_embeddings
            )}

            # --- 🔻 NEW: Encode and store resource embeddings 🔻 ---
            print("  - Generating and storing embeddings for all training resources...")
            resource_embeddings = self.sbert_model.encode(
                all_resource_names, show_progress_bar=True, normalize_embeddings=True
            )
            self.resource_embedding_map = {name: emb for name, emb in zip(
                all_resource_names, resource_embeddings
            )}
            # --- 🔺 END NEW 🔺 ---

        print("✅ Fit complete.")
        return self

    def transform(self, log_paths: dict, case_id_key='case:concept:name', activity_key='concept:name',
                  timestamp_key='time:timestamp', resource_key='org:resource', cost_key='amount'):
        if not self.training_activity_names: raise RuntimeError("Loader has not been fitted.")
        print(f"\nTransforming logs: {list(log_paths.keys())}")
        all_dfs = [pm4py.convert_to_dataframe(pm4py.read_xes(path)) for path in log_paths.values() if
                   os.path.exists(path)]
        if not all_dfs: return {}
        combined_df = pd.concat(all_dfs, keys=log_paths.keys(), names=['log_name', 'orig_index']).reset_index()
        processed_logs = {}
        for name, group_df in combined_df.groupby('log_name'):
            raw_traces = self._convert_df_to_raw_traces(group_df, case_id_key, activity_key, timestamp_key,
                                                        resource_key, cost_key)
            if self.strategy == 'learned':
                processed_logs[name] = self._transform_learned(raw_traces)
            else:
                processed_logs[name] = self._transform_pretrained(group_df, raw_traces, activity_key, resource_key)
        print("✅ Transformation complete.")
        return processed_logs

    def _transform_learned(self, raw_traces):
        """
        Passes raw strings and creates a dynamic, local mapping for labels.
        This ensures that even unseen logs can be used for classification tasks.
        """
        # --- NEW: Create a local label map for this specific log ---
        all_activities_in_log = set(event['activity'] for trace in raw_traces for event in trace)
        local_activity_to_id = {name: i for i, name in enumerate(sorted(list(all_activities_in_log)))}
        # -----------------------------------------------------------

        log_with_strings = []
        for raw_trace in raw_traces:
            processed_trace = []
            for event in raw_trace:
                processed_event = {
                    'activity_name': event['activity'],
                    'resource_name': event['resource'],
                    # Use the new local map for the classification label
                    'activity_id': local_activity_to_id.get(event['activity']),
                    'cost': event['cost'],
                    'time_from_start': event['time_from_start'],
                    'time_from_previous': event['time_from_previous'],
                    'timestamp': event['timestamp'], 'case_id': event['case_id']
                }
                processed_trace.append(processed_event)
            log_with_strings.append(processed_trace)
        return log_with_strings

    def _convert_df_to_raw_traces(self, df, case_id_key, activity_key, timestamp_key, resource_key, cost_key):
        raw_log = []
        df[timestamp_key] = pd.to_datetime(df[timestamp_key]).dt.tz_localize(None)
        df[resource_key] = df[resource_key].fillna('Unknown')
        for case_id, trace_df in df.groupby(case_id_key):
            trace_df = trace_df.sort_values(by=timestamp_key)
            if trace_df.empty: continue
            trace, start_time, prev_time = [], trace_df.iloc[0][timestamp_key], trace_df.iloc[0][timestamp_key]
            for _, event in trace_df.iterrows():
                current_time = event[timestamp_key]
                cost_val = event.get(cost_key, round(random.uniform(5.0, 100.0), 2))
                if not isinstance(cost_val, (int, float)): cost_val = round(random.uniform(5.0, 100.0), 2)
                event_dict = {
                    'case_id': case_id, 'activity': event[activity_key], 'timestamp': current_time.timestamp(),
                    'resource': event[resource_key], 'cost': cost_val,
                    'time_from_start': (current_time - start_time).total_seconds(),
                    'time_from_previous': (current_time - prev_time).total_seconds(),
                }
                trace.append(event_dict)
                prev_time = current_time
            if trace: raw_log.append(trace)
        return raw_log

    def _transform_pretrained(self, df, raw_traces, activity_key, resource_key):
        # --- 🔻 MODIFIED: Start with cached maps 🔻 ---
        activity_embedding_map = self.activity_embedding_map.copy()
        resource_embedding_map = self.resource_embedding_map.copy()

        current_activities = sorted(list(df[activity_key].unique()))
        resources_in_log = sorted(list(df[resource_key].fillna('Unknown').unique()))

        final_activity_id_map = self.activity_to_id.copy()

        # --- Handle UNSEEN activities (for classification ID mapping) ---
        unseen_activities_for_id_map = [name for name in current_activities if name not in self.activity_to_id]
        if unseen_activities_for_id_map:
            unseen_id_embeddings = self.sbert_model.encode(unseen_activities_for_id_map, normalize_embeddings=True)
            similarity_matrix = cosine_similarity(unseen_id_embeddings, self.training_activity_embeddings)
            row_ind, col_ind = linear_sum_assignment(1 - similarity_matrix)
            for r, c in zip(row_ind, col_ind):
                final_activity_id_map[unseen_activities_for_id_map[r]] = self.activity_to_id[
                    self.training_activity_names[c]]

        # --- Handle UNSEEN activities (for embedding map) ---
        unseen_activities_for_embed = [name for name in current_activities if name not in activity_embedding_map]
        if unseen_activities_for_embed:
            print(f"  - Encoding {len(unseen_activities_for_embed)} new activity embeddings...")
            new_act_embs = self.sbert_model.encode(unseen_activities_for_embed, normalize_embeddings=True)
            for name, emb in zip(unseen_activities_for_embed, new_act_embs):
                activity_embedding_map[name] = emb

        # --- Handle UNSEEN resources (for embedding map) ---
        unseen_resources_for_embed = [name for name in resources_in_log if name not in resource_embedding_map]
        if unseen_resources_for_embed:
            print(f"  - Encoding {len(unseen_resources_for_embed)} new resource embeddings...")
            new_res_embs = self.sbert_model.encode(unseen_resources_for_embed, normalize_embeddings=True)
            for name, emb in zip(unseen_resources_for_embed, new_res_embs):
                resource_embedding_map[name] = emb
        # --- 🔺 END MODIFIED 🔺 ---

        log_with_embeddings = []
        unknown_resource_emb = resource_embedding_map.get('Unknown', self.pad_embedding)
        for raw_trace in raw_traces:
            processed_trace = []
            for event in raw_trace:
                processed_trace.append({
                    'activity_embedding': activity_embedding_map.get(event['activity'], self.pad_embedding),
                    'resource_embedding': resource_embedding_map.get(event['resource'], unknown_resource_emb),
                    'activity_id': final_activity_id_map.get(event['activity'], -100),
                    'cost': event['cost'], 'time_from_start': event['time_from_start'],
                    'time_from_previous': event['time_from_previous'],
                    'timestamp': event['timestamp'], 'case_id': event['case_id']
                })
            log_with_embeddings.append(processed_trace)
        return log_with_embeddings

    def save_training_artifacts(self, path):
        artifacts = {'strategy': self.strategy, 'activity_to_id': self.activity_to_id,
                     'training_activity_names': self.training_activity_names}
        if self.strategy == 'pretrained':
            artifacts['training_activity_embeddings'] = self.training_activity_embeddings
        elif self.strategy == 'learned':
            artifacts['char_to_id'] = self.char_to_id
        torch.save(artifacts, path)
        print(f"💾 Training artifacts for '{self.strategy}' strategy saved to {path}")

    def load_training_artifacts(self, path):
        if not os.path.exists(path): raise FileNotFoundError(f"Artifacts file not found at {path}.")
        artifacts = torch.load(path, weights_only=False)
        if artifacts['strategy'] != self.strategy:
            raise ValueError(
                f"Artifact strategy '{artifacts['strategy']}' does not match loader strategy '{self.strategy}'.")
        self.activity_to_id = artifacts['activity_to_id']
        self.training_activity_names = artifacts['training_activity_names']
        if self.strategy == 'pretrained':
            self.training_activity_embeddings = artifacts['training_activity_embeddings']
        elif self.strategy == 'learned':
            self.char_to_id = artifacts['char_to_id']
        print(f"✅ Training artifacts loaded successfully from {path}")
