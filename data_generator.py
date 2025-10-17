# data_generator.py
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
from time_transf import transform_time


class XESLogLoader:
    """
    Handles data loading and processing. It learns activity and resource maps
    from training data, generates initial embeddings using a pre-trained model,
    and maps unseen test activities to their closest semantic match using the
    Hungarian algorithm for optimal 1-to-1 assignment.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # --- Mappings and Embeddings learned from training data ---
        self.activity_to_id = {}
        self.resource_to_id = {}
        self.training_activity_names = []
        self.training_resource_names = []
        self.initial_activity_embeddings = None
        self.initial_resource_embeddings = None

        # Special IDs for padding
        self.PAD_TOKEN = '<PAD>'
        self.pad_activity_id = 0
        self.pad_resource_id = 0

        # Sentence transformer model for one-off embedding generation
        try:
            self.sbert_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"❌ Error initializing SentenceTransformer model '{model_name}': {e}")
            self.sbert_model = None

    def fit(self, training_log_paths: dict, activity_key='concept:name', resource_key='org:resource'):
        """
        Fits on training data to create activity/resource -> ID maps and stores
        initial embeddings for all known items.
        """
        if not self.sbert_model:
            raise RuntimeError("SentenceTransformer model not available for generating initial embeddings.")

        print("Fitting on training data to create vocab maps and initial embeddings...")
        all_activities = set()
        all_resources = set()

        for _, path in training_log_paths.items():
            if not os.path.exists(path): continue
            try:
                log = pm4py.read_xes(path)
                df = pm4py.convert_to_dataframe(log)
                all_activities.update(df[activity_key].unique())
                all_resources.update(df[resource_key].fillna('Unknown').unique())
            except Exception as e:
                print(f"❌ Error reading file {path} during fitting: {e}")

        if not all_activities: raise ValueError("No activities found in training logs.")
        if not all_resources: raise ValueError("No resources found in training logs.")

        # --- Create Activity Map and Embeddings ---
        # Add PAD token at index 0
        self.training_activity_names = sorted(list(all_activities))
        self.activity_to_id = {name: i + 1 for i, name in enumerate(self.training_activity_names)}
        self.activity_to_id[self.PAD_TOKEN] = self.pad_activity_id
        # Generate embeddings using SBERT
        print(f"  - Created map for {len(self.activity_to_id) - 1} unique activities.")
        print("  - Generating initial embeddings for activities...")
        sbert_activity_embeddings = self.sbert_model.encode(
            self.training_activity_names, show_progress_bar=True, normalize_embeddings=True
        )
        # Add a zero vector for the PAD token at index 0
        self.initial_activity_embeddings = np.vstack([
            np.zeros((1, sbert_activity_embeddings.shape[1])), sbert_activity_embeddings
        ]).astype(np.float32)

        # --- Create Resource Map and Embeddings ---
        self.training_resource_names = sorted(list(all_resources))
        self.resource_to_id = {name: i + 1 for i, name in enumerate(self.training_resource_names)}
        self.resource_to_id[self.PAD_TOKEN] = self.pad_resource_id
        # Generate embeddings using SBERT
        print(f"  - Created map for {len(self.resource_to_id) - 1} unique resources.")
        print("  - Generating initial embeddings for resources...")
        sbert_resource_embeddings = self.sbert_model.encode(
            self.training_resource_names, show_progress_bar=True, normalize_embeddings=True
        )
        # Add a zero vector for the PAD token at index 0
        self.initial_resource_embeddings = np.vstack([
            np.zeros((1, sbert_resource_embeddings.shape[1])), sbert_resource_embeddings
        ]).astype(np.float32)

        print("✅ Fit complete.")
        return self

    def _map_unseen_items(self, current_items, training_items, training_item_sbert_embs, item_to_id_map, item_type="activity"):
        """Helper to map unseen items using the Hungarian algorithm."""
        final_map = item_to_id_map.copy()
        unseen_items = [name for name in current_items if name not in item_to_id_map]

        if unseen_items:
            print(f"  - Found {len(unseen_items)} unseen {item_type} items. Mapping them to training vocabulary...")
            unseen_embeddings = self.sbert_model.encode(unseen_items, normalize_embeddings=True)

            similarity_matrix = cosine_similarity(unseen_embeddings, training_item_sbert_embs)
            cost_matrix = 1 - similarity_matrix

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            print(f"  - Optimal 1-to-1 mapping found for {len(row_ind)} {item_type} items.")
            for r, c in zip(row_ind, col_ind):
                unseen_name = unseen_items[r]
                mapped_name = training_items[c]
                final_map[unseen_name] = item_to_id_map[mapped_name]

        return final_map

    def transform(self, log_paths: dict, case_id_key='case:concept:name', activity_key='concept:name',
                  timestamp_key='time:timestamp', resource_key='org:resource', cost_key='amount'):
        """
        Loads and processes logs, converting activity/resource names to IDs.
        Unseen items are mapped to the most semantically similar training item.
        """
        if not self.sbert_model: raise RuntimeError("SentenceTransformer model not available.")
        if not self.activity_to_id: raise RuntimeError("Loader has not been fitted.")

        print(f"\nTransforming logs: {list(log_paths.keys())}")

        all_dfs = [pm4py.convert_to_dataframe(pm4py.read_xes(path)) for path in log_paths.values() if os.path.exists(path)]
        if not all_dfs: return {}
        combined_df = pd.concat(all_dfs, keys=log_paths.keys(), names=['log_name', 'orig_index']).reset_index()

        # --- Map Activities and Resources to their IDs ---
        current_activities = sorted(list(combined_df[activity_key].unique()))
        current_resources = sorted(list(combined_df[resource_key].fillna('Unknown').unique()))

        # Get SBERT embeddings for the original training items (excluding PAD)
        training_activity_sbert_embs = self.initial_activity_embeddings[1:]
        training_resource_sbert_embs = self.initial_resource_embeddings[1:]

        final_activity_id_map = self._map_unseen_items(
            current_activities, self.training_activity_names, training_activity_sbert_embs, self.activity_to_id, "activity"
        )
        final_resource_id_map = self._map_unseen_items(
            current_resources, self.training_resource_names, training_resource_sbert_embs, self.resource_to_id, "resource"
        )

        # --- Process traces using the final ID maps ---
        processed_logs = {}
        for name, group_df in combined_df.groupby('log_name'):
            raw_traces = self._convert_df_to_raw_traces(group_df, case_id_key, activity_key, timestamp_key, resource_key, cost_key)
            processed_logs[name] = self._apply_ids_to_traces(raw_traces, final_activity_id_map, final_resource_id_map)

        print("✅ Transformation complete.")
        return processed_logs

    def _apply_ids_to_traces(self, raw_traces, activity_map, resource_map):
        log_with_ids = []
        for raw_trace in raw_traces:
            processed_trace = []
            for event in raw_trace:
                processed_event = {
                    'activity_id': activity_map.get(event['activity'], self.pad_activity_id),
                    'resource_id': resource_map.get(event['resource'], self.pad_resource_id),
                    'cost': event['cost'],
                    'time_from_start': event['time_from_start'],
                    'time_from_previous': event['time_from_previous'],
                    'timestamp': event['timestamp'], 'case_id': event['case_id']
                }
                processed_trace.append(processed_event)
            log_with_ids.append(processed_trace)
        return log_with_ids

    def _convert_df_to_raw_traces(self, df, case_id_key, activity_key, timestamp_key, resource_key, cost_key):
        # This function is mostly the same, just filling 'Unknown' for resource.
        raw_log = []
        df[timestamp_key] = pd.to_datetime(df[timestamp_key]).dt.tz_localize(None)
        df[resource_key] = df[resource_key].fillna('Unknown')

        for case_id, trace_df in df.groupby(case_id_key):
            trace_df = trace_df.sort_values(by=timestamp_key)
            if trace_df.empty: continue
            trace = []
            start_time = trace_df.iloc[0][timestamp_key]
            prev_time = start_time
            for _, event in trace_df.iterrows():
                current_time = event[timestamp_key]
                cost_val = event.get(cost_key, round(random.uniform(5.0, 100.0), 2))
                if not isinstance(cost_val, (int, float)): cost_val = round(random.uniform(5.0, 100.0), 2)

                event_dict = {
                    'case_id': case_id, 'activity': event[activity_key],
                    'timestamp': current_time.timestamp(), 'resource': event[resource_key],
                    'cost': cost_val, 'time_from_start': (current_time - start_time).total_seconds(),
                    'time_from_previous': (current_time - prev_time).total_seconds(),
                }
                trace.append(event_dict)
                prev_time = current_time
            if trace: raw_log.append(trace)
        return raw_log

    def save_training_artifacts(self, path):
        """Saves the essential training artifacts needed for testing."""
        artifacts = {
            'activity_to_id': self.activity_to_id,
            'resource_to_id': self.resource_to_id,
            'training_activity_names': self.training_activity_names,
            'training_resource_names': self.training_resource_names,
            'initial_activity_embeddings': self.initial_activity_embeddings,
            'initial_resource_embeddings': self.initial_resource_embeddings
        }
        torch.save(artifacts, path)
        print(f"💾 Training artifacts saved to {path}")

    def load_training_artifacts(self, path):
        """Loads artifacts, required for stand-alone testing."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifacts file not found at {path}.")

        artifacts = torch.load(path, weights_only=False)

        self.activity_to_id = artifacts['activity_to_id']
        self.resource_to_id = artifacts['resource_to_id']
        self.training_activity_names = artifacts['training_activity_names']
        self.training_resource_names = artifacts['training_resource_names']
        self.initial_activity_embeddings = artifacts['initial_activity_embeddings']
        self.initial_resource_embeddings = artifacts['initial_resource_embeddings']
        print(f"✅ Training artifacts loaded successfully from {path}")


def get_task_data(log, task_type, max_seq_len=10):
    tasks = []
    if not log: return tasks

    for trace in log:
        if len(trace) < 3: continue
        for i in range(1, len(trace) - 1):
            prefix = trace[:i + 1]
            if len(prefix) > max_seq_len:
                prefix = prefix[-max_seq_len:]

            if task_type == 'classification':
                # The label is the ID of the next activity
                label = trace[i + 1]['activity_id']
                tasks.append((prefix, label))
            elif task_type == 'regression':
                last_event_time = trace[-1]['timestamp']
                current_event_time = prefix[-1]['timestamp']
                remaining_time_hours = (last_event_time - current_event_time) / 3600.0
                label = transform_time(remaining_time_hours)
                tasks.append((prefix, label))
    return tasks
